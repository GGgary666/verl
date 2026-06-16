# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

from verl.utils.qat.calibration import is_scale_uninitialized
from verl.utils.qat.core import QATConfig, apply_qat
from verl.utils.qat.moe import (
    _apply_qat_moe_layers,
    _is_fused_moe_experts,
    fake_quantize_weight_2d,
    qwen3_fused_experts_forward,
)


class _MiniFusedExperts(nn.Module):
    def __init__(self, num_experts=4, hidden_dim=32, intermediate_dim=16):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_dim
        self.hidden_dim = hidden_dim
        self.gate_up_proj = nn.Parameter(torch.randn(num_experts, 2 * intermediate_dim, hidden_dim))
        self.down_proj = nn.Parameter(torch.randn(num_experts, hidden_dim, intermediate_dim))
        self.act_fn = F.silu


class _MiniMoEBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = _MiniFusedExperts()
        self.top_k = 2

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq, hidden = hidden_states.shape
        flat = hidden_states.view(-1, hidden)
        num_tokens = flat.shape[0]
        router_logits = torch.randn(num_tokens, self.experts.num_experts, device=flat.device, dtype=flat.dtype)
        routing_weights = torch.softmax(router_logits, dim=-1)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        out = self.experts(flat, selected_experts, routing_weights)
        return out.view(batch, seq, hidden)


def test_is_fused_moe_experts():
    experts = _MiniFusedExperts()
    assert _is_fused_moe_experts(experts)


def test_apply_qat_attaches_moe_buffers():
    model = _MiniMoEBlock()
    config = QATConfig(enable=True, mode="w4a4", group_size=16)
    apply_qat(model, config)
    experts = model.experts
    assert hasattr(experts, "w13_input_global_scale")
    assert hasattr(experts, "w2_input_global_scale")
    assert is_scale_uninitialized(experts.w13_input_global_scale)


def test_moe_forward_initializes_scales_in_training():
    if not torch.cuda.is_available():
        pytest_skip = __import__("pytest").skip
        pytest_skip("CUDA required for Triton FP4 fake quant")

    model = _MiniMoEBlock().cuda()
    config = QATConfig(enable=True, mode="w4a4", group_size=16)
    apply_qat(model, config)
    model.train()

    x = torch.randn(2, 4, 32, device="cuda", dtype=torch.float32)
    _ = model(x)

    assert not is_scale_uninitialized(model.experts.w13_input_global_scale)
    assert not is_scale_uninitialized(model.experts.w2_input_global_scale)


def test_fake_quant_weight_2d_shape():
    if not torch.cuda.is_available():
        pytest_skip = __import__("pytest").skip
        pytest_skip("CUDA required for Triton FP4 fake quant")

    weight = torch.randn(32, 32, device="cuda")
    fq = fake_quantize_weight_2d(weight, group_size=16)
    assert fq.shape == weight.shape


def test_apply_qat_moe_has_no_weight_cache():
    experts = _MiniFusedExperts()
    config = QATConfig(enable=True, mode="w4a4")
    _apply_qat_moe_layers(experts, config)
    assert not hasattr(experts, "_qat_moe_weight_cache")


def test_qwen3_forward_matches_loop_semantics():
    if not torch.cuda.is_available():
        pytest_skip = __import__("pytest").skip
        pytest_skip("CUDA required for Triton FP4 fake quant")

    experts = _MiniFusedExperts().cuda()
    config = QATConfig(enable=True, mode="w4a4", group_size=16)
    _apply_qat_moe_layers(experts, config)
    experts.train()

    hidden = torch.randn(8, 32, device="cuda")
    top_k = 2
    router_logits = torch.randn(8, experts.num_experts, device="cuda")
    routing_weights, top_k_index = torch.topk(torch.softmax(router_logits, dim=-1), top_k, dim=-1)

    out_qat = qwen3_fused_experts_forward(experts, hidden, top_k_index, routing_weights)
    assert out_qat.shape == hidden.shape
