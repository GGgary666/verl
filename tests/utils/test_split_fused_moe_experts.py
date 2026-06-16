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

from verl.utils.model import (
    align_actor_weight_names_for_vllm,
    prepare_moe_weights_for_vllm_rollout,
    split_fused_moe_experts,
)


def _collect(weights, hidden_size=None, moe_intermediate_size=None):
    return dict(split_fused_moe_experts(weights, hidden_size, moe_intermediate_size))


def test_split_stacked_gate_up_proj_transformers5_layout():
    # Use hidden != 2*inter so stacked vs legacy layouts are unambiguous.
    hidden, inter, experts = 8, 3, 2
    gate_up = torch.arange(experts * 2 * inter * hidden, dtype=torch.float32).reshape(experts, 2 * inter, hidden)
    prefix = "model.layers.0.mlp.experts.gate_up_proj"
    out = _collect([(prefix, gate_up)], hidden_size=hidden, moe_intermediate_size=inter)

    assert set(out) == {
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.experts.1.gate_proj.weight",
        "model.layers.0.mlp.experts.1.up_proj.weight",
    }
    assert out["model.layers.0.mlp.experts.0.gate_proj.weight"].shape == (inter, hidden)
    assert out["model.layers.0.mlp.experts.0.up_proj.weight"].shape == (inter, hidden)
    torch.testing.assert_close(out["model.layers.0.mlp.experts.0.gate_proj.weight"], gate_up[0, :inter])
    torch.testing.assert_close(out["model.layers.0.mlp.experts.0.up_proj.weight"], gate_up[0, inter:])


def test_split_legacy_gate_up_proj_transformers457_layout():
    hidden, inter, experts = 8, 4, 2
    gate_up = torch.arange(experts * hidden * 2 * inter, dtype=torch.float32).reshape(experts, hidden, 2 * inter)
    prefix = "model.layers.0.mlp.experts.gate_up_proj.weight"
    out = _collect([(prefix, gate_up)], hidden_size=hidden, moe_intermediate_size=inter)

    assert out["model.layers.0.mlp.experts.0.gate_proj.weight"].shape == (inter, hidden)
    assert out["model.layers.0.mlp.experts.0.up_proj.weight"].shape == (inter, hidden)
    torch.testing.assert_close(out["model.layers.0.mlp.experts.0.gate_proj.weight"], gate_up[0, :, :inter].T)
    torch.testing.assert_close(out["model.layers.0.mlp.experts.0.up_proj.weight"], gate_up[0, :, inter:].T)


def test_split_stacked_down_proj_emits_hidden_by_inter():
    hidden, inter, experts = 8, 4, 2
    down = torch.arange(experts * hidden * inter, dtype=torch.float32).reshape(experts, hidden, inter)
    prefix = "model.layers.0.mlp.experts.down_proj"
    out = _collect([(prefix, down)], hidden_size=hidden, moe_intermediate_size=inter)

    assert out["model.layers.0.mlp.experts.0.down_proj.weight"].shape == (hidden, inter)
    torch.testing.assert_close(out["model.layers.0.mlp.experts.0.down_proj.weight"], down[0])


def test_split_legacy_down_proj_transposes_to_hidden_by_inter():
    hidden, inter, experts = 8, 4, 2
    down = torch.arange(experts * inter * hidden, dtype=torch.float32).reshape(experts, inter, hidden)
    prefix = "model.layers.0.mlp.experts.down_proj.weight"
    out = _collect([(prefix, down)], hidden_size=hidden, moe_intermediate_size=inter)

    assert out["model.layers.0.mlp.experts.0.down_proj.weight"].shape == (hidden, inter)
    torch.testing.assert_close(out["model.layers.0.mlp.experts.0.down_proj.weight"], down[0].T)


def test_split_gate_up_proj_weight_suffix():
    hidden, inter, experts = 8, 3, 2
    gate_up = torch.arange(experts * 2 * inter * hidden, dtype=torch.float32).reshape(experts, 2 * inter, hidden)
    prefix = "model.layers.0.mlp.experts.gate_up_proj.weight"
    out = _collect([(prefix, gate_up)], hidden_size=hidden, moe_intermediate_size=inter)
    assert "model.layers.0.mlp.experts.0.gate_proj.weight" in out


def test_shared_expert_gate_up_proj_not_split():
    # shared_expert gate_up is 2D; must not be touched by the routed-expert splitter.
    tensor = torch.randn(6, 8)
    out = _collect([("model.layers.0.mlp.shared_expert.gate_up_proj.weight", tensor)])
    assert out["model.layers.0.mlp.shared_expert.gate_up_proj.weight"] is tensor


def test_non_moe_tensor_passthrough():
    tensor = torch.ones(2, 3, 4)
    out = _collect([("other.weight", tensor)])
    assert out["other.weight"] is tensor


def test_split_stacked_gate_proj_without_config_sizes():
    hidden, inter, experts = 8, 3, 2
    gate = torch.arange(experts * inter * hidden, dtype=torch.float32).reshape(experts, inter, hidden)
    prefix = "model.layers.0.mlp.experts.gate_proj"
    out = _collect([(prefix, gate)])
    assert out["model.layers.0.mlp.experts.0.gate_proj.weight"].shape == (inter, hidden)
    torch.testing.assert_close(out["model.layers.0.mlp.experts.0.gate_proj.weight"], gate[0])


def test_split_gate_up_proj_base_layer_weight_suffix():
    hidden, inter, experts = 8, 3, 2
    gate_up = torch.arange(experts * 2 * inter * hidden, dtype=torch.float32).reshape(experts, 2 * inter, hidden)
    prefix = "model.layers.0.mlp.experts.gate_up_proj.base_layer.weight"
    out = _collect([(prefix, gate_up)], hidden_size=hidden, moe_intermediate_size=inter)
    assert "model.layers.0.mlp.experts.0.gate_proj.weight" in out
    assert out["model.layers.0.mlp.experts.0.gate_proj.weight"].shape == (inter, hidden)


def test_split_gate_up_heuristic_without_config_sizes():
    hidden, inter, experts = 8, 3, 2
    gate_up = torch.arange(experts * 2 * inter * hidden, dtype=torch.float32).reshape(experts, 2 * inter, hidden)
    prefix = "model.layers.0.mlp.experts.gate_up_proj"
    out = _collect([(prefix, gate_up)])
    assert "model.layers.0.mlp.experts.0.gate_proj.weight" in out
    assert out["model.layers.0.mlp.experts.0.gate_proj.weight"].shape == (inter, hidden)


class _DummyVllmLmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.model = nn.Linear(2, 2)


class _DummyVllmCausalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(2, 2)


def test_align_actor_weight_names_for_vllm_hybrid_prefix():
    tensor = torch.ones(2)
    vllm_model = _DummyVllmLmModel()
    out = dict(
        align_actor_weight_names_for_vllm(
            [
                ("model.layers.0.mlp.experts.gate_up_proj", tensor),
                ("model.language_model.layers.0.weight", tensor),
                ("model.visual.blocks.0.weight", tensor),
            ],
            vllm_model,
        )
    )
    assert out["language_model.model.layers.0.mlp.experts.gate_up_proj"] is tensor
    assert out["model.language_model.layers.0.weight"] is tensor
    assert out["model.visual.blocks.0.weight"] is tensor


def test_align_actor_weight_names_for_vllm_causal_passthrough():
    tensor = torch.ones(2)
    vllm_model = _DummyVllmCausalModel()
    out = dict(
        align_actor_weight_names_for_vllm([("model.layers.0.mlp.experts.gate_up_proj", tensor)], vllm_model)
    )
    assert out["model.layers.0.mlp.experts.gate_up_proj"] is tensor


def test_prepare_moe_weights_for_vllm_rollout_splits_fused_experts():
    hidden, inter, experts = 8, 3, 2
    gate_up = torch.arange(experts * 2 * inter * hidden, dtype=torch.float32).reshape(experts, 2 * inter, hidden)
    out = prepare_moe_weights_for_vllm_rollout(
        [("model.language_model.layers.0.mlp.experts.gate_up_proj", gate_up)],
        hidden_size=hidden,
        moe_intermediate_size=inter,
    )
    assert all(weight.dim() == 2 for _, weight in out)
    assert "model.language_model.layers.0.mlp.experts.0.gate_proj.weight" in dict(out)
