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

"""W4A4 fake quant for fused MoE expert modules (Qwen3 / Qwen3.5 style)."""

from __future__ import annotations

import logging
import os
import re
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from verl.utils.qat.calibration import UNINITIALIZED_SCALE, is_scale_uninitialized
from verl.utils.qat.core import QATConfig
from verl.utils.qat.linear import FP4_E2M1_MAX, FP8_E4M3_MAX, STEFP4QuantTriton

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

MOE_W13_BUFFER_PREFIX = "w13_"
MOE_W2_BUFFER_PREFIX = "w2_"


def _is_fused_moe_experts(module: nn.Module) -> bool:
    gate_up = getattr(module, "gate_up_proj", None)
    down = getattr(module, "down_proj", None)
    return (
        isinstance(gate_up, nn.Parameter)
        and isinstance(down, nn.Parameter)
        and gate_up.dim() == 3
        and down.dim() == 3
    )


def _should_ignore_moe(name: str, config: QATConfig) -> bool:
    for pattern in config.ignore_patterns:
        if pattern.startswith("re:"):
            regex = pattern[3:]
            if re.match(regex, name):
                return True
        elif pattern in name:
            return True
    return False


def _is_scale_buffer_initialized(module: nn.Module, prefix: str) -> bool:
    scale = getattr(module, f"{prefix}input_global_scale", None)
    return scale is not None and not is_scale_uninitialized(scale)


def _update_moe_input_scale(module: nn.Module, x: torch.Tensor, prefix: str) -> None:
    """Update layer-level w13/w2 input scale buffers (mirrors QATLinear observer)."""
    observer = getattr(module, "_qat_activation_observer", "static_minmax")
    amax_buf = getattr(module, f"{prefix}input_amax")
    scale_buf = getattr(module, f"{prefix}input_global_scale")

    current_amax = torch.amax(torch.abs(x)).detach().to(torch.float32)
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        torch.distributed.all_reduce(current_amax, op=torch.distributed.ReduceOp.MAX)

    scale_factor = FP8_E4M3_MAX * FP4_E2M1_MAX

    if observer == "memoryless_minmax":
        new_scale = (scale_factor / (current_amax + 1e-12)).view(1)
        scale_buf.copy_(new_scale.to(scale_buf.device))
    elif observer == "static_minmax":
        if is_scale_uninitialized(amax_buf):
            amax_buf.copy_(current_amax.view(1).to(amax_buf.device))
        else:
            new_amax = torch.maximum(amax_buf, current_amax.view(1).to(amax_buf.device))
            amax_buf.copy_(new_amax)
        amax_f32 = amax_buf.to(torch.float32)
        new_scale = (scale_factor / (amax_f32 + 1e-12)).float().view(1)
        scale_buf.copy_(new_scale.to(scale_buf.device))
    elif observer == "minmax":
        ema_decay = 0.01
        if is_scale_uninitialized(amax_buf):
            amax_buf.copy_(current_amax.view(1).to(amax_buf.device))
        else:
            new_amax = (1 - ema_decay) * amax_buf + ema_decay * current_amax.view(1).to(amax_buf.device)
            amax_buf.copy_(new_amax)
        amax_f32 = amax_buf.to(torch.float32)
        new_scale = (scale_factor / (amax_f32 + 1e-12)).float().view(1)
        scale_buf.copy_(new_scale.to(scale_buf.device))
    else:
        raise ValueError(f"Unknown activation_observer: {observer}")


def fake_quantize_activation(
    x: torch.Tensor,
    input_global_scale: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    original_shape = x.shape
    x_2d = x.view(-1, x.shape[-1]) if x.dim() == 3 else x
    if is_scale_uninitialized(input_global_scale):
        raise RuntimeError("MoE input_global_scale uninitialized. Run W4A4 calibration first.")
    global_amax = (FP4_E2M1_MAX * FP8_E4M3_MAX) / input_global_scale.to(x.device)
    result = STEFP4QuantTriton.apply(x_2d, global_amax, group_size)
    return result.view(original_shape)


def fake_quantize_weight_2d(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    global_amax = weight.abs().max().to(torch.float32)
    return STEFP4QuantTriton.apply(weight, global_amax, group_size)


def _attach_qat_moe_state(module: nn.Module, config: QATConfig) -> None:
    module.register_buffer(
        "w13_input_global_scale",
        torch.tensor([UNINITIALIZED_SCALE], dtype=torch.float32),
        persistent=True,
    )
    module.register_buffer(
        "w13_input_amax",
        torch.tensor([UNINITIALIZED_SCALE], dtype=torch.float32),
        persistent=True,
    )
    module.register_buffer(
        "w2_input_global_scale",
        torch.tensor([UNINITIALIZED_SCALE], dtype=torch.float32),
        persistent=True,
    )
    module.register_buffer(
        "w2_input_amax",
        torch.tensor([UNINITIALIZED_SCALE], dtype=torch.float32),
        persistent=True,
    )
    module._qat_activation_observer = config.activation_observer
    module._qat_group_size = config.group_size
    module.fake_quant_enabled = True


def qwen3_fused_experts_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """QAT-wrapped forward for Qwen3/Qwen3.5 fused MoE experts (loop over routed experts)."""
    final_hidden_states = torch.zeros_like(hidden_states)
    fake_quant_enabled = getattr(self, "fake_quant_enabled", True)
    group_size = self._qat_group_size

    with torch.no_grad():
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    if fake_quant_enabled and self.training:
        routed_token_indices = torch.unique(top_k_index.reshape(-1))
        if routed_token_indices.numel() > 0:
            _update_moe_input_scale(self, hidden_states[routed_token_indices], MOE_W13_BUFFER_PREFIX)

    for expert_idx_tensor in expert_hit:
        expert_idx = int(expert_idx_tensor[0])
        if expert_idx == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]

        if fake_quant_enabled:
            if self.training and not _is_scale_buffer_initialized(self, MOE_W13_BUFFER_PREFIX):
                _update_moe_input_scale(self, current_state, MOE_W13_BUFFER_PREFIX)
            current_state = fake_quantize_activation(
                current_state, self.w13_input_global_scale, group_size
            )

        gate_up_w = self.gate_up_proj[expert_idx]
        if fake_quant_enabled:
            gate_up_w = fake_quantize_weight_2d(gate_up_w, group_size)

        gate, up = F.linear(current_state, gate_up_w).chunk(2, dim=-1)
        inter = self.act_fn(gate) * up

        if fake_quant_enabled:
            if self.training:
                _update_moe_input_scale(self, inter, MOE_W2_BUFFER_PREFIX)
            inter = fake_quantize_activation(inter, self.w2_input_global_scale, group_size)

        down_w = self.down_proj[expert_idx]
        if fake_quant_enabled:
            down_w = fake_quantize_weight_2d(down_w, group_size)

        current_hidden_states = F.linear(inter, down_w)
        current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

    return final_hidden_states


MOE_QAT_REGISTRY: dict[str, Callable[[nn.Module, QATConfig], None]] = {}


def _register_default_moe_wrappers() -> None:
    def _wrap_qwen3(module: nn.Module, config: QATConfig) -> None:
        _attach_qat_moe_state(module, config)
        module.forward = qwen3_fused_experts_forward.__get__(module, type(module))

    for cls_name in (
        "Qwen3MoeExperts",
        "Qwen3_5MoeExperts",
        "Qwen3VLMoeTextExperts",
        "Qwen3NextExperts",
    ):
        MOE_QAT_REGISTRY[cls_name] = _wrap_qwen3


_register_default_moe_wrappers()


def _resolve_moe_wrapper(module: nn.Module) -> Optional[Callable[[nn.Module, QATConfig], None]]:
    cls_name = type(module).__name__
    if cls_name in MOE_QAT_REGISTRY:
        return MOE_QAT_REGISTRY[cls_name]
    if _is_fused_moe_experts(module):
        return MOE_QAT_REGISTRY["Qwen3MoeExperts"]
    return None


def _apply_qat_moe_layers(model: nn.Module, config: QATConfig) -> int:
    """Attach W4A4 fake quant to fused MoE expert modules. Called from apply_qat()."""
    wrapped = 0
    for name, module in model.named_modules():
        if not _is_fused_moe_experts(module):
            continue
        if _should_ignore_moe(name, config):
            logger.debug(f"Skipping MoE QAT for {name} due to ignore_patterns")
            continue
        if hasattr(module, "w13_input_global_scale"):
            continue

        wrapper_fn = _resolve_moe_wrapper(module)
        if wrapper_fn is None:
            logger.warning(f"No MoE QAT wrapper for {name} ({type(module).__name__}), skipping")
            continue

        wrapper_fn(module, config)
        wrapped += 1
        logger.debug(f"Applied MoE QAT to {name}")

    return wrapped


def iter_qat_moe_modules(model: nn.Module):
    """Yield modules with MoE QAT buffers attached."""
    for module in model.modules():
        if hasattr(module, "w13_input_global_scale") and hasattr(module, "w2_input_global_scale"):
            yield module


def count_uninitialized_moe_layers(model: nn.Module) -> tuple[int, int]:
    """Return (total_moe_layers, uninitialized_layers). Each layer has w13 + w2 scales."""
    total = 0
    uninitialized = 0
    for module in iter_qat_moe_modules(model):
        total += 1
        w13_bad = is_scale_uninitialized(module.w13_input_global_scale) or is_scale_uninitialized(
            module.w13_input_amax
        )
        w2_bad = is_scale_uninitialized(module.w2_input_global_scale) or is_scale_uninitialized(module.w2_input_amax)
        if w13_bad or w2_bad:
            uninitialized += 1
    return total, uninitialized


def needs_moe_calibration(model: nn.Module) -> bool:
    total, uninitialized = count_uninitialized_moe_layers(model)
    return total > 0 and uninitialized > 0


def fallback_moe_scales_from_weights(model: nn.Module) -> int:
    """Cold-start w13/w2 input scales from expert weight amax (layer-level, 2 scales per layer)."""
    from verl.utils.qat.calibration import input_global_scale_from_amax

    initialized = 0
    for module in iter_qat_moe_modules(model):
        if is_scale_uninitialized(module.w13_input_global_scale):
            gate_up = module.gate_up_proj
            if gate_up.numel() > 0:
                amax = torch.amax(torch.abs(gate_up)).to(torch.float32)
                module.w13_input_amax.copy_(amax.view(1))
                module.w13_input_global_scale.copy_(input_global_scale_from_amax(amax))
                initialized += 1
        if is_scale_uninitialized(module.w2_input_global_scale):
            down = module.down_proj
            if down.numel() > 0:
                amax = torch.amax(torch.abs(down)).to(torch.float32)
                module.w2_input_amax.copy_(amax.view(1))
                module.w2_input_global_scale.copy_(input_global_scale_from_amax(amax))
                initialized += 1
    return initialized


__all__ = [
    "MOE_QAT_REGISTRY",
    "count_uninitialized_moe_layers",
    "fake_quantize_activation",
    "fake_quantize_weight_2d",
    "fallback_moe_scales_from_weights",
    "iter_qat_moe_modules",
    "needs_moe_calibration",
    "qwen3_fused_experts_forward",
]
