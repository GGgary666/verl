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

"""
Fast NVFP4 Quantizer for verl FSDP training.

Directly computes scales and quantizes weights using compressed_tensors APIs.
Includes scale computation utilities for weight quantization.
"""

import logging
import os
import re
from typing import Generator, Iterable, Optional

import torch
from compressed_tensors.quantization.quant_args import (
    FP4_E2M1_DATA,
    FP8_E4M3_DATA,
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.utils.helpers import generate_gparam

from verl.utils.device import get_device_name, get_torch_device
from verl.utils.qat.calibration import input_global_scale_from_amax, is_scale_uninitialized
from verl.utils.qat.compressed_tensors_compat import create_nvfp4_weight_packer

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_LAYER_IDX_RE = re.compile(r"layers\.(\d+)\.")
_MOE_EXPERT_PROJ_RE = re.compile(r"\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)$")


def _collect_moe_expert_input_scales(layer_params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map MoE expert projection module names to layer-level w13/w2 input scales."""
    w13_by_experts: dict[str, torch.Tensor] = {}
    w2_by_experts: dict[str, torch.Tensor] = {}

    for name, tensor in layer_params.items():
        if name.endswith(".experts.w13_input_global_scale") and not is_scale_uninitialized(tensor):
            w13_by_experts[name.replace(".w13_input_global_scale", "")] = tensor
        elif name.endswith(".experts.w2_input_global_scale") and not is_scale_uninitialized(tensor):
            w2_by_experts[name.replace(".w2_input_global_scale", "")] = tensor
        elif name.endswith(".experts.w13_input_amax") and not is_scale_uninitialized(tensor):
            experts_base = name.replace(".w13_input_amax", "")
            w13_by_experts.setdefault(experts_base, input_global_scale_from_amax(tensor))
        elif name.endswith(".experts.w2_input_amax") and not is_scale_uninitialized(tensor):
            experts_base = name.replace(".w2_input_amax", "")
            w2_by_experts.setdefault(experts_base, input_global_scale_from_amax(tensor))

    expert_scales: dict[str, torch.Tensor] = {}
    for layer_name in layer_params:
        if not layer_name.endswith(".weight"):
            continue
        module_name = layer_name.rsplit(".weight", 1)[0]
        match = _MOE_EXPERT_PROJ_RE.search(module_name)
        if not match:
            continue
        experts_base = module_name[: match.start() + len(".experts")]
        proj = match.group(2)
        if proj in ("gate_proj", "up_proj"):
            scale = w13_by_experts.get(experts_base)
        else:
            scale = w2_by_experts.get(experts_base)
        if scale is not None:
            expert_scales[module_name] = scale
    return expert_scales


def _is_moe_expert_projection(layer_name: str) -> bool:
    return _MOE_EXPERT_PROJ_RE.search(layer_name) is not None

def _collect_fused_input_global_scales(
    layer_params: dict[str, torch.Tensor],
    input_global_scales: dict[str, torch.Tensor],
    layer_names: set[str],
) -> dict[str, torch.Tensor]:
    """Resolve per-layer input_global_scale from buffers, with QKV/GateUp fusion."""
    collected: dict[str, torch.Tensor] = {}

    for layer_name in layer_names:
        scale_key = f"{layer_name}.input_global_scale"
        if scale_key in layer_params and not is_scale_uninitialized(layer_params[scale_key]):
            collected[layer_name] = layer_params[scale_key]
            continue

        amax_key = f"{layer_name}.input_amax"
        if amax_key in layer_params and not is_scale_uninitialized(layer_params[amax_key]):
            collected[layer_name] = input_global_scale_from_amax(layer_params[amax_key])
            continue

        streamed = input_global_scales.get(layer_name)
        if streamed is not None and not is_scale_uninitialized(streamed):
            collected[layer_name] = streamed

    if not collected:
        return {}
    return fuse_global_scales(collected, strategy="min")


def compute_blockwise_scale(
    weight: torch.Tensor,
    global_scale: torch.Tensor,
    group_size: int = 16,
) -> torch.Tensor:
    """Compute blockwise scale using pre-computed global_scale (for fusion).
    Returns FP8 E4M3 blockwise scale tensor.
    """
    out_features, in_features = weight.shape
    num_groups = in_features // group_size
    weight_reshaped = weight.view(out_features, num_groups, group_size)
    block_max = torch.amax(torch.abs(weight_reshaped), dim=-1).to(torch.float32)

    local_scale = block_max / FP4_E2M1_DATA.max
    blockwise_scale_f32 = torch.clamp(
        global_scale * local_scale,
        min=-FP8_E4M3_DATA.max,
        max=FP8_E4M3_DATA.max,
    )

    blockwise_scale = blockwise_scale_f32.to(torch.float8_e4m3fn)
    eps = torch.finfo(torch.float8_e4m3fn).eps
    blockwise_scale = torch.where(
        blockwise_scale == 0,
        torch.tensor(eps, dtype=blockwise_scale.dtype, device=weight.device),
        blockwise_scale,
    )

    return blockwise_scale


# Fusion patterns for transformer models
FUSE_PATTERNS = {
    "qkv": ["q_proj", "k_proj", "v_proj"],
    "gate_up": ["gate_proj", "up_proj"],
}


def fuse_global_scales(
    layer_global_scales: dict[str, torch.Tensor],
    strategy: str = "min",
) -> dict[str, torch.Tensor]:
    """Fuse global scales for QKV/GateUp groups (take min across group)."""
    if not layer_global_scales:
        return {}

    # Group by parent module
    parent_to_children: dict[str, dict[str, str]] = {}
    for name in layer_global_scales:
        parent, child = name.rsplit(".", 1) if "." in name else ("", name)
        parent_to_children.setdefault(parent, {})[child] = name

    fused_scales = {}
    processed = set()

    for parent, children in parent_to_children.items():
        for _, patterns in FUSE_PATTERNS.items():
            matched = [children[p] for p in patterns if p in children]
            if len(matched) == len(patterns):
                group_scales = [layer_global_scales[n] for n in matched]
                if strategy == "min":
                    fused_scale = torch.min(torch.cat(group_scales)).reshape([1])
                else:
                    raise ValueError(f"Unknown fuse strategy: {strategy}")
                for layer_name in matched:
                    fused_scales[layer_name] = fused_scale.clone()
                    processed.add(layer_name)

    for name, scale in layer_global_scales.items():
        if name not in processed:
            fused_scales[name] = scale

    return fused_scales


class QATQuantizer:
    """Quantizer for QAT-trained weights using compressed_tensors APIs."""

    def __init__(
        self,
        mode: str = "w4a16",
        group_size: int = 16,
        ignore_patterns: Optional[list] = None,
        device: Optional[torch.device] = None,
        param_dtype: Optional[torch.dtype] = None,
    ):
        self.mode = mode.lower()
        self._is_w4a4 = self.mode == "w4a4"  # W4A4 needs input_global_scale
        self.group_size = group_size
        self.ignore_patterns = ignore_patterns or ["lm_head", "embed_tokens", "re:.*mlp.gate$"]
        self.device = device or torch.device(get_device_name())
        self.param_dtype = param_dtype

        self._compressor = create_nvfp4_weight_packer()
        self._quant_args = QuantizationArgs(
            num_bits=4,
            type=QuantizationType.FLOAT,
            symmetric=True,
            strategy=QuantizationStrategy.TENSOR_GROUP,
            group_size=group_size,
            scale_dtype=FP8_E4M3_DATA.dtype,
        )

    def _module_name(self, name: str) -> str:
        if name.endswith(".weight"):
            return name.rsplit(".weight", 1)[0]
        return name

    def _matches_ignore_pattern(self, module_name: str) -> bool:
        for pattern in self.ignore_patterns:
            if pattern.startswith("re:"):
                if re.match(pattern[3:], module_name):
                    return True
            elif pattern in module_name:
                return True
        return False

    def _should_quantize(self, name: str, tensor: torch.Tensor) -> bool:
        """Check if parameter should be quantized."""
        if not name.endswith(".weight"):
            return False
        if tensor.dim() != 2:
            return False
        if tensor.shape[1] % self.group_size != 0:
            return False

        return not self._matches_ignore_pattern(self._module_name(name))

    def _should_sync_to_vllm(self, name: str) -> bool:
        """Whether a non-quantized weight should still be synced to vLLM.

        Vision tower weights are skipped: vLLM uses a different internal layout than
        HF state_dict (e.g. pos_embed / attn.qkv shapes). Keep the initial vLLM
        checkpoint weights for text-only RL workloads.
        """
        module_name = self._module_name(name)
        if not self._matches_ignore_pattern(module_name):
            return True
        return "visual" not in module_name

    @staticmethod
    def _extract_layer_idx(name: str) -> Optional[int]:
        """Extract decoder layer index from parameter name."""
        match = _LAYER_IDX_RE.search(name)
        return int(match.group(1)) if match else None

    def _process_layer_group(
        self,
        layer_idx: Optional[int],
        layer_params: dict[str, torch.Tensor],
        input_global_scales: dict[str, torch.Tensor],
        output_device: torch.device,
    ) -> list[tuple[str, torch.Tensor]]:
        """Quantize one decoder layer's buffered params. Returns list of (name, tensor)."""
        layer_weights = {}
        layer_passthrough = {}

        for name, tensor in layer_params.items():
            if "input_global_scale" in name or "input_amax" in name:
                continue

            if self._should_quantize(name, tensor):
                layer_name = name.rsplit(".weight", 1)[0]
                layer_weights[layer_name] = (name, tensor)
            elif self._should_sync_to_vllm(name):
                layer_passthrough[name] = tensor

        if layer_idx is None and layer_weights:
            raise RuntimeError(
                f"[QAT Quantizer] Unexpected quantizable weights outside decoder layers: "
                f"{list(layer_weights.keys())}. These should be in ignore_patterns."
            )

        if not layer_weights:
            return [(name, tensor.to(output_device)) for name, tensor in layer_passthrough.items()]

        # Move weights to GPU, compute global scales
        weights_on_gpu = {}
        layer_global_scales = {}

        for layer_name, (_, tensor) in layer_weights.items():
            weight_gpu = tensor.to(device=self.device, dtype=self.param_dtype)
            weights_on_gpu[layer_name] = weight_gpu
            amax = torch.amax(torch.abs(weight_gpu)).to(torch.float32)
            layer_global_scales[layer_name] = generate_gparam(
                -amax.unsqueeze(0),
                amax.unsqueeze(0),
                scale_data=FP8_E4M3_DATA,
                quant_data=FP4_E2M1_DATA,
                dtype=torch.float32,
            )

        fused_global_scales = fuse_global_scales(layer_global_scales, strategy="min")
        fused_input_global_scales = (
            _collect_fused_input_global_scales(layer_params, input_global_scales, set(weights_on_gpu.keys()))
            if self._is_w4a4
            else {}
        )
        if self._is_w4a4:
            moe_input_scales = _collect_moe_expert_input_scales(layer_params)
            for layer_name, scale in moe_input_scales.items():
                fused_input_global_scales.setdefault(layer_name, scale)

        moe_buffer_results: list[tuple[str, torch.Tensor]] = []
        if self._is_w4a4:
            for name, tensor in layer_params.items():
                if name.endswith(".experts.w13_input_global_scale") or name.endswith(".experts.w2_input_global_scale"):
                    if not is_scale_uninitialized(tensor):
                        moe_buffer_results.append((name, tensor.float().to(output_device)))

        results = []

        for layer_name, weight_gpu in weights_on_gpu.items():
            fused_global_scale = fused_global_scales[layer_name]
            weight_scale = compute_blockwise_scale(weight_gpu, fused_global_scale, self.group_size)
            weight_packed = self._compressor.compress_weight(
                weight=weight_gpu,
                scale=weight_scale.float(),
                global_scale=fused_global_scale,
                quantization_args=self._quant_args,
            )["weight_packed"]

            results.append((f"{layer_name}.weight_packed", weight_packed.to(output_device)))
            results.append((f"{layer_name}.weight_scale", weight_scale.to(output_device)))
            results.append((f"{layer_name}.weight_global_scale", fused_global_scale.to(output_device)))

            if self._is_w4a4:
                input_scale = fused_input_global_scales.get(layer_name)
                if input_scale is None:
                    if _is_moe_expert_projection(layer_name):
                        raise RuntimeError(
                            f"W4A4 MoE: {layer_name} missing layer-level w13/w2 input_global_scale buffer"
                        )
                    logger.warning(
                        f"W4A4: {layer_name} input_global_scale uninitialized, "
                        "bootstrapping from weight amax until forward pass updates it"
                    )
                    input_scale = input_global_scale_from_amax(torch.amax(torch.abs(weight_gpu)))
                results.append((f"{layer_name}.input_global_scale", input_scale.float().to(output_device)))

        results.extend(moe_buffer_results)

        del weights_on_gpu, layer_global_scales, fused_global_scales

        for name, tensor in layer_passthrough.items():
            results.append((name, tensor.to(output_device)))

        return results

    def quantize_with_fusion(
        self,
        params: dict[str, torch.Tensor] | Iterable[tuple[str, torch.Tensor]],
        target_device: Optional[torch.device] = None,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Streaming quantize: consume input layer by layer, yield (name, tensor) pairs."""
        if isinstance(params, dict):
            params = params.items()

        output_device = target_device or torch.device("cpu")

        _sentinel = object()
        current_layer_idx = _sentinel
        layer_buffer: dict[str, torch.Tensor] = {}
        input_global_scales: dict[str, torch.Tensor] = {}
        for name, tensor in params:
            tensor_cpu = tensor.to("cpu") if tensor.is_cuda else tensor
            layer_idx = self._extract_layer_idx(name)

            # Collect input scales for W4A4 as we go (from scale buffer or amax observer)
            if self._is_w4a4 and "input_global_scale" in name:
                scale_layer_name = name.replace(".input_global_scale", "")
                if is_scale_uninitialized(tensor_cpu):
                    logger.warning(f"W4A4: {scale_layer_name} input_global_scale is uninitialized")
                else:
                    input_global_scales[scale_layer_name] = tensor_cpu
            elif self._is_w4a4 and "input_amax" in name:
                scale_layer_name = name.replace(".input_amax", "")
                if not is_scale_uninitialized(tensor_cpu):
                    input_global_scales[scale_layer_name] = input_global_scale_from_amax(tensor_cpu)

            # Layer boundary: flush previous layer
            if layer_idx != current_layer_idx and current_layer_idx is not _sentinel and layer_buffer:
                yield from self._process_layer_group(
                    current_layer_idx, layer_buffer, input_global_scales, output_device
                )
                layer_buffer = {}

            current_layer_idx = layer_idx
            layer_buffer[name] = tensor_cpu

        # Flush last buffered layer
        if layer_buffer:
            yield from self._process_layer_group(current_layer_idx, layer_buffer, input_global_scales, output_device)

        get_torch_device().empty_cache()


__all__ = [
    "QATQuantizer",
]
