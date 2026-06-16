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
vLLM NVFP4 Patches for Dynamic Weight Updates.

Enables dynamic weight reloading for NVFP4 quantized models in vLLM.

Supported schemes:
- Dense: W4A16-FP4, W4A4-FP4
- MoE: NVFP4-MoE

vLLM compatibility is resolved at runtime via API inspection (no hard-coded
version checks):
- <=0.18: nvfp4_marlin_process_scales returns a tensor; global scale uses
  weight_scale_2 and half/bfloat16 inputs.
- >=0.19: nvfp4_marlin_process_scales returns (tensor, scale_factor); global
  scale uses weight_global_scale and float32 inputs with a_dtype.
- >=0.20: MoE classes move under compressed_tensors_moe.* submodules.
- W4A4 >=0.19: swizzle_blockscale moves to nvfp4_utils; linear uses kernel API
  (weight_packed -> weight + kernel.process_weights_after_loading).
"""

import importlib
import inspect
import logging
import os
from typing import Optional
from unittest.mock import patch

import torch
from torch.nn import Parameter

from verl.utils.device import get_device_name

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ParamMetaDict(dict):
    """
    Dict-like class for parameter management with metadata-based rebuild and tensor swap.

    Supports:
    - Rebuild of deleted parameters from saved metadata
    - Tensor Swap for parameters with shape changes (address stability for CUDA Graph)
    """

    def __init__(self, model: torch.nn.Module, device: Optional[torch.device] = None):
        """
        Initialize ParamMetaDict from a model.

        Args:
            model: vLLM model (may be wrapped in ModelRunner)
            device: Device for created parameters
        """
        super().__init__()
        self.device = device

        # Get the actual model (handle vLLM's wrapper structure)
        actual_model = model
        if hasattr(model, "model"):
            actual_model = model.model
        self._model = actual_model

        # Build mappings by scanning all modules
        self._layer_meta_cache: dict[str, dict] = {}  # Cache of _hf_param_meta
        self._tensor_swap_layers: dict[str, dict] = {}  # Layers needing tensor swap

        self._build_mappings()

        # Initialize with current parameters
        for name, param in actual_model.named_parameters():
            self[name] = param

    def _build_mappings(self):
        """Build layer metadata cache for rebuild and tensor swap."""
        for layer_name, module in self._model.named_modules():
            # Check for _hf_param_meta which indicates this layer has HF format params
            if hasattr(module, "_hf_param_meta"):
                self._layer_meta_cache[layer_name] = {
                    "module": module,
                    "meta": module._hf_param_meta,
                }

                # Check for tensor swap layers (weight_scale with shape change)
                if "weight_scale" in module._hf_param_meta:
                    marlin_refs = getattr(module, "_marlin_tensor_refs", {})
                    if "weight_scale" in marlin_refs:
                        self._tensor_swap_layers[layer_name] = {
                            "module": module,
                            "marlin_ref": marlin_refs["weight_scale"],
                            "hf_meta": module._hf_param_meta["weight_scale"],
                        }

                # MoE layers (w13_weight_scale, w2_weight_scale)
                if "w13_weight_scale" in module._hf_param_meta:
                    marlin_refs = getattr(module, "_marlin_tensor_refs", {})
                    if "w13_weight_scale" in marlin_refs:
                        self._tensor_swap_layers[f"{layer_name}.w13"] = {
                            "module": module,
                            "param_name": "w13_weight_scale",
                            "marlin_ref": marlin_refs["w13_weight_scale"],
                            "hf_meta": module._hf_param_meta["w13_weight_scale"],
                        }
                    if "w2_weight_scale" in marlin_refs:
                        self._tensor_swap_layers[f"{layer_name}.w2"] = {
                            "module": module,
                            "param_name": "w2_weight_scale",
                            "marlin_ref": marlin_refs["w2_weight_scale"],
                            "hf_meta": module._hf_param_meta["w2_weight_scale"],
                        }

    def _try_rebuild(self, key: str) -> Optional[Parameter]:
        """
        Try to rebuild a parameter from metadata if it was deleted.

        Args:
            key: Full parameter name

        Returns:
            Rebuilt parameter or None if cannot rebuild
        """
        # Extract layer name and param name
        parts = key.rsplit(".", 1)
        if len(parts) != 2:
            return None

        layer_name, param_name = parts

        # Check if we have metadata for this layer
        if layer_name not in self._layer_meta_cache:
            return None

        cache_entry = self._layer_meta_cache[layer_name]
        module = cache_entry["module"]
        meta = cache_entry["meta"]

        # Check if this param needs rebuild
        if param_name not in meta:
            return None

        # Already exists on module?
        if hasattr(module, param_name):
            param = getattr(module, param_name)
            if param is not None:
                return param

        # Rebuild from metadata
        new_param = _create_param_from_meta(module, param_name, meta[param_name], self.device)
        module.register_parameter(param_name, new_param)
        return new_param

    def prepare_for_reload(self) -> None:
        """Replace Marlin-format tensors with HF-shape tensors for reload."""
        for layer_name, swap_info in self._tensor_swap_layers.items():
            module = swap_info["module"]
            param_name = swap_info.get("param_name", "weight_scale")
            hf_meta = swap_info["hf_meta"]
            if hasattr(module, param_name):
                new_param = _create_param_from_meta(module, param_name, hf_meta, self.device)
                setattr(module, param_name, new_param)

    def __getitem__(self, key: str) -> Parameter:
        """Get parameter with rebuild support."""
        # Try standard lookup first
        if key in dict.keys(self):
            return super().__getitem__(key)

        # Try rebuild from metadata
        param = self._try_rebuild(key)
        if param is not None:
            self[key] = param
            return param

        raise KeyError(f"Parameter not found: {key}")

    def __contains__(self, key: str) -> bool:
        """Check if parameter exists (with rebuild check)."""
        if super().__contains__(key):
            return True

        # Check if can rebuild from metadata
        parts = key.rsplit(".", 1)
        if len(parts) == 2:
            layer_name, param_name = parts
            if layer_name in self._layer_meta_cache:
                meta = self._layer_meta_cache[layer_name]["meta"]
                if param_name in meta:
                    return True

        return False

    def get(self, key: str, default=None):
        """Get parameter with default."""
        try:
            return self[key]
        except KeyError:
            return default


def _create_param_from_meta(
    module: torch.nn.Module,
    param_name: str,
    meta: dict,
    device: Optional[torch.device] = None,
) -> Parameter:
    """Create a Parameter from saved metadata. Used by rebuild and tensor swap."""
    shape = meta["shape"]
    dtype = meta["dtype"]
    dev = device or meta.get("device", get_device_name())
    param_class = meta.get("param_class", Parameter)

    weight_loaders = getattr(module, "_weight_loaders", {})
    weight_loader = weight_loaders.get(param_name)

    data = torch.empty(shape, dtype=dtype, device=dev)

    try:
        if param_class is not Parameter and weight_loader is not None:
            kwargs = {"data": data, "weight_loader": weight_loader}
            if "input_dim" in meta:
                kwargs["input_dim"] = meta["input_dim"]
            if "output_dim" in meta:
                kwargs["output_dim"] = meta["output_dim"]
            new_param = param_class(**kwargs)
        else:
            new_param = Parameter(data, requires_grad=False)
            if weight_loader is not None:
                new_param.weight_loader = weight_loader
    except Exception as e:
        logger.warning(f"Failed to create param {param_name} with class {param_class}: {e}, using Parameter")
        new_param = Parameter(data, requires_grad=False)
        if weight_loader is not None:
            new_param.weight_loader = weight_loader

    if "quant_method" in meta:
        new_param.quant_method = meta["quant_method"]

    return new_param


def save_param_meta(layer: torch.nn.Module, param_name: str):
    """Save parameter metadata for rebuild."""
    if not hasattr(layer, "_hf_param_meta"):
        layer._hf_param_meta = {}

    param = getattr(layer, param_name, None)
    if param is None:
        return

    meta = {
        "shape": tuple(param.shape),
        "dtype": param.dtype,
        "device": str(param.device),
        "param_class": type(param),  # Save the actual parameter class
    }

    # Save vLLM-specific attributes needed for reconstruction
    if hasattr(param, "_input_dim"):
        meta["input_dim"] = param._input_dim
    if hasattr(param, "_output_dim"):
        meta["output_dim"] = param._output_dim

    # Save MoE-specific attributes (quant_method is required by weight_loader)
    if hasattr(param, "quant_method"):
        meta["quant_method"] = param.quant_method

    layer._hf_param_meta[param_name] = meta


def _check_first_call(layer: torch.nn.Module) -> bool:
    """Check if this is the first process_weights call, and increment counter."""
    count = getattr(layer, "_process_weights_call_count", 0)
    layer._process_weights_call_count = count + 1
    return count == 0


def _split_marlin_scale(value):
    """Unpack nvfp4_marlin_process_scales output (tensor or (tensor, scale_factor))."""
    return value if isinstance(value, tuple) else (value, 1.0)


def _nvfp4_marlin_supports_a_dtype(func) -> bool:
    import inspect

    return "a_dtype" in inspect.signature(func).parameters


def _w4a16_global_scale_param_name() -> str:
    import inspect

    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import apply_fp4_marlin_linear

    if "weight_global_scale" in inspect.signature(apply_fp4_marlin_linear).parameters:
        return "weight_global_scale"
    return "weight_scale_2"


def _call_nvfp4_marlin_process_scales(scales, param_dtype):
    """Call nvfp4_marlin_process_scales across vLLM <=0.18 and >=0.19 APIs."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import nvfp4_marlin_process_scales

    if _nvfp4_marlin_supports_a_dtype(nvfp4_marlin_process_scales):
        raw = nvfp4_marlin_process_scales(scales, a_dtype=param_dtype)
    else:
        raw = nvfp4_marlin_process_scales(scales)
    return _split_marlin_scale(raw)


def _call_nvfp4_marlin_process_global_scale(inverted_global_scale, param_dtype, scale_factor=1.0):
    """Process inverted global scale (1/max) into Marlin format across vLLM versions."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import nvfp4_marlin_process_global_scale

    if _nvfp4_marlin_supports_a_dtype(nvfp4_marlin_process_global_scale):
        value = (
            inverted_global_scale
            if inverted_global_scale.dtype == torch.float32
            else inverted_global_scale.to(torch.float32)
        )
        processed = nvfp4_marlin_process_global_scale(value, a_dtype=param_dtype)
    else:
        processed = nvfp4_marlin_process_global_scale(inverted_global_scale.to(param_dtype))
    return processed / scale_factor


# Dense W4A16 Patches
def patched_w4a16_process_weights_after_loading(self, layer: torch.nn.Module) -> None:
    """Patched process_weights_after_loading for W4A16 Dense layer."""
    import vllm._custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        marlin_make_workspace_new,
        marlin_permute_scales,
    )

    is_first_call = _check_first_call(layer)

    group_size = 16
    part_size_n = layer.output_size_per_partition
    part_size_k = layer.input_size_per_partition
    device = layer.weight_packed.device
    param_dtype = getattr(layer, "params_dtype", torch.float16)

    # Save metadata (first call only)
    if is_first_call:
        save_param_meta(layer, "weight_packed")
        save_param_meta(layer, "weight_global_scale")
        save_param_meta(layer, "weight_scale")
        if not hasattr(layer, "_weight_loaders"):
            layer._weight_loaders = {}
        for pname in ["weight_packed", "weight_global_scale", "weight_scale"]:
            param = getattr(layer, pname, None)
            if param is not None and hasattr(param, "weight_loader"):
                layer._weight_loaders[pname] = param.weight_loader

    # Get HF format data
    weight_packed_hf = layer.weight_packed.data
    weight_global_scale_hf = layer.weight_global_scale.data
    weight_scale_hf = layer.weight_scale.data

    # Create workspace (first call only)
    if is_first_call:
        layer.workspace = marlin_make_workspace_new(device)

    # Convert to Marlin format
    perm = torch.empty(0, dtype=torch.int, device=device)
    qweight = weight_packed_hf.view(torch.int32).T.contiguous()
    marlin_weight = ops.gptq_marlin_repack(
        b_q_weight=qweight,
        perm=perm,
        size_k=part_size_k,
        size_n=part_size_n,
        num_bits=4,
        is_a_8bit=False,
    )

    weight_scale = weight_scale_hf.T.contiguous().to(param_dtype)
    weight_scale_permuted = marlin_permute_scales(
        s=weight_scale,
        size_k=part_size_k,
        size_n=part_size_n,
        group_size=group_size,
        is_a_8bit=False,
    )
    marlin_weight_scale, scale_factor = _call_nvfp4_marlin_process_scales(weight_scale_permuted, param_dtype)
    marlin_weight_global_scale = _call_nvfp4_marlin_process_global_scale(
        1.0 / weight_global_scale_hf.max(),
        param_dtype,
        scale_factor=scale_factor,
    )

    global_scale_attr = _w4a16_global_scale_param_name()

    # Update compute parameters
    if is_first_call:
        layer.weight = Parameter(marlin_weight, requires_grad=False)
        layer.weight_scale = Parameter(marlin_weight_scale, requires_grad=False)
        setattr(layer, global_scale_attr, Parameter(marlin_weight_global_scale, requires_grad=False))
        if not hasattr(layer, "_marlin_tensor_refs"):
            layer._marlin_tensor_refs = {}
        layer._marlin_tensor_refs["weight_scale"] = layer.weight_scale.data
        layer._marlin_tensor_refs[global_scale_attr] = getattr(layer, global_scale_attr).data
    else:
        layer.weight.data.copy_(marlin_weight)
        getattr(layer, global_scale_attr).data.copy_(marlin_weight_global_scale)
        marlin_scale_ref = layer._marlin_tensor_refs.get("weight_scale")
        if marlin_scale_ref is not None:
            marlin_scale_ref.copy_(marlin_weight_scale)
            layer.weight_scale = Parameter(marlin_scale_ref, requires_grad=False)
        else:
            logger.warning("W4A16: _marlin_tensor_refs['weight_scale'] not found")
            layer.weight_scale = Parameter(marlin_weight_scale, requires_grad=False)

    # Delete HF parameters
    if hasattr(layer, "weight_packed"):
        delattr(layer, "weight_packed")
    if global_scale_attr == "weight_scale_2" and hasattr(layer, "weight_global_scale"):
        delattr(layer, "weight_global_scale")


def _resolve_swizzle_blockscale():
    """Resolve swizzle_blockscale across vLLM <=0.18 and >=0.19 layouts."""
    for module_path in (
        "vllm.model_executor.layers.quantization.utils.nvfp4_utils",
        "vllm.model_executor.layers.quantization.utils.quant_utils",
    ):
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError:
            continue
        if hasattr(module, "swizzle_blockscale"):
            return module.swizzle_blockscale
    raise ImportError(
        "Cannot import swizzle_blockscale from vLLM. "
        "Upgrade vLLM or use a version that provides nvfp4_utils/quant_utils."
    )


def _w4a4_uses_kernel_api(scheme_self) -> bool:
    """Detect NVFP4 linear kernel API introduced in vLLM 0.19+."""
    if hasattr(scheme_self, "kernel"):
        return True
    backend = getattr(scheme_self, "backend", None)
    if backend is not None and not isinstance(backend, str):
        try:
            importlib.import_module("vllm.model_executor.layers.quantization.utils.nvfp4_utils")
            return True
        except ModuleNotFoundError:
            return False
    return False


def _patched_w4a4_legacy_process_weights(self, layer: torch.nn.Module) -> None:
    """W4A4 patch for legacy vLLM layouts that keep weight_packed (<=0.18)."""
    swizzle_blockscale = _resolve_swizzle_blockscale()

    is_first_call = _check_first_call(layer)

    _W4A4_HF_PARAMS = ["weight_packed", "weight_scale", "weight_global_scale", "input_global_scale"]

    if is_first_call:
        for pname in _W4A4_HF_PARAMS:
            save_param_meta(layer, pname)
        if not hasattr(layer, "_weight_loaders"):
            layer._weight_loaders = {}
        for pname in _W4A4_HF_PARAMS:
            param = getattr(layer, pname, None)
            if param is not None and hasattr(param, "weight_loader"):
                layer._weight_loaders[pname] = param.weight_loader

    weight_packed_data = layer.weight_packed.data
    weight_scale_data = layer.weight_scale.data
    input_global_scale_data = layer.input_global_scale.data
    weight_global_scale_data = layer.weight_global_scale.data

    global_input_scale = input_global_scale_data.max().to(torch.float32)
    global_weight_scale = weight_global_scale_data.max().to(torch.float32)

    if self.backend == "flashinfer-trtllm":
        from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a

        epilogue_tile_m = 128
        processed_weight = shuffle_matrix_a(weight_packed_data.view(torch.uint8), epilogue_tile_m)
        processed_weight_scale = (
            shuffle_matrix_sf_a(weight_scale_data.view(torch.uint8), epilogue_tile_m)
            .reshape(weight_scale_data.shape)
            .view(torch.float8_e4m3fn)
        )
    elif self.backend == "fbgemm":
        processed_weight_scale = swizzle_blockscale(weight_scale_data).view(-1).view(torch.uint8)
        processed_weight = weight_packed_data
    else:
        # cutlass / flashinfer-cutlass
        processed_weight_scale = swizzle_blockscale(weight_scale_data)
        processed_weight = weight_packed_data

    alpha = 1.0 / (global_input_scale * global_weight_scale)

    if is_first_call:
        layer.weight_packed = Parameter(processed_weight, requires_grad=False)
        layer.weight_scale = Parameter(processed_weight_scale, requires_grad=False)
        layer.input_global_scale = Parameter(global_input_scale, requires_grad=False)
        layer.weight_global_scale = Parameter(global_weight_scale, requires_grad=False)
        layer.alpha = Parameter(alpha, requires_grad=False)

        if not hasattr(layer, "_marlin_tensor_refs"):
            layer._marlin_tensor_refs = {}
        layer._marlin_tensor_refs["weight_packed"] = layer.weight_packed.data
        layer._marlin_tensor_refs["weight_scale"] = layer.weight_scale.data
        layer._marlin_tensor_refs["input_global_scale"] = layer.input_global_scale.data
        layer._marlin_tensor_refs["weight_global_scale"] = layer.weight_global_scale.data
        layer._marlin_tensor_refs["alpha"] = layer.alpha.data
    else:
        refs = layer._marlin_tensor_refs
        for ref_name, new_data in [
            ("weight_packed", processed_weight),
            ("weight_scale", processed_weight_scale),
            ("input_global_scale", global_input_scale),
            ("weight_global_scale", global_weight_scale),
            ("alpha", alpha),
        ]:
            ref = refs.get(ref_name)
            if ref is not None:
                ref.copy_(new_data)
                setattr(layer, ref_name, Parameter(ref, requires_grad=False))
            else:
                logger.warning(f"W4A4: _marlin_tensor_refs['{ref_name}'] not found, creating new Parameter")
                setattr(
                    layer,
                    ref_name,
                    Parameter(
                        new_data.clone() if isinstance(new_data, torch.Tensor) else torch.tensor(new_data),
                        requires_grad=False,
                    ),
                )


def _update_w4a4_tensor_ref(layer: torch.nn.Module, ref_name: str, new_param: Parameter) -> None:
    """Update a CUDA-graph ref, replacing it when kernel repacking changes shape."""
    refs = layer._marlin_tensor_refs
    ref = refs.get(ref_name)
    if ref is not None and ref.shape == new_param.shape:
        ref.copy_(new_param.data)
        setattr(layer, ref_name, Parameter(ref, requires_grad=False))
    else:
        if ref is not None:
            logger.warning(
                f"W4A4: _marlin_tensor_refs['{ref_name}'] shape changed "
                f"{tuple(ref.shape)} -> {tuple(new_param.shape)}, replacing ref"
            )
        refs[ref_name] = new_param.data
        setattr(layer, ref_name, new_param)


def _patched_w4a4_kernel_process_weights(self, layer: torch.nn.Module) -> None:
    """W4A4 patch for vLLM 0.19+ kernel API (weight_packed -> weight + kernel format)."""
    is_first_call = _check_first_call(layer)

    _W4A4_HF_PARAMS = ["weight_packed", "weight_scale", "weight_global_scale", "input_global_scale"]

    if is_first_call:
        for pname in _W4A4_HF_PARAMS:
            save_param_meta(layer, pname)
        if not hasattr(layer, "_weight_loaders"):
            layer._weight_loaders = {}
        for pname in _W4A4_HF_PARAMS:
            param = getattr(layer, pname, None)
            if param is not None and hasattr(param, "weight_loader"):
                layer._weight_loaders[pname] = param.weight_loader

    weight_packed_data = layer.weight_packed.data
    weight_scale_data = layer.weight_scale.data
    input_global_scale_data = layer.input_global_scale.data
    weight_global_scale_data = layer.weight_global_scale.data

    input_global_scale_inv = input_global_scale_data.max().to(torch.float32)
    processed_input_global_scale = (1.0 / input_global_scale_inv).to(torch.float32)
    weight_global_scale_max = weight_global_scale_data.max().to(torch.float32)
    processed_weight_global_scale = 1.0 / weight_global_scale_max
    processed_alpha = processed_input_global_scale * processed_weight_global_scale

    layer.weight = Parameter(weight_packed_data.clone(), requires_grad=False)
    if hasattr(layer, "weight_packed"):
        delattr(layer, "weight_packed")
    layer.weight_scale = Parameter(weight_scale_data.clone(), requires_grad=False)

    if is_first_call:
        layer.input_global_scale = Parameter(processed_input_global_scale, requires_grad=False)
        layer.weight_global_scale = Parameter(processed_weight_global_scale, requires_grad=False)
        layer.input_global_scale_inv = Parameter(input_global_scale_inv, requires_grad=False)
        layer.alpha = Parameter(processed_alpha, requires_grad=False)
        layer._marlin_tensor_refs = {
            "weight": layer.weight.data,
            "weight_scale": layer.weight_scale.data,
            "input_global_scale": layer.input_global_scale.data,
            "weight_global_scale": layer.weight_global_scale.data,
            "input_global_scale_inv": layer.input_global_scale_inv.data,
            "alpha": layer.alpha.data,
        }
    else:
        layer.weight.data.copy_(weight_packed_data)
        layer.weight_scale.data.copy_(weight_scale_data)
        refs = layer._marlin_tensor_refs
        refs["input_global_scale"].copy_(processed_input_global_scale)
        refs["weight_global_scale"].copy_(processed_weight_global_scale)
        refs["input_global_scale_inv"].copy_(input_global_scale_inv)
        refs["alpha"].copy_(processed_alpha)
        layer.input_global_scale = Parameter(refs["input_global_scale"], requires_grad=False)
        layer.weight_global_scale = Parameter(refs["weight_global_scale"], requires_grad=False)
        layer.input_global_scale_inv = Parameter(refs["input_global_scale_inv"], requires_grad=False)
        layer.alpha = Parameter(refs["alpha"], requires_grad=False)

    if hasattr(self, "kernel"):
        self.kernel.process_weights_after_loading(layer)
    else:
        from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
            convert_to_nvfp4_linear_kernel_format,
        )

        convert_to_nvfp4_linear_kernel_format(self.backend, layer)

    _update_w4a4_tensor_ref(layer, "weight", layer.weight)
    _update_w4a4_tensor_ref(layer, "weight_scale", layer.weight_scale)


def patched_w4a4_process_weights_after_loading(self, layer: torch.nn.Module) -> None:
    """Patched process_weights_after_loading for W4A4 Dense (all backends)."""
    if _w4a4_uses_kernel_api(self):
        _patched_w4a4_kernel_process_weights(self, layer)
    else:
        _patched_w4a4_legacy_process_weights(self, layer)


def _marlin_repack_experts(packed, perm, size_k, size_n, num_experts):
    """Repack weight for each expert into Marlin format and stack."""
    import vllm._custom_ops as ops

    result = []
    for i in range(num_experts):
        qweight = packed[i].view(torch.int32).T.contiguous()
        result.append(
            ops.gptq_marlin_repack(
                b_q_weight=qweight,
                perm=perm,
                size_k=size_k,
                size_n=size_n,
                num_bits=4,
                is_a_8bit=False,
            )
        )
    return torch.stack(result)


def _marlin_process_scales_experts(scale_hf, param_dtype, size_k, size_n, group_size, num_experts):
    """Process scales for each expert into Marlin format and stack."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import marlin_permute_scales

    result = []
    scales = scale_hf.to(param_dtype)
    for i in range(num_experts):
        s = marlin_permute_scales(
            s=scales[i].T,
            size_k=size_k,
            size_n=size_n,
            group_size=group_size,
            is_a_8bit=False,
        )
        processed_scale, _ = _call_nvfp4_marlin_process_scales(s, param_dtype)
        result.append(processed_scale)
    return torch.stack(result)


def _init_nvfp4_moe_kernel(self, layer: torch.nn.Module) -> None:
    """Create NVFP4 MoE kernel on self.moe_kernel (vLLM >=0.20 API)."""
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import make_nvfp4_moe_kernel

    self.moe_quant_config = self.get_fused_moe_quant_config(layer)
    if self.moe_quant_config is None:
        return
    if self.moe.moe_parallel_config.use_all2all_kernels and not self.moe.moe_parallel_config.use_naive_all2all_kernels:
        return
    assert self.experts_cls is not None

    kernel_kwargs = {
        "moe_quant_config": self.moe_quant_config,
        "moe_config": self.moe,
        "experts_cls": self.experts_cls,
    }
    sig = inspect.signature(make_nvfp4_moe_kernel)
    if "routing_tables" in sig.parameters:
        routing_tables = (
            layer._maybe_init_expert_routing_tables() if hasattr(layer, "_maybe_init_expert_routing_tables") else None
        )
        kernel_kwargs["routing_tables"] = routing_tables
    if "shared_experts" in sig.parameters:
        kernel_kwargs["shared_experts"] = getattr(layer, "shared_experts", None)

    self.moe_kernel = make_nvfp4_moe_kernel(**kernel_kwargs)
    # Backward compat for vLLM versions that still read self.kernel.
    self.kernel = self.moe_kernel
    if hasattr(self.moe_kernel, "fused_experts"):
        self.moe_kernel.fused_experts.process_weights_after_loading(layer)


def _process_nvfp4_moe_marlin(self, layer: torch.nn.Module, is_first_call: bool) -> None:
    """Process MoE layer with MARLIN backend (W4A16)."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import marlin_make_workspace_new

    group_size = 16
    e = layer.num_experts
    k = layer.hidden_size
    n = layer.intermediate_size_per_partition
    device = layer.w13_weight_packed.device
    param_dtype = layer.params_dtype
    w13_num_shards = 2 if self.moe.is_act_and_mul else 1

    if is_first_call:
        layer.workspace = marlin_make_workspace_new(device, 4)

    perm = torch.empty(0, dtype=torch.int, device=device)

    if self.moe.is_act_and_mul and not torch.allclose(
        layer.w13_weight_global_scale[:, 0], layer.w13_weight_global_scale[:, 1]
    ):
        logger.warning("w1_weight_global_scale must match w3_weight_global_scale. Accuracy may be affected.")

    size_n_w13, size_k_w13 = n * w13_num_shards, k
    size_n_w2, size_k_w2 = k, n

    w13_weight_marlin = _marlin_repack_experts(layer.w13_weight_packed.data, perm, size_k_w13, size_n_w13, e)
    w2_weight_marlin = _marlin_repack_experts(layer.w2_weight_packed.data, perm, size_k_w2, size_n_w2, e)
    w13_weight_scale_marlin = _marlin_process_scales_experts(
        layer.w13_weight_scale.data, param_dtype, size_k_w13, size_n_w13, group_size, e
    )
    w2_weight_scale_marlin = _marlin_process_scales_experts(
        layer.w2_weight_scale.data, param_dtype, size_k_w2, size_n_w2, group_size, e
    )

    # Process global scales
    w13_scale_2 = 1.0 / layer.w13_weight_global_scale[:, 0]
    w2_scale_2 = 1.0 / layer.w2_weight_global_scale.data
    w13_scale_2_processed = _call_nvfp4_marlin_process_global_scale(w13_scale_2, param_dtype)
    w2_scale_2_processed = _call_nvfp4_marlin_process_global_scale(w2_scale_2, param_dtype)

    # Update parameters
    if is_first_call:
        layer.w13_weight = Parameter(w13_weight_marlin, requires_grad=False)
        layer.w2_weight = Parameter(w2_weight_marlin, requires_grad=False)
        layer.w13_weight_scale = Parameter(w13_weight_scale_marlin, requires_grad=False)
        layer.w2_weight_scale = Parameter(w2_weight_scale_marlin, requires_grad=False)
        layer.w13_weight_scale_2 = Parameter(w13_scale_2_processed, requires_grad=False)
        layer.w2_weight_scale_2 = Parameter(w2_scale_2_processed, requires_grad=False)
        if not hasattr(layer, "_marlin_tensor_refs"):
            layer._marlin_tensor_refs = {}
        layer._marlin_tensor_refs["w13_weight_scale"] = layer.w13_weight_scale.data
        layer._marlin_tensor_refs["w2_weight_scale"] = layer.w2_weight_scale.data
    else:
        layer.w13_weight.data.copy_(w13_weight_marlin)
        layer.w2_weight.data.copy_(w2_weight_marlin)
        layer.w13_weight_scale_2.data.copy_(w13_scale_2_processed)
        layer.w2_weight_scale_2.data.copy_(w2_scale_2_processed)
        w13_marlin_ref = layer._marlin_tensor_refs.get("w13_weight_scale")
        w2_marlin_ref = layer._marlin_tensor_refs.get("w2_weight_scale")
        if w13_marlin_ref is not None:
            w13_marlin_ref.copy_(w13_weight_scale_marlin)
            layer.w13_weight_scale = Parameter(w13_marlin_ref, requires_grad=False)
        else:
            logger.warning("MoE: _marlin_tensor_refs['w13_weight_scale'] not found")
            layer.w13_weight_scale.data.copy_(w13_weight_scale_marlin)
        if w2_marlin_ref is not None:
            w2_marlin_ref.copy_(w2_weight_scale_marlin)
            layer.w2_weight_scale = Parameter(w2_marlin_ref, requires_grad=False)
        else:
            logger.warning("MoE: _marlin_tensor_refs['w2_weight_scale'] not found")
            layer.w2_weight_scale.data.copy_(w2_weight_scale_marlin)

    layer.w13_input_scale = None
    layer.w2_input_scale = None

    _init_nvfp4_moe_kernel(self, layer)


def _process_nvfp4_moe_flashinfer_cutlass(self, layer: torch.nn.Module, is_first_call: bool) -> None:
    """Process MoE layer with FlashInfer/CUTLASS backend (W4A4)."""
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import convert_to_nvfp4_moe_kernel_format
    from vllm.model_executor.utils import replace_parameter

    w13_packed = layer.w13_weight_packed.data
    w2_packed = layer.w2_weight_packed.data
    w13_scale_hf = layer.w13_weight_scale.data
    w2_scale_hf = layer.w2_weight_scale.data

    if self.moe.is_act_and_mul and not torch.allclose(
        layer.w13_weight_global_scale[:, 0], layer.w13_weight_global_scale[:, 1]
    ):
        logger.warning("w1_weight_global_scale must match w3_weight_global_scale. Accuracy may be affected.")
    w13_weight_global_scale = layer.w13_weight_global_scale[:, 0].contiguous()

    w13_temp = Parameter(w13_packed.clone(), requires_grad=False)
    w2_temp = Parameter(w2_packed.clone(), requires_grad=False)

    if is_first_call:
        layer.w13_weight = w13_temp
        layer.w2_weight = w2_temp

    (
        w13,
        w13_scale,
        w13_scale_2,
        a13_scale,
        w2,
        w2_scale,
        w2_scale_2,
        a2_scale,
    ) = convert_to_nvfp4_moe_kernel_format(
        nvfp4_backend=self.nvfp4_backend,
        layer=layer,
        w13=w13_temp,
        w13_scale=w13_scale_hf,
        w13_scale_2=(1.0 / w13_weight_global_scale),
        a13_scale=(1.0 / layer.w13_input_global_scale),
        w2=w2_temp,
        w2_scale=w2_scale_hf,
        w2_scale_2=(1.0 / layer.w2_weight_global_scale),
        a2_scale=(1.0 / layer.w2_input_global_scale),
        is_act_and_mul=self.moe.is_act_and_mul,
    )

    # Update parameters
    if is_first_call:
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)
        layer.w13_weight_scale = Parameter(w13_scale, requires_grad=False)
        layer.w2_weight_scale = Parameter(w2_scale, requires_grad=False)
        if not hasattr(layer, "_marlin_tensor_refs"):
            layer._marlin_tensor_refs = {}
        layer._marlin_tensor_refs["w13_weight_scale"] = layer.w13_weight_scale.data
        layer._marlin_tensor_refs["w2_weight_scale"] = layer.w2_weight_scale.data
    else:
        layer.w13_weight.data.copy_(w13.data)
        layer.w2_weight.data.copy_(w2.data)
        w13_scale_ref = layer._marlin_tensor_refs.get("w13_weight_scale")
        w2_scale_ref = layer._marlin_tensor_refs.get("w2_weight_scale")
        if w13_scale_ref is not None:
            w13_scale_ref.copy_(w13_scale)
            layer.w13_weight_scale = Parameter(w13_scale_ref, requires_grad=False)
        else:
            logger.warning("MoE W4A4: _marlin_tensor_refs['w13_weight_scale'] not found")
            layer.w13_weight_scale.data.copy_(w13_scale)
        if w2_scale_ref is not None:
            w2_scale_ref.copy_(w2_scale)
            layer.w2_weight_scale = Parameter(w2_scale_ref, requires_grad=False)
        else:
            logger.warning("MoE W4A4: _marlin_tensor_refs['w2_weight_scale'] not found")
            layer.w2_weight_scale.data.copy_(w2_scale)

    layer.w13_weight_scale_2 = w13_scale_2
    layer.w2_weight_scale_2 = w2_scale_2
    layer.w13_input_scale = a13_scale
    layer.w2_input_scale = a2_scale

    _init_nvfp4_moe_kernel(self, layer)


# MoE NVFP4 Patches (entry points)
def patched_nvfp4_moe_process_weights_after_loading(self, layer: torch.nn.Module) -> None:
    """Patched process_weights_after_loading for NVFP4 MoE layer."""
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend

    is_first_call = _check_first_call(layer)

    # Save metadata (first call only)
    if is_first_call:
        save_param_meta(layer, "w13_weight_packed")
        save_param_meta(layer, "w2_weight_packed")
        save_param_meta(layer, "w13_weight_scale")
        save_param_meta(layer, "w2_weight_scale")
        save_param_meta(layer, "w13_input_global_scale")
        save_param_meta(layer, "w2_input_global_scale")
        if not hasattr(layer, "_weight_loaders"):
            layer._weight_loaders = {}
        for pname in ["w13_weight_packed", "w2_weight_packed", "w13_weight_scale", "w2_weight_scale"]:
            param = getattr(layer, pname, None)
            if param is not None and hasattr(param, "weight_loader"):
                layer._weight_loaders[pname] = param.weight_loader

    is_marlin = self.nvfp4_backend == NvFp4MoeBackend.MARLIN
    if is_marlin:
        _process_nvfp4_moe_marlin(self, layer, is_first_call)
    else:
        _process_nvfp4_moe_flashinfer_cutlass(self, layer, is_first_call)

    # Delete HF parameters
    if hasattr(layer, "w13_weight_packed"):
        delattr(layer, "w13_weight_packed")
    if hasattr(layer, "w2_weight_packed"):
        delattr(layer, "w2_weight_packed")


_MOE_PKG = (
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe"
)


def _resolve_moe_nvfp4_patch_target() -> str:
    """Resolve MoE NVFP4 patch target across vLLM <=0.19 and >=0.20 layouts."""
    moe_mod = importlib.import_module(_MOE_PKG)
    if hasattr(moe_mod, "CompressedTensorsW4A4Nvfp4MoEMethod"):
        cls = moe_mod.CompressedTensorsW4A4Nvfp4MoEMethod
    else:
        nvfp4_mod = importlib.import_module(f"{_MOE_PKG}.compressed_tensors_moe_w4a4_nvfp4")
        cls = nvfp4_mod.CompressedTensorsW4A4Nvfp4MoEMethod
    return f"{cls.__module__}.{cls.__qualname__}.process_weights_after_loading"


def _build_patch_targets():
    return [
        # Dense W4A16
        (
            "vllm.model_executor.layers.quantization.compressed_tensors.schemes."
            "compressed_tensors_w4a16_nvfp4.CompressedTensorsW4A16Fp4.process_weights_after_loading",
            patched_w4a16_process_weights_after_loading,
        ),
        # Dense W4A4
        (
            "vllm.model_executor.layers.quantization.compressed_tensors.schemes."
            "compressed_tensors_w4a4_nvfp4.CompressedTensorsW4A4Fp4.process_weights_after_loading",
            patched_w4a4_process_weights_after_loading,
        ),
        # MoE NVFP4 (module path differs between vLLM <=0.19 and >=0.20)
        (
            _resolve_moe_nvfp4_patch_target(),
            patched_nvfp4_moe_process_weights_after_loading,
        ),
    ]

_applied_patches = []


def apply_qat_patches():
    """Apply NVFP4 patches to support dynamic weight updates. Call before model loading."""
    global _applied_patches

    if _applied_patches:
        logger.warning("QAT patches already applied, skipping")
        return _applied_patches

    logger.info("Applying NVFP4 patches for dynamic weight loading...")

    for target, replacement in _build_patch_targets():
        p = patch(target, replacement)
        _applied_patches.append(p)
        p.start()

    logger.info(f"Applied {len(_applied_patches)} NVFP4 patches for dynamic weight loading")
    return _applied_patches


_MOE_KERNEL_WEIGHT_NAMES = ("w13_weight", "w2_weight")
_MOE_PACKED_TO_LOAD_ALIAS = (
    ("w13_weight_packed", "w13_weight"),
    ("w2_weight_packed", "w2_weight"),
)


def _reset_moe_quant_method_kernels(quant_method) -> None:
    if quant_method is None:
        return
    for attr in ("moe_kernel", "kernel"):
        if hasattr(quant_method, attr):
            setattr(quant_method, attr, None)


def _prepare_moe_layer_for_hf_reload(module: torch.nn.Module) -> None:
    """Drop kernel-layout fused weights so the next load_weights sees HF buffers."""
    for name in _MOE_KERNEL_WEIGHT_NAMES:
        if hasattr(module, name):
            delattr(module, name)
    _reset_moe_quant_method_kernels(getattr(module, "quant_method", None))


def _alias_moe_packed_weights_for_load(module: torch.nn.Module) -> None:
    """Mirror vLLM pre-process_weights layout: w13/w2_weight alias HF packed tensors."""
    weight_loaders = getattr(module, "_weight_loaders", {})
    for packed_name, alias_name in _MOE_PACKED_TO_LOAD_ALIAS:
        packed = getattr(module, packed_name, None)
        if packed is None:
            continue
        loader = weight_loaders.get(packed_name) or getattr(packed, "weight_loader", None)
        alias = Parameter(packed.data, requires_grad=False)
        if loader is not None:
            alias.weight_loader = loader
        module.register_parameter(alias_name, alias)


_MOE_LAYER_INPUT_SCALES = ("w13_input_global_scale", "w2_input_global_scale")


def _ensure_moe_input_global_scale_params(module: torch.nn.Module, device=None) -> None:
    """Ensure MoE layer has reloadable w13/w2 input scale parameters."""
    dev = device or get_device_name()
    for scale_name in _MOE_LAYER_INPUT_SCALES:
        if hasattr(module, scale_name):
            continue
        param = Parameter(torch.tensor([1.0], dtype=torch.float32, device=dev), requires_grad=False)
        module.register_parameter(scale_name, param)


def _install_moe_input_scale_reload_hooks(module: torch.nn.Module, device=None) -> None:
    """Allow FSDP export keys (experts.w13/w2_input_global_scale) to reload MoE layer scales."""
    _ensure_moe_input_global_scale_params(module, device=device)
    for scale_name in _MOE_LAYER_INPUT_SCALES:
        param = getattr(module, scale_name)

        def _make_loader(target_param):
            def _loader(param_like, loaded_weight, name=None):
                target_param.data.copy_(loaded_weight.reshape_as(target_param.data).to(target_param.dtype))

            return _loader

        if not hasattr(param, "weight_loader") or param.weight_loader is None:
            param.weight_loader = _make_loader(param)


def _aggregate_moe_expert_input_scales(module: torch.nn.Module) -> None:
    """If only per-expert HF input scales were loaded, promote min scale to layer w13/w2 buffers."""
    w13_candidates = []
    w2_candidates = []
    for name, param in module.named_parameters(recurse=False):
        if name.endswith("input_global_scale") and param.numel() == 1:
            if "w13" in name or name.startswith("w13"):
                w13_candidates.append(param.data)
    for child_name, child in module.named_modules():
        if child is module:
            continue
        if not child_name.endswith(".gate_proj") and not child_name.endswith(".up_proj"):
            if child_name.endswith(".down_proj"):
                scale = getattr(child, "input_global_scale", None)
                if scale is not None and scale.numel() == 1:
                    w2_candidates.append(scale.data)
            continue
        scale = getattr(child, "input_global_scale", None)
        if scale is not None and scale.numel() == 1:
            w13_candidates.append(scale.data)

    if w13_candidates and hasattr(module, "w13_input_global_scale"):
        fused = torch.min(torch.stack([s.float().reshape(1) for s in w13_candidates]))
        module.w13_input_global_scale.data.copy_(fused.to(module.w13_input_global_scale.dtype))
    if w2_candidates and hasattr(module, "w2_input_global_scale"):
        fused = torch.min(torch.stack([s.float().reshape(1) for s in w2_candidates]))
        module.w2_input_global_scale.data.copy_(fused.to(module.w2_input_global_scale.dtype))


def prepare_qat_for_load_weights(model, device=None):
    """
    Prepare QAT model for weight loading. Call ONCE before multi-bucket weight loading.

    Args:
        model: vLLM model
        device: Device for created parameters
    """
    inner_model = model
    if hasattr(model, "model"):
        inner_model = model.model

    param_meta = ParamMetaDict(inner_model, device=device)

    param_meta.prepare_for_reload()
    logger.info(f"[prepare_qat] Tensor swap prepared for {len(param_meta._tensor_swap_layers)} layers")

    moe_reset_count = 0
    for cache_entry in param_meta._layer_meta_cache.values():
        meta = cache_entry["meta"]
        if "w13_weight_packed" not in meta:
            continue
        _prepare_moe_layer_for_hf_reload(cache_entry["module"])
        moe_reset_count += 1

    # Rebuild deleted (W4A16/W4A4) or kernel-overwritten params back to HF format
    rebuilt_count = 0
    for cache_entry in param_meta._layer_meta_cache.values():
        module = cache_entry["module"]
        for param_name, pm in cache_entry["meta"].items():
            existing = getattr(module, param_name, None)
            if existing is not None:
                hf_shape = tuple(pm["shape"])
                hf_dtype = pm["dtype"]
                if (
                    tuple(existing.shape) == hf_shape
                    and existing.dtype == hf_dtype
                    and hasattr(existing, "weight_loader")
                ):
                    continue
            new_param = _create_param_from_meta(module, param_name, pm, device)
            module.register_parameter(param_name, new_param)
            rebuilt_count += 1

    alias_count = 0
    for cache_entry in param_meta._layer_meta_cache.values():
        if "w13_weight_packed" not in cache_entry["meta"]:
            continue
        _alias_moe_packed_weights_for_load(cache_entry["module"])
        _install_moe_input_scale_reload_hooks(cache_entry["module"], device=device)
        alias_count += 1

    logger.info(
        f"[prepare_qat] MoE HF reload prep: reset={moe_reset_count}, "
        f"rebuilt={rebuilt_count}, aliased={alias_count}"
    )
    inner_model._param_meta_for_restore = param_meta
    return param_meta


def manual_process_weights_after_loading(model):
    """Trigger weight post-processing for all quantized layers after load_weights."""
    dense_count = 0
    moe_count = 0

    actual_model = model
    if hasattr(model, "model"):
        actual_model = model.model

    for module in actual_model.modules():
        if hasattr(module, "scheme"):
            module.scheme.process_weights_after_loading(module)
            dense_count += 1

        quant_method = getattr(module, "quant_method", None)
        if quant_method is not None and not hasattr(module, "scheme"):
            if hasattr(quant_method, "process_weights_after_loading"):
                # Skip KV cache quantization methods
                if "KVCache" in quant_method.__class__.__name__:
                    continue
                quant_method.process_weights_after_loading(module)
                if "w13_weight_packed" in getattr(module, "_hf_param_meta", {}):
                    _aggregate_moe_expert_input_scales(module)
                moe_count += 1

    logger.debug(f"Processed {dense_count} dense layers, {moe_count} MoE layers")
    return dense_count + moe_count


__all__ = [
    "apply_qat_patches",
    "prepare_qat_for_load_weights",
    "manual_process_weights_after_loading",
]
