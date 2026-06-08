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

"""Compatibility helpers for compressed-tensors package layout/API changes."""

from __future__ import annotations

import importlib
from typing import Any, Protocol

import torch

_NVFP4_COMPRESSOR_MODULES = (
    # <=0.12: quantized_compressors layout
    "compressed_tensors.compressors.quantized_compressors.nvfp4_quantized",
    "compressed_tensors.compressors.quantized_compressors.fp4_quantized",
    # >=0.13/main: flattened compressor layout
    "compressed_tensors.compressors.nvfp4.base",
    "compressed_tensors.compressors.nvfp4",
)

_PACK_FP4_MODULES = (
    "compressed_tensors.compressors.nvfp4.helpers",
    "compressed_tensors.compressors.quantized_compressors.nvfp4_quantized",
)


class NVFP4WeightPacker(Protocol):
    def compress_weight(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        global_scale: torch.Tensor,
        quantization_args: Any,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        ...


def _import_symbol(module_paths: tuple[str, ...], symbol: str):
    last_error: ModuleNotFoundError | None = None
    for module_path in module_paths:
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            last_error = exc
            continue
        if hasattr(module, symbol):
            return getattr(module, symbol)
    if last_error is not None:
        raise ImportError(
            f"Cannot import {symbol} from compressed-tensors. "
            "Install compressed-tensors>=0.11.0."
        ) from last_error
    raise ImportError(
        f"Cannot import {symbol} from compressed-tensors. "
        "Install compressed-tensors>=0.11.0."
    )


class _DirectNVFP4WeightPacker:
    """Fallback packer using quantize + pack_fp4_to_uint8 for newer APIs."""

    def __init__(self):
        from compressed_tensors.quantization.lifecycle.forward import quantize

        self._quantize = quantize
        self._pack_fp4_to_uint8 = _import_symbol(_PACK_FP4_MODULES, "pack_fp4_to_uint8")

    def compress_weight(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        global_scale: torch.Tensor,
        quantization_args: Any,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        quantized_weight = self._quantize(
            x=weight,
            scale=scale,
            global_scale=global_scale,
            zero_point=kwargs.get("zero_point"),
            args=quantization_args,
        )
        return {"weight_packed": self._pack_fp4_to_uint8(quantized_weight)}


def create_nvfp4_weight_packer() -> NVFP4WeightPacker:
    """Create a version-agnostic NVFP4 weight packer."""
    for module_path in _NVFP4_COMPRESSOR_MODULES:
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError:
            continue

        compressor_cls = getattr(module, "NVFP4PackedCompressor", None)
        if compressor_cls is None:
            continue

        try:
            compressor = compressor_cls()
        except TypeError:
            continue

        if hasattr(compressor, "compress_weight"):
            return compressor

    return _DirectNVFP4WeightPacker()


__all__ = ["create_nvfp4_weight_packer"]
