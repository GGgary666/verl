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

"""QAT FakeQuantized Linear module for NVFP4 (W4A4/W4A16) with FSDP compatibility.

Includes Triton kernels for high-performance FP4 quantization.
"""

import atexit
import os
import time
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["QATLinear", "QATMode"]


import triton
import triton.language as tl
from triton.language.target_info import cuda_capability_geq

try:
    from torchao.prototype.mx_formats.nvfp4_tensor import per_tensor_amax_to_scale as _torchao_per_tensor_amax_to_scale
except Exception:
    _torchao_per_tensor_amax_to_scale = None

try:
    from vllm._custom_ops import cutlass_scaled_fp4_mm, scaled_fp4_quant
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
        dequantize_to_dtype,
    )

    _VLLM_NVFP4_AVAILABLE = True
except Exception:
    cutlass_scaled_fp4_mm = None
    scaled_fp4_quant = None
    dequantize_to_dtype = None
    _VLLM_NVFP4_AVAILABLE = False

try:
    from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm
except Exception:
    flashinfer_scaled_fp4_mm = None

_TORCH_TO_TL_DTYPE = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}
FP4_E2M1_MAX: float = 6.0
FP8_E4M3_MAX: float = 448.0
MXFP_BLOCK_SIZE = tl.constexpr(16)


def _per_tensor_amax_to_scale(amax: torch.Tensor) -> torch.Tensor:
    if _torchao_per_tensor_amax_to_scale is not None:
        return _torchao_per_tensor_amax_to_scale(amax)
    return amax.to(torch.float32) / (FP4_E2M1_MAX * FP8_E4M3_MAX)


def _resolve_vllm_nvfp4_backend() -> str:
    backend = os.environ.get("VERL_QAT_VLLM_NVFP4_BACKEND", "cutlass").strip().lower()
    if not backend:
        backend = "cutlass"
    if backend == "flashinfer":
        backend = "flashinfer-cutlass"
    return backend


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class _SimpleQuantProfiler:
    """Best-effort coarse profiler for torchao_real quantization hotspots."""

    def __init__(self) -> None:
        self.enabled = _env_flag("VERL_QAT_SIMPLE_PROFILE", False)
        rank = os.environ.get("RANK", "")
        self._should_print = rank in {"", "0"}
        self._totals_ms: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        if self.enabled and self._should_print:
            atexit.register(self.report)

    def add(self, name: str, elapsed_ms: float) -> None:
        if not self.enabled:
            return
        self._totals_ms[name] = self._totals_ms.get(name, 0.0) + elapsed_ms
        self._counts[name] = self._counts.get(name, 0) + 1

    def timed_call(self, name: str, fn, *args, **kwargs):
        if not self.enabled:
            return fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.add(name, elapsed_ms)
        return out

    def report(self) -> None:
        if not self.enabled or not self._should_print or not self._totals_ms:
            return
        ordered_keys = [
            "fwd_quant_x_ms",
            "fwd_quant_w_ms",
            "fwd_gemm_ms",
            "bwd_dequant_x_ms",
            "bwd_dequant_w_ms",
            "bwd_mm_ms",
        ]
        print("\n=== QAT Simple Quant Profile (torchao_real) ===")
        fwd_quant_total = 0.0
        bwd_quant_total = 0.0
        for key in ordered_keys:
            total = self._totals_ms.get(key, 0.0)
            count = self._counts.get(key, 0)
            if count == 0:
                continue
            avg = total / count
            print(f"{key}: total_ms={total:.3f}, calls={count}, avg_ms={avg:.3f}")
            if key.startswith("fwd_quant_"):
                fwd_quant_total += total
            if key.startswith("bwd_dequant_"):
                bwd_quant_total += total
        if fwd_quant_total > 0.0 or bwd_quant_total > 0.0:
            print(f"fwd_quant_total_ms={fwd_quant_total:.3f}")
            print(f"bwd_dequant_total_ms={bwd_quant_total:.3f}")


_SIMPLE_QPROF = _SimpleQuantProfiler()


@triton.jit
def _fp4_fake_quant_kernel_legacy(
    x_ptr,
    y_ptr,
    M,
    N,
    global_scale_ptr,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    BLOCK_SIZE: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    NUM_FP4_BLOCKS: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    FP4_MAX: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    row_start = pid_m * TILE_M
    col_start = pid_n * TILE_N

    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(M, N),
        strides=(stride_xm, stride_xn),
        offsets=(row_start, col_start),
        block_shape=(TILE_M, TILE_N),
        order=(1, 0),
    )
    y_block_ptr = tl.make_block_ptr(
        base=y_ptr,
        shape=(M, N),
        strides=(stride_ym, stride_yn),
        offsets=(row_start, col_start),
        block_shape=(TILE_M, TILE_N),
        order=(1, 0),
    )

    global_scale = tl.load(global_scale_ptr).to(tl.float32)
    global_scale_safe = tl.where(global_scale > 0.0, global_scale, 1e-12)

    tile = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    tile_reshaped = tl.reshape(tile, (TILE_M, NUM_FP4_BLOCKS, BLOCK_SIZE))
    x_abs = tl.abs(tile_reshaped)

    block_max = tl.max(x_abs, axis=2, keep_dims=True)
    block_max_scaled = block_max / (FP4_MAX * global_scale_safe)
    block_max_scaled = tl.minimum(block_max_scaled, FP8_MAX)
    block_max_quant = block_max_scaled.to(tl.float8e4nv).to(tl.float32) * global_scale
    block_max_quant = tl.where(block_max_quant >= 1e-5, block_max_quant, 1.0)

    block_max_quant_broadcast = tl.broadcast_to(block_max_quant, (TILE_M, NUM_FP4_BLOCKS, BLOCK_SIZE))
    abs_scaled = x_abs / block_max_quant_broadcast

    q_val = tl.where(
        abs_scaled <= 0.25,
        0.0,
        tl.where(
            abs_scaled < 0.75,
            0.5,
            tl.where(
                abs_scaled <= 1.25,
                1.0,
                tl.where(
                    abs_scaled < 1.75,
                    1.5,
                    tl.where(
                        abs_scaled <= 2.5,
                        2.0,
                        tl.where(abs_scaled < 3.5, 3.0, tl.where(abs_scaled <= 5.0, 4.0, FP4_MAX)),
                    ),
                ),
            ),
        ),
    )

    x_rescaled = q_val * block_max_quant_broadcast
    x_rescaled = tl.where(tile_reshaped >= 0, x_rescaled, -x_rescaled)
    tile_quant = tl.reshape(x_rescaled, (TILE_M, TILE_N))

    tl.store(y_block_ptr, tile_quant.to(OUT_DTYPE), boundary_check=(0, 1))


@triton.jit
def _compute_quant_and_scale(
    src_tensor,
    valid_src_mask,
    mx_tensor_dtype: tl.constexpr = tl.uint8,
    use_global_sf=True,
    external_global_amax=0.0,
    use_external_global_amax: tl.constexpr = False,
):
    BLOCK_SIZE_OUT_DIM: tl.constexpr = src_tensor.shape[0]
    BLOCK_SIZE_QUANT_DIM: tl.constexpr = src_tensor.shape[1]
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = src_tensor.shape[1] // MXFP_BLOCK_SIZE
    is_fp4: tl.constexpr = mx_tensor_dtype == tl.uint8

    tl.static_assert(
        is_fp4 or mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5,
        "mx_tensor_dtype must be uint8, float8e4nv, or float8e5",
    )

    # Explicit cast to fp32 since most ops are not supported on bfloat16.
    f32_tensor = src_tensor.to(tl.float32)
    abs_tensor = tl.abs(f32_tensor)
    abs_tensor = tl.where(valid_src_mask, abs_tensor, -1.0)
    abs_tensor = tl.reshape(abs_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, MXFP_BLOCK_SIZE])

    if use_global_sf:
        if use_external_global_amax:
            global_max_val = external_global_amax.to(tl.float32)
        else:
            global_max_val = tl.max(abs_tensor)
        global_max_val = tl.maximum(global_max_val, 1e-8)
        s_enc = (6 * 448) / global_max_val
        s_dec = 1 / s_enc
    else:
        s_dec = 1.0
        s_enc = 1.0

    max_val = tl.max(abs_tensor, axis=2, keep_dims=True)
    s_dec_b = max_val / 6
    s_dec_b_e4m3 = (s_dec_b * s_enc).to(tl.float8e4nv)
    s_enc_b = 1 / (s_dec_b_e4m3.to(tl.float32) * s_dec)

    f32_tensor = tl.reshape(f32_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, MXFP_BLOCK_SIZE])
    quant_tensor = f32_tensor * s_enc_b
    quant_tensor = quant_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM])
    quant_tensor = tl.where(valid_src_mask, quant_tensor, 0.0)
    dequant_scale = s_dec_b_e4m3.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE])

    if is_fp4 and cuda_capability_geq(10, 0):
        pairs = tl.reshape(quant_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM // 2, 2])
        lo_f, hi_f = tl.split(pairs)
        lo_f32 = lo_f.to(tl.float32)
        hi_f32 = hi_f.to(tl.float32)
        out_tensor = tl.inline_asm_elementwise(
            """
            {
                .reg .b8 r;
                cvt.rn.satfinite.e2m1x2.f32 r, $1, $2;
                mov.b32 $0, {r, r, r, r};
            }
            """,
            constraints="=r,f,f",
            args=[hi_f32, lo_f32],
            dtype=tl.uint8,
            is_pure=True,
            pack=1,
        )
    elif is_fp4:
        quant_tensor = quant_tensor.to(tl.uint32, bitcast=True)
        signs = quant_tensor & 0x80000000
        exponents = (quant_tensor >> 23) & 0xFF
        mantissas_orig = quant_tensor & 0x7FFFFF

        E8_BIAS = 127
        E2_BIAS = 1
        is_subnormal = exponents < E8_BIAS
        adjusted_exponents = tl.core.sub(E8_BIAS, exponents + 1, sanitize_overflow=False)
        mantissas_pre = 0x400000 | (mantissas_orig >> 1)
        mantissas = tl.where(is_subnormal, mantissas_pre >> adjusted_exponents, mantissas_orig)

        exponents = tl.maximum(exponents, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)

        m2bits = mantissas >> 21
        lsb_keep = (m2bits >> 1) & 0x1
        guard = m2bits & 0x1
        IS_SRC_FP32: tl.constexpr = src_tensor.dtype == tl.float32
        if IS_SRC_FP32:
            bit0_dropped = (mantissas_orig & 0x1) != 0
            mask = (1 << tl.minimum(adjusted_exponents, 31)) - 1
            dropped_post = (mantissas_pre & mask) != 0
            sticky = is_subnormal & (bit0_dropped | dropped_post)
            sticky |= ((mantissas & 0x1FFFFF) != 0).to(tl.uint32)
        else:
            sticky = ((mantissas & 0x1FFFFF) != 0).to(tl.uint32)
        round_inc = guard & (sticky | lsb_keep)
        e2m1_tmp = tl.minimum((((exponents << 2) | m2bits) + round_inc) >> 1, 0x7)
        e2m1_value = ((signs >> 28) | e2m1_tmp).to(tl.uint8)

        e2m1_value = tl.reshape(e2m1_value, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM // 2, 2])
        evens, odds = tl.split(e2m1_value)
        out_tensor = evens | (odds << 4)
    else:
        out_tensor = quant_tensor.to(mx_tensor_dtype)

    return out_tensor, dequant_scale, s_dec


@triton.jit
def _compute_dequant(
    mx_tensor,
    scale,
    s_dec,
    BLOCK_SIZE_OUT_DIM: tl.constexpr,
    BLOCK_SIZE_QUANT_DIM: tl.constexpr,
    dst_dtype: tl.constexpr,
):
    tl.static_assert(
        BLOCK_SIZE_QUANT_DIM % MXFP_BLOCK_SIZE == 0,
        f"Block size along quantization block must be a multiple of {MXFP_BLOCK_SIZE=}",
    )
    mx_tensor_dtype: tl.constexpr = mx_tensor.dtype
    tl.static_assert(dst_dtype == tl.float16 or dst_dtype == tl.bfloat16 or dst_dtype == tl.float32)
    tl.static_assert(
        mx_tensor_dtype == tl.uint8
        or ((mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5) or mx_tensor_dtype == dst_dtype),
        "mx_tensor_ptr must be uint8 or float8 or dst_dtype",
    )
    tl.static_assert(scale.dtype == tl.float8e4nv, "scale must be float8e4nv")

    is_fp4: tl.constexpr = mx_tensor_dtype == tl.uint8
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = BLOCK_SIZE_QUANT_DIM // MXFP_BLOCK_SIZE

    if dst_dtype == tl.bfloat16:
        dst_scale = scale.to(tl.bfloat16)
    else:
        dst_scale = scale.to(tl.float32)
        if dst_dtype == tl.float16:
            dst_scale = dst_scale.to(tl.float16)

    intermediate_dtype: tl.constexpr = tl.bfloat16 if dst_dtype == tl.float32 else dst_dtype
    if cuda_capability_geq(10, 0):
        assert is_fp4
        packed_u32 = tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b8 in_8;
            .reg .f16x2 out;
            cvt.u8.u32 in_8, $1;
            cvt.rn.f16x2.e2m1x2 out, in_8;
            mov.b32 $0, out;
            }
            """,
            constraints="=r,r",
            args=[mx_tensor],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
        lo_u16 = (packed_u32 & 0xFFFF).to(tl.uint16)
        hi_u16 = (packed_u32 >> 16).to(tl.uint16)
        lo_f16 = lo_u16.to(tl.float16, bitcast=True)
        hi_f16 = hi_u16.to(tl.float16, bitcast=True)

        if intermediate_dtype == tl.float16:
            x0, x1 = lo_f16, hi_f16
        else:
            x0 = lo_f16.to(intermediate_dtype)
            x1 = hi_f16.to(intermediate_dtype)

        dst_tensor = tl.interleave(x0, x1)
    else:
        assert is_fp4
        dst_bias: tl.constexpr = 127 if intermediate_dtype == tl.bfloat16 else 15
        dst_0p5: tl.constexpr = 16128 if intermediate_dtype == tl.bfloat16 else 0x3800
        dst_m_bits: tl.constexpr = 7 if intermediate_dtype == tl.bfloat16 else 10
        em0 = mx_tensor & 0x07
        em1 = mx_tensor & 0x70
        x0 = (em0.to(tl.uint16) << (dst_m_bits - 1)) | ((mx_tensor & 0x08).to(tl.uint16) << 12)
        x1 = (em1.to(tl.uint16) << (dst_m_bits - 5)) | ((mx_tensor & 0x80).to(tl.uint16) << 8)
        x0 = tl.where((em0 & 0x06) != 0, x0 + ((dst_bias - 1) << dst_m_bits), x0)
        x1 = tl.where((em1 & 0x60) != 0, x1 + ((dst_bias - 1) << dst_m_bits), x1)
        x0 = tl.where(em0 == 0x01, dst_0p5 | (x0 & 0x8000), x0)
        x1 = tl.where(em1 == 0x10, dst_0p5 | (x1 & 0x8000), x1)
        dst_tensor = tl.interleave(x0, x1).to(intermediate_dtype, bitcast=True)

    dst_tensor = dst_tensor.to(dst_dtype)
    dst_tensor = dst_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, MXFP_BLOCK_SIZE])
    dst_scale = dst_scale.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 1])
    out_tensor = dst_tensor * dst_scale * s_dec
    if dst_dtype == tl.float32:
        max_fin = 3.4028234663852886e38
    elif dst_dtype == tl.bfloat16:
        max_fin = 3.3895313892515355e38
    else:
        tl.static_assert(dst_dtype == tl.float16)
        max_fin = 65504
    out_tensor = tl.clamp(out_tensor, min=-max_fin, max=max_fin)
    out_tensor = out_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM])
    out_tensor = out_tensor.to(dst_dtype)
    return out_tensor


@triton.jit
def _fake_quantize_nvfp4(
    src_tensor,
    valid_src_mask,
    BLOCK_SIZE_OUT_DIM: tl.constexpr,
    BLOCK_SIZE_QUANT_DIM: tl.constexpr,
    dst_dtype: tl.constexpr,
    mx_tensor_dtype: tl.constexpr = tl.uint8,
    use_global_sf: tl.constexpr = True,
    external_global_amax=0.0,
    use_external_global_amax: tl.constexpr = False,
):
    src_tensor, src_scale, src_s_dec = _compute_quant_and_scale(
        src_tensor=src_tensor,
        valid_src_mask=valid_src_mask,
        mx_tensor_dtype=mx_tensor_dtype,
        use_global_sf=use_global_sf,
        external_global_amax=external_global_amax,
        use_external_global_amax=use_external_global_amax,
    )
    return _compute_dequant(
        mx_tensor=src_tensor,
        scale=src_scale,
        s_dec=src_s_dec,
        BLOCK_SIZE_OUT_DIM=BLOCK_SIZE_OUT_DIM,
        BLOCK_SIZE_QUANT_DIM=BLOCK_SIZE_QUANT_DIM,
        dst_dtype=dst_dtype,
    )


@triton.jit
def _fp4_fake_quant_kernel_nvfp4(
    x_ptr,
    y_ptr,
    global_amax_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    BLOCK_SIZE: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    NUM_FP4_BLOCKS: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    use_global_sf: tl.constexpr = True,
):
    tl.static_assert(BLOCK_SIZE == MXFP_BLOCK_SIZE)
    tl.static_assert(TILE_N % BLOCK_SIZE == 0)
    tl.static_assert(NUM_FP4_BLOCKS == TILE_N // BLOCK_SIZE)

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    row_start = pid_m * TILE_M
    col_start = pid_n * TILE_N

    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(M, N),
        strides=(stride_xm, stride_xn),
        offsets=(row_start, col_start),
        block_shape=(TILE_M, TILE_N),
        order=(1, 0),
    )
    y_block_ptr = tl.make_block_ptr(
        base=y_ptr,
        shape=(M, N),
        strides=(stride_ym, stride_yn),
        offsets=(row_start, col_start),
        block_shape=(TILE_M, TILE_N),
        order=(1, 0),
    )

    tile = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
    external_global_amax = tl.load(global_amax_ptr).to(tl.float32)
    offs_m = row_start + tl.arange(0, TILE_M)
    offs_n = col_start + tl.arange(0, TILE_N)
    valid_src_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    tile_quant = _fake_quantize_nvfp4(
        src_tensor=tile,
        valid_src_mask=valid_src_mask,
        BLOCK_SIZE_OUT_DIM=TILE_M,
        BLOCK_SIZE_QUANT_DIM=TILE_N,
        dst_dtype=OUT_DTYPE,
        use_global_sf=use_global_sf,
        external_global_amax=external_global_amax,
        use_external_global_amax=use_global_sf,
    )

    tl.store(y_block_ptr, tile_quant.to(OUT_DTYPE), boundary_check=(0, 1))


def fp4_fake_quant_weight(
    weight: torch.Tensor,
    global_amax: torch.Tensor = None,
    block_size: int = 16,
    tile_rows: int = 16,
    tile_cols: int = 64,
    use_global_sf: bool = True,
    kernel_impl: str = "legacy",
) -> torch.Tensor:
    """Apply FP4 fake quantization with selectable Triton kernel."""
    if kernel_impl not in {"legacy", "nvfp4"}:
        raise ValueError(f"Unsupported kernel_impl={kernel_impl!r}, expected one of ['legacy', 'nvfp4'].")
    assert weight.is_cuda, "weight must be on CUDA."
    assert weight.dtype in _TORCH_TO_TL_DTYPE, "weight dtype must be fp16/bf16/fp32."

    x_shape = weight.shape
    x_dtype = weight.dtype
    x = weight.reshape(-1, x_shape[-1]).contiguous()
    M, N = x.shape
    y = torch.empty_like(x)

    stride_xm, stride_xn = x.stride()
    stride_ym, stride_yn = y.stride()

    tile_cols = max(tile_cols, block_size)
    tile_cols_aligned = ((tile_cols + block_size - 1) // block_size) * block_size
    num_fp4_blocks = tile_cols_aligned // block_size

    grid = (triton.cdiv(M, tile_rows), triton.cdiv(N, tile_cols_aligned))

    if kernel_impl == "legacy":
        if global_amax is None:
            global_amax = weight.abs().max().to(torch.float32)
        global_scale = global_amax.float() / (FP4_E2M1_MAX * FP8_E4M3_MAX)
        _fp4_fake_quant_kernel_legacy[grid](
            x,
            y,
            M,
            N,
            global_scale,
            stride_xm,
            stride_xn,
            stride_ym,
            stride_yn,
            BLOCK_SIZE=block_size,
            TILE_M=tile_rows,
            TILE_N=tile_cols_aligned,
            NUM_FP4_BLOCKS=num_fp4_blocks,
            OUT_DTYPE=_TORCH_TO_TL_DTYPE[x_dtype],
            FP4_MAX=FP4_E2M1_MAX,
            FP8_MAX=FP8_E4M3_MAX,
        )
        return y.view(*x_shape)

    if block_size != 16:
        raise ValueError("Current NVFP4 fake-quant kernel requires block_size=16.")
    if not use_global_sf:
        raise ValueError("NVFP4 fake-quant kernel requires use_global_sf=True.")
    if global_amax is None:
        global_amax = weight.abs().max().to(torch.float32)
    if global_amax.device != weight.device or global_amax.dtype != torch.float32:
        global_amax_t = global_amax.to(device=weight.device, dtype=torch.float32)
    else:
        global_amax_t = global_amax
    global_amax_t = global_amax_t.reshape(())

    _fp4_fake_quant_kernel_nvfp4[grid](
        x,
        y,
        global_amax_t,
        M,
        N,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        BLOCK_SIZE=block_size,
        TILE_M=tile_rows,
        TILE_N=tile_cols_aligned,
        NUM_FP4_BLOCKS=num_fp4_blocks,
        OUT_DTYPE=_TORCH_TO_TL_DTYPE[x_dtype],
        use_global_sf=use_global_sf,
        num_warps=4,
    )
    return y.view(*x_shape)


class STEFP4QuantTriton(torch.autograd.Function):
    """Straight-Through Estimator wrapper for Triton FP4 quantization kernel."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        global_amax: torch.Tensor,
        block_size: int,
        kernel_impl: str,
    ) -> torch.Tensor:
        return fp4_fake_quant_weight(
            x,
            global_amax=global_amax,
            block_size=block_size,
            kernel_impl=kernel_impl,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        return grad_output, None, None, None


class _TorchAONVFP4RealW4A4Linear(torch.autograd.Function):
    """Real W4A4 path: vLLM NVFP4 forward with dequantized STE backward."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        x_2d: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        input_global_amax: torch.Tensor,
        weight_global_amax: torch.Tensor,
        backend: str,
    ) -> torch.Tensor:
        if not _VLLM_NVFP4_AVAILABLE:
            raise RuntimeError(
                "fake_quant_kernel_impl='torchao_real' requires vLLM NVFP4 ops "
                "(scaled_fp4_quant/cutlass_scaled_fp4_mm). Install vLLM with CUDA ops."
            )

        if input_global_amax.device != x_2d.device or input_global_amax.dtype != torch.float32:
            input_global_amax = input_global_amax.to(device=x_2d.device, dtype=torch.float32)
        if weight_global_amax.device != weight.device or weight_global_amax.dtype != torch.float32:
            weight_global_amax = weight_global_amax.to(device=weight.device, dtype=torch.float32)

        # Keep QAT scale semantics:
        # per_tensor_scale = amax / (FP4_MAX * FP8_MAX).
        input_scale = _per_tensor_amax_to_scale(input_global_amax.reshape(())).to(device=x_2d.device, dtype=torch.float32)
        weight_scale = _per_tensor_amax_to_scale(weight_global_amax.reshape(())).to(device=weight.device, dtype=torch.float32)
        # vLLM scaled_fp4_quant expects "global scale" in inverse form (SFScale),
        # i.e. reciprocal of per_tensor_scale.
        input_sf = (1.0 / (input_scale + 1e-12)).to(device=x_2d.device, dtype=torch.float32)
        weight_sf = (1.0 / (weight_scale + 1e-12)).to(device=weight.device, dtype=torch.float32)

        # Quantize activations/weights into vLLM NVFP4 packed format and block scales.
        x_fp4, x_blockscale = _SIMPLE_QPROF.timed_call(
            "fwd_quant_x_ms",
            scaled_fp4_quant,
            x_2d,
            input_sf,
            is_sf_swizzled_layout=True,
            backend=backend,
        )
        weight_fp4, weight_blockscale = _SIMPLE_QPROF.timed_call(
            "fwd_quant_w_ms",
            scaled_fp4_quant,
            weight,
            weight_sf,
            is_sf_swizzled_layout=True,
            backend=backend,
        )

        # Pair with SFScale (inverse scales) used in quantization path.
        alpha = (1.0 / (input_sf * weight_sf + 1e-12)).to(device=x_2d.device, dtype=torch.float32)
        out_dtype = x_2d.dtype

        ctx.has_bias = bias is not None
        ctx.out_dtype = out_dtype
        # Backward dequantizes the same real NVFP4 tensors used in forward GEMM.
        ctx.save_for_backward(
            x_fp4,
            x_blockscale,
            weight_fp4,
            weight_blockscale,
            input_sf.reshape(()),
            weight_sf.reshape(()),
        )

        if backend.startswith("flashinfer-"):
            if flashinfer_scaled_fp4_mm is None:
                raise RuntimeError("backend requires flashinfer path, but flashinfer_scaled_fp4_mm is unavailable.")
            flashinfer_backend = backend[len("flashinfer-") :]
            out = _SIMPLE_QPROF.timed_call(
                "fwd_gemm_ms",
                flashinfer_scaled_fp4_mm,
                x_fp4,
                weight_fp4,
                x_blockscale,
                weight_blockscale,
                alpha,
                out_dtype,
                backend=flashinfer_backend,
            )
        else:
            out = _SIMPLE_QPROF.timed_call(
                "fwd_gemm_ms",
                cutlass_scaled_fp4_mm,
                x_fp4,
                weight_fp4,
                x_blockscale,
                weight_blockscale,
                alpha,
                out_dtype,
            )
        if bias is not None:
            out = out + bias
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        if dequantize_to_dtype is None:
            raise RuntimeError(
                "fake_quant_kernel_impl='torchao_real' requires vLLM dequantize_to_dtype."
            )

        x_fp4, x_blockscale, weight_fp4, weight_blockscale, input_sf, weight_sf = ctx.saved_tensors
        out_dtype = ctx.out_dtype

        x_dq = _SIMPLE_QPROF.timed_call(
            "bwd_dequant_x_ms",
            dequantize_to_dtype,
            x_fp4,
            x_blockscale,
            input_sf,
            out_dtype,
            x_fp4.device,
        )
        weight_dq = _SIMPLE_QPROF.timed_call(
            "bwd_dequant_w_ms",
            dequantize_to_dtype,
            weight_fp4,
            weight_blockscale,
            weight_sf,
            out_dtype,
            weight_fp4.device,
        )

        if _SIMPLE_QPROF.enabled:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            grad_input = torch.mm(grad_output, weight_dq)
            grad_weight = torch.mm(grad_output.t(), x_dq)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _SIMPLE_QPROF.add("bwd_mm_ms", (time.perf_counter() - t0) * 1000.0)
        else:
            grad_input = torch.mm(grad_output, weight_dq)
            grad_weight = torch.mm(grad_output.t(), x_dq)
        grad_bias = grad_output.sum(dim=0) if (ctx.has_bias and grad_output is not None) else None
        return grad_input, grad_weight, grad_bias, None, None, None


class QATMode(str, Enum):
    """QAT quantization mode."""

    W4A4 = "w4a4"  # Weight 4-bit, Activation 4-bit (dynamic)
    W4A16 = "w4a16"  # Weight 4-bit, Activation 16-bit (weight only)


class QATLinear(nn.Linear):
    """QAT FakeQuantized Linear layer with FSDP compatibility."""

    _UNINITIALIZED_SCALE = -1.0

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode: QATMode = QATMode.W4A4,
        group_size: int = 16,
        activation_observer: str = "static_minmax",  # Observer strategy for activation global_scale
        activation_observer_update_interval: int = 1,
        activation_observer_freeze_after_steps: int = -1,
        activation_observer_sync_interval: int = 1,
        fake_quant_kernel_impl: str = "legacy",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        skip_input_activation_fake_quant: bool = False,
    ):
        super().__init__(in_features, out_features, bias, device=device, dtype=dtype)

        self.mode = mode
        self.group_size = group_size
        self.activation_observer = activation_observer
        if activation_observer_update_interval < 1:
            raise ValueError(
                "activation_observer_update_interval must be >= 1, "
                f"got {activation_observer_update_interval}."
            )
        self.activation_observer_update_interval = activation_observer_update_interval
        if activation_observer_freeze_after_steps < -1:
            raise ValueError(
                "activation_observer_freeze_after_steps must be >= -1, "
                f"got {activation_observer_freeze_after_steps}."
            )
        self.activation_observer_freeze_after_steps = activation_observer_freeze_after_steps
        if activation_observer_sync_interval < 1:
            raise ValueError(
                "activation_observer_sync_interval must be >= 1, "
                f"got {activation_observer_sync_interval}."
            )
        self.activation_observer_sync_interval = activation_observer_sync_interval
        # Observer cadence is driven by optimizer-step index (set externally by
        # actor worker) instead of per-forward count.
        self._activation_observer_train_step = 0
        self._last_activation_observer_update_step = -1
        if fake_quant_kernel_impl not in {"legacy", "nvfp4", "torchao_real"}:
            raise ValueError(
                "Unknown fake_quant_kernel_impl: "
                f"{fake_quant_kernel_impl}. Supported: ['legacy', 'nvfp4', 'torchao_real']"
            )
        if fake_quant_kernel_impl == "torchao_real":
            if mode != QATMode.W4A4:
                raise ValueError("fake_quant_kernel_impl='torchao_real' is only supported for mode='w4a4'.")
            if not _VLLM_NVFP4_AVAILABLE:
                raise RuntimeError(
                    "fake_quant_kernel_impl='torchao_real' requires vLLM NVFP4 ops "
                    "(scaled_fp4_quant/cutlass_scaled_fp4_mm)."
                )
        self.fake_quant_kernel_impl = fake_quant_kernel_impl

        self._cached_weight_amax: Optional[torch.Tensor] = None
        self._fusion_siblings_ref = None

        if mode == QATMode.W4A4:
            self.register_buffer(
                "input_global_scale", torch.tensor([self._UNINITIALIZED_SCALE], dtype=torch.float32), persistent=True
            )

            self.register_buffer(
                "input_amax", torch.tensor([self._UNINITIALIZED_SCALE], dtype=torch.float32), persistent=True
            )
            # Cached activation global_amax = (FP4_E2M1_MAX * FP8_E4M3_MAX) / input_global_scale.
            # Not persisted in checkpoints; lazily recovered from input_global_scale when needed.
            self.register_buffer(
                "cached_input_global_amax",
                torch.tensor([self._UNINITIALIZED_SCALE], dtype=torch.float32),
                persistent=False,
            )

            self._ema_decay: float = 0.01
            # Lazy init flags to avoid hot-path .item() GPU sync every forward.
            # These booleans are not checkpointed; they are lazily recovered from
            # buffer values (input_amax/input_global_scale) on first access after load.
            self._input_amax_initialized: bool = False
            self._input_global_scale_initialized: bool = False
            self._cached_input_global_amax_initialized: bool = False
            self._input_amax_checked: bool = False
            self._input_global_scale_checked: bool = False
            self._cached_input_global_amax_runtime: Optional[torch.Tensor] = None

        self.fake_quant_enabled = True
        # When True and fake quant is disabled, W4A4 forward still updates
        # input scale observers in BF16/FP forward for rollout export.
        self.observe_input_scale_only = False
        # When True (W4A4): input activation is already fake-quantized upstream (e.g.
        # ``FusedRMSNormFakeQuant``); skip per-linear activation fake-quant.
        self.skip_input_activation_fake_quant: bool = skip_input_activation_fake_quant

        self._use_batched_amax_sync: bool = False
        self._pending_local_amax: Optional[torch.Tensor] = None

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        mode: QATMode = QATMode.W4A4,
        group_size: int = 16,
        activation_observer: str = "static_minmax",
        activation_observer_update_interval: int = 1,
        activation_observer_freeze_after_steps: int = -1,
        activation_observer_sync_interval: int = 1,
        fake_quant_kernel_impl: str = "legacy",
        skip_input_activation_fake_quant: bool = False,
    ) -> "QATLinear":
        """Create QATLinear from an existing nn.Linear."""
        has_bias = linear.bias is not None

        new_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=has_bias,
            mode=mode,
            group_size=group_size,
            activation_observer=activation_observer,
            activation_observer_update_interval=activation_observer_update_interval,
            activation_observer_freeze_after_steps=activation_observer_freeze_after_steps,
            activation_observer_sync_interval=activation_observer_sync_interval,
            fake_quant_kernel_impl=fake_quant_kernel_impl,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
            skip_input_activation_fake_quant=skip_input_activation_fake_quant,
        )

        if linear.weight.device != torch.device("meta"):
            new_linear.weight = nn.Parameter(linear.weight.clone())
            if has_bias:
                new_linear.bias = nn.Parameter(linear.bias.clone())

        return new_linear

    def _ensure_w4a4_init_flags(self):
        """Lazily recover W4A4 init flags from persisted buffers once."""
        if self.mode != QATMode.W4A4:
            return
        if not self._input_amax_checked:
            self._input_amax_initialized = self.input_amax.item() != self._UNINITIALIZED_SCALE
            self._input_amax_checked = True
        if not self._input_global_scale_checked:
            self._input_global_scale_initialized = self.input_global_scale.item() != self._UNINITIALIZED_SCALE
            self._input_global_scale_checked = True
            if self._input_global_scale_initialized:
                cached = (FP4_E2M1_MAX * FP8_E4M3_MAX) / self.input_global_scale
                self.cached_input_global_amax.copy_(cached.to(self.cached_input_global_amax.device))
                self._cached_input_global_amax_initialized = True
                self._invalidate_cached_input_global_amax_runtime()
            else:
                self._cached_input_global_amax_initialized = False
                self._invalidate_cached_input_global_amax_runtime()

    def _is_amax_initialized(self) -> bool:
        """Check if input_amax has been initialized."""
        if not hasattr(self, "input_amax"):
            return False
        self._ensure_w4a4_init_flags()
        return self._input_amax_initialized

    def _invalidate_cached_input_global_amax_runtime(self):
        if self.mode == QATMode.W4A4:
            self._cached_input_global_amax_runtime = None

    def sync_observer_from(self, source: "QATLinear") -> None:
        """Copy activation observer state from a shared-input sibling.

        In the shared QKV / GateUp path, only the "leader" projection
        (q_proj / gate_proj) runs the observer update.  Call this on
        siblings (k_proj, v_proj / up_proj) so their ``input_global_scale``
        stays current for weight-sync to vLLM.
        """
        if self.mode != QATMode.W4A4 or source.mode != QATMode.W4A4:
            return
        if source._last_activation_observer_update_step == self._last_activation_observer_update_step:
            return
        self.input_global_scale.copy_(source.input_global_scale)
        self.input_amax.copy_(source.input_amax)
        self.cached_input_global_amax.copy_(source.cached_input_global_amax)
        self._input_amax_initialized = source._input_amax_initialized
        self._input_amax_checked = source._input_amax_checked
        self._input_global_scale_initialized = source._input_global_scale_initialized
        self._input_global_scale_checked = source._input_global_scale_checked
        self._cached_input_global_amax_initialized = source._cached_input_global_amax_initialized
        self._last_activation_observer_update_step = source._last_activation_observer_update_step
        self._invalidate_cached_input_global_amax_runtime()

    def set_activation_observer_train_step(self, step: int):
        """Set current optimizer-step index for observer cadence."""
        if step < 0:
            raise ValueError(f"activation observer train step must be >= 0, got {step}.")
        self._activation_observer_train_step = step

    def _should_update_activation_observer(self) -> bool:
        """Decide whether activation observer should update on this train step."""
        train_step = self._activation_observer_train_step
        if self.activation_observer_freeze_after_steps >= 0:
            if train_step > self.activation_observer_freeze_after_steps:
                return False
        if self._last_activation_observer_update_step == train_step:
            return False
        return (train_step % self.activation_observer_update_interval) == 0

    def _update_input_global_scale(self, x: torch.Tensor):
        """Update static input_global_scale based on observer strategy."""
        assert self.mode == QATMode.W4A4, "_update_input_global_scale should only be called in W4A4 mode"

        train_step = self._activation_observer_train_step
        current_amax = torch.amax(torch.abs(x)).detach().to(torch.float32)

        should_sync = (train_step % self.activation_observer_sync_interval) == 0
        if should_sync and torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            if self._use_batched_amax_sync:
                self._pending_local_amax = current_amax
                self._last_activation_observer_update_step = train_step
                return
            torch.distributed.all_reduce(current_amax, op=torch.distributed.ReduceOp.MAX)

        self._apply_amax_to_observer(current_amax)

    def _apply_amax_to_observer(self, current_amax: torch.Tensor) -> None:
        """Apply (possibly synced) amax to observer state and derive input_global_scale."""
        scale_factor = FP8_E4M3_MAX * FP4_E2M1_MAX

        if self.activation_observer == "memoryless_minmax":
            new_scale = (scale_factor / (current_amax + 1e-12)).view(1)
            self.input_global_scale.copy_(new_scale.to(self.input_global_scale.device))
            self._input_global_scale_initialized = True
            self._input_global_scale_checked = True
            self.cached_input_global_amax.copy_(
                current_amax.view(1).to(self.cached_input_global_amax.device)
            )
            self._cached_input_global_amax_initialized = True
            self._invalidate_cached_input_global_amax_runtime()

        elif self.activation_observer == "static_minmax":
            if not self._is_amax_initialized():
                self.input_amax.copy_(current_amax.view(1).to(self.input_amax.device))
                self._input_amax_initialized = True
                self._input_amax_checked = True
            else:
                new_amax = torch.maximum(self.input_amax, current_amax.view(1).to(self.input_amax.device))
                self.input_amax.copy_(new_amax)
            new_scale = (scale_factor / (self.input_amax + 1e-12)).float().view(1)
            self.input_global_scale.copy_(new_scale.to(self.input_global_scale.device))
            self._input_global_scale_initialized = True
            self._input_global_scale_checked = True
            self.cached_input_global_amax.copy_(self.input_amax)
            self._cached_input_global_amax_initialized = True
            self._invalidate_cached_input_global_amax_runtime()

        elif self.activation_observer == "minmax":
            if not self._is_amax_initialized():
                self.input_amax.copy_(current_amax.view(1).to(self.input_amax.device))
                self._input_amax_initialized = True
                self._input_amax_checked = True
            else:
                new_amax = (1 - self._ema_decay) * self.input_amax + self._ema_decay * current_amax.view(1).to(
                    self.input_amax.device
                )
                self.input_amax.copy_(new_amax)
            new_scale = (scale_factor / (self.input_amax + 1e-12)).float().view(1)
            self.input_global_scale.copy_(new_scale.to(self.input_global_scale.device))
            self._input_global_scale_initialized = True
            self._input_global_scale_checked = True
            self.cached_input_global_amax.copy_(self.input_amax)
            self._cached_input_global_amax_initialized = True
            self._invalidate_cached_input_global_amax_runtime()

        else:
            raise ValueError(f"Unknown activation_observer: {self.activation_observer}")
        self._last_activation_observer_update_step = self._activation_observer_train_step

    def _resolve_weight_global_amax(self, weight: torch.Tensor) -> torch.Tensor:
        """Resolve (and sibling-share) weight global amax for W4A4 behavior parity."""
        with torch.no_grad():
            if self._cached_weight_amax is not None:
                global_amax = self._cached_weight_amax
            else:
                siblings_ref = getattr(self, "_fusion_siblings_ref", None)

                if siblings_ref is not None:
                    siblings = [ref() for ref in siblings_ref if ref() is not None]
                    siblings = [s for s in siblings if s.weight.device != torch.device("meta")]

                    for sibling in siblings:
                        sibling_amax = getattr(sibling, "_cached_weight_amax", None)
                        if sibling_amax is not None:
                            global_amax = sibling_amax
                            self._cached_weight_amax = global_amax
                            break
                    else:
                        all_modules = [self] + siblings
                        amaxes = [m.weight.abs().max().to(torch.float32) for m in all_modules]
                        global_amax = torch.max(torch.stack(amaxes))

                        self._cached_weight_amax = global_amax
                        for sibling in siblings:
                            sibling._cached_weight_amax = global_amax
                else:
                    global_amax = weight.abs().max().to(torch.float32)
                    self._cached_weight_amax = global_amax

            if global_amax.device != weight.device or global_amax.dtype != torch.float32:
                global_amax = global_amax.to(device=weight.device, dtype=torch.float32)
                self._cached_weight_amax = global_amax
        return global_amax

    def _fake_quantize_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Apply fake quantization to weight tensor using Triton kernel."""
        global_amax = self._resolve_weight_global_amax(weight)
        return STEFP4QuantTriton.apply(
            weight,
            global_amax,
            self.group_size,
            self.fake_quant_kernel_impl,
        )

    def _get_runtime_input_global_amax(self, x_device: torch.device) -> torch.Tensor:
        """Return cached input global amax on runtime device as fp32 tensor."""
        self._ensure_w4a4_init_flags()
        if not self._input_global_scale_initialized:
            raise RuntimeError("W4A4 input_global_scale uninitialized. Load PTQ model first.")

        if not self._cached_input_global_amax_initialized:
            cached = (FP4_E2M1_MAX * FP8_E4M3_MAX) / self.input_global_scale
            self.cached_input_global_amax.copy_(cached.to(self.cached_input_global_amax.device))
            self._cached_input_global_amax_initialized = True
            self._invalidate_cached_input_global_amax_runtime()

        global_amax = self.cached_input_global_amax
        if global_amax.device != x_device or global_amax.dtype != torch.float32:
            runtime = self._cached_input_global_amax_runtime
            if runtime is None or runtime.device != x_device or runtime.dtype != torch.float32:
                runtime = global_amax.to(device=x_device, dtype=torch.float32)
                self._cached_input_global_amax_runtime = runtime
            global_amax = runtime
        return global_amax

    def _fake_quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fake quantization to activation tensor (W4A4 mode only)."""
        original_shape = x.shape

        # Avoid `.reshape` (which may copy on non-contiguous input); use `.view`
        # when contiguous, otherwise materialize once explicitly so the cost is
        # visible in profiles rather than hidden inside reshape.
        if x.dim() == 3:
            if x.is_contiguous():
                x_2d = x.view(-1, x.shape[-1])
            else:
                x_2d = x.contiguous().view(-1, x.shape[-1])
        else:
            x_2d = x if x.is_contiguous() else x.contiguous()

        if self.training:
            # Observer updates must not build autograd nodes: gradient checkpointing
            # re-runs forward in the same train step, and per-step dedup would skip
            # observer ops on recompute while the first pass still saved them.
            with torch.no_grad():
                if self._should_update_activation_observer():
                    self._update_input_global_scale(x_2d)

        global_amax = self._get_runtime_input_global_amax(x.device)

        result = STEFP4QuantTriton.apply(
            x_2d,
            global_amax,
            self.group_size,
            self.fake_quant_kernel_impl,
        )
        return result.view(original_shape)

    def _forward_real_w4a4_torchao(self, x: torch.Tensor) -> torch.Tensor:
        """torchao-style real W4A4 path without fused multi-op forward."""
        original_shape = x.shape
        if x.dim() == 3:
            x_2d = x.view(-1, x.shape[-1]) if x.is_contiguous() else x.contiguous().view(-1, x.shape[-1])
        else:
            x_2d = x if x.is_contiguous() else x.contiguous()

        if self.training:
            with torch.no_grad():
                if self._should_update_activation_observer():
                    self._update_input_global_scale(x_2d)

        input_global_amax = self._get_runtime_input_global_amax(x_2d.device)
        weight_global_amax = self._resolve_weight_global_amax(self.weight)

        out_2d = _TorchAONVFP4RealW4A4Linear.apply(
            x_2d,
            self.weight,
            self.bias,
            input_global_amax,
            weight_global_amax,
            _resolve_vllm_nvfp4_backend(),
        )
        return out_2d.view(original_shape[:-1] + (self.out_features,)) if x.dim() == 3 else out_2d

    def quantize_activation_once(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize activation once for shared-input projection groups.

        This helper is intentionally explicit so callers (e.g. q/k/v or gate/up
        branches) can quantize a common input tensor once and reuse it across
        multiple projections in the same forward pass.

        When ``skip_input_activation_fake_quant`` is True (e.g. W4A4 fused RMSNorm
        activation FQ upstream), the tensor is already activation-fake-quantized;
        return it unchanged so behavior matches :meth:`forward` and we avoid a
        second STE / mismatched scale semantics on the same activations.
        """
        if not self.fake_quant_enabled or self.mode != QATMode.W4A4:
            return x
        if self.skip_input_activation_fake_quant:
            return x
        if self.fake_quant_kernel_impl == "torchao_real":
            # For real W4A4 path, keep shared-input API compatibility while
            # letting each projection run torchao-style real quantized linear.
            return x
        return self._fake_quantize_activation(x)

    def forward_with_prequantized_input(self, x_fq: torch.Tensor) -> torch.Tensor:
        """Forward path that reuses an already fake-quantized activation."""
        if not self.fake_quant_enabled:
            return F.linear(x_fq, self.weight, self.bias)
        if self.mode == QATMode.W4A4 and self.fake_quant_kernel_impl == "torchao_real":
            return self._forward_real_w4a4_torchao(x_fq)
        weight_fq = self._fake_quantize_weight(self.weight)
        return F.linear(x_fq, weight_fq, self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with fake quantization."""
        if not self.fake_quant_enabled:
            if self.mode == QATMode.W4A4 and self.training and self.observe_input_scale_only:
                # Keep observer statistics fresh in BF16/FP forward path.
                with torch.no_grad():
                    if self._should_update_activation_observer():
                        x_2d = x.view(-1, x.shape[-1]) if x.dim() > 2 else x
                        if not x_2d.is_contiguous():
                            x_2d = x_2d.contiguous()
                        self._update_input_global_scale(x_2d)
            return F.linear(x, self.weight, self.bias)
        if self.mode == QATMode.W4A4 and self.fake_quant_kernel_impl == "torchao_real":
            return self._forward_real_w4a4_torchao(x)

        weight_fq = self._fake_quantize_weight(self.weight)

        if self.mode == QATMode.W4A4:
            if self.skip_input_activation_fake_quant:
                x_fq = x
            else:
                x_fq = self._fake_quantize_activation(x)
        else:
            x_fq = x

        return F.linear(x_fq, weight_fq, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, mode={self.mode.value}, "
            f"group_size={self.group_size}, "
            f"activation_observer_update_interval={self.activation_observer_update_interval}, "
            f"activation_observer_freeze_after_steps={self.activation_observer_freeze_after_steps}, "
            f"activation_observer_sync_interval={self.activation_observer_sync_interval}, "
            f"fake_quant_kernel_impl={self.fake_quant_kernel_impl}, "
            f"fake_quant_enabled={self.fake_quant_enabled}, "
            f"observe_input_scale_only={self.observe_input_scale_only}, "
            f"skip_input_activation_fake_quant={getattr(self, 'skip_input_activation_fake_quant', False)}"
        )
