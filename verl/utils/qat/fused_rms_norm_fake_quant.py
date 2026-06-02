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

"""Fused RMSNorm + W4A4 activation fake-quant for dense decoder blocks (e.g. Qwen3).

When enabled, ``input_layernorm`` / ``post_attention_layernorm`` outputs are
fake-quantized once; ``q_proj``/``k_proj``/``v_proj`` and ``gate_proj``/``up_proj``
should set ``skip_input_activation_fake_quant=True`` on their ``QATLinear`` modules.

Observer / Triton activation path is kept consistent with ``QATLinear`` W4A4 logic
in ``verl/utils/qat/linear.py`` — update both if you change quantization semantics.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from verl.utils.qat.linear import FP4_E2M1_MAX, FP8_E4M3_MAX, STEFP4QuantTriton

if TYPE_CHECKING:
    from verl.utils.qat.core import QATConfig

logger = logging.getLogger(__name__)


def _is_hf_rmsnorm(module: nn.Module) -> bool:
    if not isinstance(module, nn.Module):
        return False
    if not hasattr(module, "weight") or not isinstance(module.weight, nn.Parameter):
        return False
    if module.weight.dim() != 1:
        return False
    return hasattr(module, "variance_epsilon") or hasattr(module, "eps")


class FusedRMSNormFakeQuant(nn.Module):
    """RMSNorm followed by W4A4 activation fake-quant (same semantics as ``QATLinear``)."""

    _UNINITIALIZED_SCALE = -1.0

    def __init__(
        self,
        hidden_size: int,
        variance_epsilon: float,
        *,
        group_size: int,
        activation_observer: str,
        activation_observer_update_interval: int,
        activation_observer_freeze_after_steps: int,
        activation_observer_sync_interval: int,
        fake_quant_kernel_impl: str,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.variance_epsilon = float(variance_epsilon)
        self.weight = nn.Parameter(torch.ones(hidden_size))

        self.group_size = group_size
        self.activation_observer = activation_observer
        self.activation_observer_update_interval = activation_observer_update_interval
        self.activation_observer_freeze_after_steps = activation_observer_freeze_after_steps
        self.activation_observer_sync_interval = activation_observer_sync_interval
        self.fake_quant_kernel_impl = fake_quant_kernel_impl

        self._activation_observer_train_step = 0
        self._last_activation_observer_update_step = -1
        self._ema_decay: float = 0.01
        self._use_batched_amax_sync: bool = False
        self._pending_local_amax: Optional[torch.Tensor] = None
        self.fake_quant_enabled: bool = True
        self.observe_input_scale_only: bool = False

        self.register_buffer(
            "input_global_scale", torch.tensor([self._UNINITIALIZED_SCALE], dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "input_amax", torch.tensor([self._UNINITIALIZED_SCALE], dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "cached_input_global_amax",
            torch.tensor([self._UNINITIALIZED_SCALE], dtype=torch.float32),
            persistent=False,
        )
        self._input_amax_initialized: bool = False
        self._input_global_scale_initialized: bool = False
        self._cached_input_global_amax_initialized: bool = False
        self._input_amax_checked: bool = False
        self._input_global_scale_checked: bool = False
        self._cached_input_global_amax_runtime: Optional[torch.Tensor] = None

    @classmethod
    def from_rmsnorm(cls, rms: nn.Module, *, config: "QATConfig") -> "FusedRMSNormFakeQuant":
        if not _is_hf_rmsnorm(rms):
            raise TypeError(f"Expected an HF-style RMSNorm module, got {type(rms)}")
        eps = float(getattr(rms, "variance_epsilon", getattr(rms, "eps", 1e-6)))
        hidden = int(rms.weight.shape[0])
        m = cls(
            hidden,
            eps,
            group_size=config.group_size,
            activation_observer=config.activation_observer,
            activation_observer_update_interval=config.activation_observer_update_interval,
            activation_observer_freeze_after_steps=config.activation_observer_freeze_after_steps,
            activation_observer_sync_interval=config.activation_observer_sync_interval,
            fake_quant_kernel_impl=config.fake_quant_kernel_impl,
        )
        # Match QATLinear.from_linear: FSDP/meta init has no materialized weights yet.
        if rms.weight.device != torch.device("meta"):
            with torch.no_grad():
                m.weight = nn.Parameter(rms.weight.clone())
        return m.to(device=rms.weight.device, dtype=rms.weight.dtype)

    def set_activation_observer_train_step(self, step: int) -> None:
        if step < 0:
            raise ValueError(f"activation observer train step must be >= 0, got {step}.")
        self._activation_observer_train_step = step

    def _invalidate_cached_input_global_amax_runtime(self) -> None:
        self._cached_input_global_amax_runtime = None

    def _ensure_w4a4_init_flags(self) -> None:
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
        self._ensure_w4a4_init_flags()
        return self._input_amax_initialized

    def _should_update_activation_observer(self) -> bool:
        train_step = self._activation_observer_train_step
        if self.activation_observer_freeze_after_steps >= 0:
            if train_step > self.activation_observer_freeze_after_steps:
                return False
        if self._last_activation_observer_update_step == train_step:
            return False
        return (train_step % self.activation_observer_update_interval) == 0

    def _update_input_global_scale(self, x: torch.Tensor) -> None:
        train_step = self._activation_observer_train_step
        current_amax = torch.amax(torch.abs(x)).detach().to(torch.float32)

        should_sync = (train_step % self.activation_observer_sync_interval) == 0
        if should_sync and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            if self._use_batched_amax_sync:
                self._pending_local_amax = current_amax
                self._last_activation_observer_update_step = train_step
                return
            dist.all_reduce(current_amax, op=dist.ReduceOp.MAX)

        self._apply_amax_to_observer(current_amax)

    def _apply_amax_to_observer(self, current_amax: torch.Tensor) -> None:
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

    def _rms_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Match HF Qwen/Llama-style RMSNorm (compute variance in fp32)."""
        input_dtype = hidden_states.dtype
        h = hidden_states.to(torch.float32)
        variance = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(variance + self.variance_epsilon)
        h = h.to(input_dtype) * self.weight
        return h

    def _fake_quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        if x.dim() == 3:
            if x.is_contiguous():
                x_2d = x.view(-1, x.shape[-1])
            else:
                x_2d = x.contiguous().view(-1, x.shape[-1])
        else:
            x_2d = x if x.is_contiguous() else x.contiguous()

        if self.training:
            with torch.no_grad():
                if self._should_update_activation_observer():
                    self._update_input_global_scale(x_2d)

        self._ensure_w4a4_init_flags()
        if not self._input_global_scale_initialized:
            raise RuntimeError("W4A4 input_global_scale uninitialized. Load PTQ scales or run calibration.")

        if not self._cached_input_global_amax_initialized:
            cached = (FP4_E2M1_MAX * FP8_E4M3_MAX) / self.input_global_scale
            self.cached_input_global_amax.copy_(cached.to(self.cached_input_global_amax.device))
            self._cached_input_global_amax_initialized = True
            self._invalidate_cached_input_global_amax_runtime()

        global_amax = self.cached_input_global_amax
        if global_amax.device != x.device or global_amax.dtype != torch.float32:
            runtime = self._cached_input_global_amax_runtime
            if runtime is None or runtime.device != x.device or runtime.dtype != torch.float32:
                runtime = global_amax.to(device=x.device, dtype=torch.float32)
                self._cached_input_global_amax_runtime = runtime
            global_amax = runtime

        kernel_impl = self.fake_quant_kernel_impl
        if kernel_impl == "torchao_real":
            kernel_impl = "nvfp4"
        result = STEFP4QuantTriton.apply(
            x_2d,
            global_amax,
            self.group_size,
            kernel_impl,
        )
        return result.view(original_shape)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed = self._rms_forward(hidden_states)
        if not self.fake_quant_enabled:
            # Mirror QATLinear's observe-only path: keep observer statistics
            # fresh without applying fake quantization.  Critical for
            # _calibrate_w4a4_input_scales() when loading a pure BF16 model
            # (input_global_scale not yet initialized).
            if self.training and self.observe_input_scale_only:
                with torch.no_grad():
                    if self._should_update_activation_observer():
                        x_2d = normed.view(-1, normed.shape[-1]) if normed.dim() > 2 else normed
                        if not x_2d.is_contiguous():
                            x_2d = x_2d.contiguous()
                        self._update_input_global_scale(x_2d)
            return normed
        return self._fake_quantize_activation(normed)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, eps={self.variance_epsilon}, "
            f"group_size={self.group_size}, observer={self.activation_observer}, "
            f"kernel={self.fake_quant_kernel_impl}"
        )


def apply_w4a4_fused_rms_norm_fake_quant(model: nn.Module, config: "QATConfig") -> tuple[int, int]:
    """Replace decoder RMSNorms and mark q/k/v/gate/up QATLinear to skip activation FQ.

    Returns:
        (num_norms_replaced, num_linears_marked_skipped)
    """
    from verl.utils.qat.core import _is_global_rank_zero, _set_module
    from verl.utils.qat.linear import QATLinear, QATMode

    if config.mode != "w4a4":
        raise ValueError("fuse_w4a4_rms_norm_activation is only supported for mode='w4a4'.")

    norm_suffixes = (".input_layernorm", ".post_attention_layernorm")
    replaced = 0
    for name, module in list(model.named_modules()):
        if not any(name.endswith(s) for s in norm_suffixes):
            continue
        if ".layers." not in name:
            continue
        if isinstance(module, FusedRMSNormFakeQuant):
            continue
        if not _is_hf_rmsnorm(module):
            continue

        fused = FusedRMSNormFakeQuant.from_rmsnorm(module, config=config)
        _set_module(model, name, fused)
        replaced += 1
        if _is_global_rank_zero():
            logger.info("[QAT][FusedRMSNormFakeQuant] Replaced %s", name)

    skip_tail = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
    marked = 0
    for name, module in model.named_modules():
        if not isinstance(module, QATLinear) or module.mode != QATMode.W4A4:
            continue
        short = name.rsplit(".", 1)[-1]
        if short in skip_tail:
            module.skip_input_activation_fake_quant = True
            marked += 1

    if _is_global_rank_zero():
        logger.info(
            "[QAT][FusedRMSNormFakeQuant] norms_replaced=%d qat_linears_skip_act=%d",
            replaced,
            marked,
        )
    return replaced, marked


def sync_fused_norm_buffers_from_projections(model: nn.Module) -> None:
    """After loading ``*.q_proj`` / ``*.gate_proj`` W4A4 buffers, copy scales into fused norms.

    PTQ / HF checkpoints typically store activation scales on projection modules; fused norms own
    the live observer buffers used at runtime.
    """
    from verl.utils.qat.linear import QATLinear, QATMode

    for name, module in model.named_modules():
        if not isinstance(module, FusedRMSNormFakeQuant):
            continue
        if name.endswith(".input_layernorm"):
            base = name[: -len(".input_layernorm")]
            proj_name = f"{base}.self_attn.q_proj"
        elif name.endswith(".post_attention_layernorm"):
            base = name[: -len(".post_attention_layernorm")]
            proj_name = f"{base}.mlp.gate_proj"
        else:
            continue
        try:
            proj = model.get_submodule(proj_name)
        except AttributeError:
            continue
        if not isinstance(proj, QATLinear) or proj.mode != QATMode.W4A4:
            continue
        module.input_global_scale.copy_(proj.input_global_scale)
        module.input_amax.copy_(proj.input_amax)
        scale_factor = FP8_E4M3_MAX * FP4_E2M1_MAX
        cached = (scale_factor / (module.input_global_scale + 1e-12)).view(1)
        module.cached_input_global_amax.copy_(cached.to(module.cached_input_global_amax.device))
        module._input_global_scale_initialized = proj.input_global_scale.item() != FusedRMSNormFakeQuant._UNINITIALIZED_SCALE
        module._input_global_scale_checked = True
        module._input_amax_initialized = proj.input_amax.item() != FusedRMSNormFakeQuant._UNINITIALIZED_SCALE
        module._input_amax_checked = True
        module._cached_input_global_amax_initialized = module._input_global_scale_initialized
        module._cached_input_global_amax_runtime = None
        module._last_activation_observer_update_step = getattr(
            proj, "_last_activation_observer_update_step", -1
        )


def sync_projection_buffers_from_fused_norms(model: nn.Module) -> None:
    """Copy fused RMSNorm W4A4 activation buffers onto attention / MLP projections for export.

    With ``fuse_w4a4_rms_norm_activation``, observers update on ``FusedRMSNormFakeQuant`` while
    ``q_proj``/``k_proj``/``v_proj`` and ``gate_proj``/``up_proj`` use
    ``skip_input_activation_fake_quant``; vLLM and ``scales.safetensors`` still expect per-linear
    ``input_global_scale`` on each ``QATLinear``. Call on all ranks immediately before
    ``get_fsdp_full_state_dict`` when saving HF checkpoints.
    """
    from verl.utils.qat.linear import FP4_E2M1_MAX, FP8_E4M3_MAX, QATLinear, QATMode

    scale_factor = FP8_E4M3_MAX * FP4_E2M1_MAX
    uninitialized = FusedRMSNormFakeQuant._UNINITIALIZED_SCALE

    for name, module in model.named_modules():
        if not isinstance(module, FusedRMSNormFakeQuant):
            continue
        if name.endswith(".input_layernorm"):
            base = name[: -len(".input_layernorm")]
            proj_suffixes = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")
        elif name.endswith(".post_attention_layernorm"):
            base = name[: -len(".post_attention_layernorm")]
            proj_suffixes = ("mlp.gate_proj", "mlp.up_proj")
        else:
            continue
        last_step = getattr(module, "_last_activation_observer_update_step", -1)
        for suffix in proj_suffixes:
            proj_name = f"{base}.{suffix}"
            try:
                proj = model.get_submodule(proj_name)
            except AttributeError:
                continue
            if not isinstance(proj, QATLinear) or proj.mode != QATMode.W4A4:
                continue
            proj.input_global_scale.copy_(module.input_global_scale)
            proj.input_amax.copy_(module.input_amax)
            cached = (scale_factor / (proj.input_global_scale + 1e-12)).view(1)
            proj.cached_input_global_amax.copy_(cached.to(proj.cached_input_global_amax.device))
            proj._input_global_scale_initialized = module.input_global_scale.item() != uninitialized
            proj._input_global_scale_checked = True
            proj._input_amax_initialized = module.input_amax.item() != uninitialized
            proj._input_amax_checked = getattr(module, "_input_amax_checked", True)
            proj._cached_input_global_amax_initialized = proj._input_global_scale_initialized
            proj._invalidate_cached_input_global_amax_runtime()
            proj._last_activation_observer_update_step = last_step
