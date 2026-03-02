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

"""QAT (Quantization-Aware Training) utilities for verl FSDP training."""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.distributed as dist

from verl.base_config import BaseConfig

logger = logging.getLogger(__name__)


@dataclass
class QATConfig(BaseConfig):
    """Unified configuration for QAT (Quantization-Aware Training)."""

    enable: bool = False
    mode: str = "w4a16"
    group_size: int = 16
    ignore_patterns: list[str] = field(default_factory=lambda: ["lm_head", "embed_tokens", "re:.*mlp.gate$"])
    activation_observer: str = "static_minmax"
    quantization_config_path: Optional[str] = None
    # Update-Aware Quantization (UAQ) from QuRL paper (QURL.2602.13953)
    # Invariant scaling s > 1 reduces quantization error and amplifies weight updates (s^2 improvement).
    # Default 1.0 disables UAQ; use 1.5 for INT8/FP8 per paper ablation.
    uaq_scale: float = 1.0


def _is_global_rank_zero() -> bool:
    """Best-effort check for global rank 0; fall back to True if unavailable."""
    try:
        if not dist.is_available() or not dist.is_initialized():
            return True
        return dist.get_rank() == 0
    except Exception:
        # In case distributed is misconfigured, avoid crashing on logging.
        return True


def load_quantization_config(qat_config: QATConfig) -> dict[str, Any]:
    """Load quantization config JSON file from QATConfig."""
    if not qat_config.quantization_config_path:
        raise ValueError("quantization_config_path is required when QAT is enabled")

    logger.info(f"Loading QAT quantization config from: {qat_config.quantization_config_path}")

    with open(qat_config.quantization_config_path) as f:
        quant_config = json.load(f)

    if qat_config.ignore_patterns:
        original_ignore = quant_config.get("ignore", [])
        quant_config["ignore"] = qat_config.ignore_patterns
        if original_ignore != qat_config.ignore_patterns:
            logger.info(f"Overriding JSON 'ignore' field: {original_ignore} -> {qat_config.ignore_patterns}")

    logger.info("Successfully loaded QAT quantization config")
    return quant_config


def apply_update_aware_quantization(
    model: nn.Module,
    scale: float = 1.5,
) -> nn.Module:
    """Apply Update-Aware Quantization (UAQ) via invariant scaling.

    From QURL paper (QURL.2602.13953): Uses invariant scaling WX = (W/s)·(sX) to simultaneously
    reduce quantization error and amplify weight updates. The scale s is applied column-wise to
    linear weights and absorbed into the preceding LayerNorm, preserving output.

    This is a one-time weight adjustment before QAT/RL training. Empirical s=1.5 works well for
    INT8/FP8 per paper ablation.

    Args:
        model: The model to apply UAQ to (e.g. LlamaForCausalLM, Qwen2ForCausalLM).
        scale: Scaling factor s > 1. Use 1.0 to disable (no-op).

    Returns:
        The same model with weights modified in-place.
    """
    if scale <= 1.0:
        # UAQ disabled via scale; keep this quiet to avoid log spam.
        return model

    # Resolve inner model (PEFT wraps in base_model); name_to_module uses names relative to inner
    inner = getattr(model, "base_model", model)
    if hasattr(inner, "model"):
        inner = inner.model

    # Build layer name -> module map (names are relative to inner, e.g. "layers.0.input_layernorm")
    name_to_module: dict[str, nn.Module] = {}
    for name, module in inner.named_modules():
        name_to_module[name] = module

    def get_layer_idx(name: str) -> Optional[int]:
        m = re.search(r"layers\.(\d+)", name)
        return int(m.group(1)) if m else None

    layers_indices: set[int] = set()
    for name in name_to_module:
        idx = get_layer_idx(name)
        if idx is not None:
            layers_indices.add(idx)

    if not layers_indices:
        logger.warning("[UAQ] No decoder layers found, skipping Update-Aware Quantization")
        return model

    uaq_count = 0
    for layer_idx in sorted(layers_indices):
        base = f"layers.{layer_idx}."

        def find_module(suffix: str) -> Optional[nn.Module]:
            key = base + suffix
            return name_to_module.get(key)

        def find_param(module: nn.Module, attr: str = "weight") -> Optional[torch.Tensor]:
            t = getattr(module, attr, None)
            if t is not None and isinstance(t, torch.Tensor):
                return t
            return None

        # QKV group: input_layernorm -> q_proj, k_proj, v_proj
        input_ln = find_module("input_layernorm")
        q_proj = find_module("self_attn.q_proj")
        k_proj = find_module("self_attn.k_proj")
        v_proj = find_module("self_attn.v_proj")

        if input_ln is not None and q_proj is not None and k_proj is not None and v_proj is not None:
            ln_w = find_param(input_ln)
            q_w = find_param(q_proj)
            k_w = find_param(k_proj)
            v_w = find_param(v_proj)
            if ln_w is not None and q_w is not None and k_w is not None and v_w is not None:
                with torch.no_grad():
                    q_w.div_(scale)
                    k_w.div_(scale)
                    v_w.div_(scale)
                    ln_w.mul_(scale)
                uaq_count += 1

        # GateUp group: post_attention_layernorm -> gate_proj, up_proj
        post_ln = find_module("post_attention_layernorm")
        gate_proj = find_module("mlp.gate_proj")
        up_proj = find_module("mlp.up_proj")

        if post_ln is not None and gate_proj is not None and up_proj is not None:
            ln_w = find_param(post_ln)
            g_w = find_param(gate_proj)
            u_w = find_param(up_proj)
            if ln_w is not None and g_w is not None and u_w is not None:
                with torch.no_grad():
                    g_w.div_(scale)
                    u_w.div_(scale)
                    ln_w.mul_(scale)
                uaq_count += 1

    if uaq_count > 0:
        msg = f"[UAQ] Applied Update-Aware Quantization with scale={scale} to {uaq_count} layer groups"
        # Use WARNING level so it survives VERL_LOGGING_LEVEL=WARNING, and also print with banners.
        logger.warning(msg)
        if _is_global_rank_zero():
            banner = "=" * 80
            print(banner)
            print(f"[UAQ][APPLIED] {msg}")
            print(banner)
    return model


def _should_quantize(name: str, module: nn.Module, config: QATConfig) -> bool:
    """Check if a module should be quantized."""
    if not isinstance(module, nn.Linear):
        return False

    for pattern in config.ignore_patterns:
        if pattern.startswith("re:"):
            regex = pattern[3:]
            if re.match(regex, name):
                logger.debug(f"Ignoring {name} due to regex pattern: {regex}")
                return False
        else:
            if pattern in name:
                logger.debug(f"Ignoring {name} due to pattern: {pattern}")
                return False

    if module.in_features % config.group_size != 0:
        logger.warning(
            f"Skipping {name}: in_features={module.in_features} not divisible by group_size={config.group_size}"
        )
        return False

    return True


def apply_qat(
    model: nn.Module,
    config: QATConfig | dict[str, Any],
) -> nn.Module:
    """Apply QAT to a model by replacing nn.Linear with QATLinear."""
    from verl.utils.qat.linear import QATLinear, QATMode

    if not isinstance(config, QATConfig):
        config = QATConfig(**config)

    if not config.enable:
        logger.info("QAT is disabled, returning original model")
        return model

    # Update-Aware Quantization: one-time invariant scaling before QAT (QuRL paper).
    # For safety and backwards-compatibility we *only* apply UAQ in W4A4 mode.
    # W4A16 baselines stay bit‑for‑bit identical to upstream even if uaq_scale>1 is
    # accidentally set in configs.
    if config.mode == "w4a4" and config.uaq_scale > 1.0:
        logger.warning(
            f"[QAT][UAQ] Enabling Update-Aware Quantization: mode={config.mode}, "
            f"group_size={config.group_size}, uaq_scale={config.uaq_scale}"
        )
        if _is_global_rank_zero():
            print(
                f"[QAT][UAQ] >>> UAQ ENABLED: mode={config.mode}, "
                f"group_size={config.group_size}, scale={config.uaq_scale}"
            )
        apply_update_aware_quantization(model, scale=config.uaq_scale)
    elif config.mode != "w4a4" and config.uaq_scale > 1.0:
        # Log once per rank when UAQ is requested for non‑W4A4 modes but skipped.
        logger.warning(
            f"[QAT][UAQ] uaq_scale={config.uaq_scale} requested for mode={config.mode}, "
            "but UAQ is currently only enabled for mode='w4a4'. Skipping UAQ to keep "
            "non‑w4a4 baselines (e.g. w4a16) numerically consistent with upstream."
        )

    mode = QATMode(config.mode.lower())
    logger.info(f"Applying QAT with mode={mode.value}, group_size={config.group_size}")

    modules_to_replace = []
    for name, module in model.named_modules():
        if _should_quantize(name, module, config):
            modules_to_replace.append((name, module))

    logger.info(f"Found {len(modules_to_replace)} Linear layers to convert to QAT")

    converted_count = 0
    for name, module in modules_to_replace:
        if isinstance(module, QATLinear):
            continue

        fake_quant_module = QATLinear.from_linear(
            module,
            mode=mode,
            group_size=config.group_size,
            activation_observer=config.activation_observer,
        )

        _set_module(model, name, fake_quant_module)
        converted_count += 1

    logger.info(f"Successfully applied QAT to {converted_count} layers")

    return model


def _set_module(model: nn.Module, name: str, new_module: nn.Module):
    """Set a module in the model by its full name."""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


FUSION_PATTERNS = {
    "qkv": ["q_proj", "k_proj", "v_proj"],
    "gate_up": ["gate_proj", "up_proj"],
}


def setup_fusion_siblings(model: nn.Module):
    """Setup fusion siblings for QKV and GateUp layers."""
    import weakref

    from verl.utils.qat.linear import QATLinear

    qat_modules = {name: m for name, m in model.named_modules() if isinstance(m, QATLinear)}

    counts = {}
    for group_name, suffixes in FUSION_PATTERNS.items():
        groups: dict[str, dict[str, nn.Module]] = {}
        for name, module in qat_modules.items():
            for suffix in suffixes:
                if name.endswith(suffix):
                    parent = name.rsplit(".", 1)[0]
                    groups.setdefault(parent, {})[suffix] = module

        count = 0
        for parent, projs in groups.items():
            if len(projs) >= 2:
                modules = list(projs.values())
                for i, m in enumerate(modules):
                    siblings = modules[:i] + modules[i + 1 :]
                    m._fusion_siblings_ref = [weakref.ref(s) for s in siblings]
                count += 1
        counts[group_name] = count

    logger.info(f"[QAT Fuse] Setup fusion siblings: {counts}")
    return counts


def enable_qat_fuse(model: nn.Module):
    """Enable QAT fuse mode: sets up fusion siblings for weight scale fusion."""
    setup_fusion_siblings(model)
    model._qat_fuse_enabled = True
    logger.info("[QAT Fuse] Enabled QAT fuse mode")


def invalidate_all_scales(model: nn.Module):
    """Clear all cached weight scales after optimizer.step()."""
    from verl.utils.qat.linear import QATLinear

    count = 0
    for module in model.modules():
        if isinstance(module, QATLinear):
            module._weight_blockwise_scale = None
            module._weight_global_scale = None
            module._cached_weight_amax = None
            count += 1

    logger.debug(f"[QAT Fuse] Invalidated scales for {count} QATLinear layers")
