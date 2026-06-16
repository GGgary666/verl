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

"""W4A4 activation calibration for QAT (input_global_scale / input_amax)."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Optional

import torch
import torch.nn as nn
from compressed_tensors.quantization.quant_args import FP4_E2M1_DATA, FP8_E4M3_DATA

from verl.utils.device import get_device_name, get_torch_device

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

UNINITIALIZED_SCALE = -1.0


def is_scale_uninitialized(tensor: torch.Tensor) -> bool:
    if tensor.numel() == 0:
        return True
    return tensor.numel() == 1 and float(tensor.reshape(-1)[0].item()) == UNINITIALIZED_SCALE


def input_global_scale_from_amax(amax: torch.Tensor) -> torch.Tensor:
    amax_val = amax.to(torch.float32).reshape(-1).max()
    return (FP8_E4M3_DATA.max * FP4_E2M1_DATA.max / (amax_val + 1e-12)).float().reshape(1)


def count_uninitialized_w4a4_modules(model: nn.Module) -> tuple[int, int]:
    """Return (total_w4a4_layers, uninitialized_layers)."""
    from verl.utils.qat.linear import QATLinear, QATMode

    total = 0
    uninitialized = 0
    for module in model.modules():
        if not isinstance(module, QATLinear) or module.mode != QATMode.W4A4:
            continue
        total += 1
        if is_scale_uninitialized(module.input_global_scale) or is_scale_uninitialized(module.input_amax):
            uninitialized += 1
    return total, uninitialized


def count_uninitialized_moe_layers(model: nn.Module) -> tuple[int, int]:
    """Return (total_moe_layers, uninitialized_layers)."""
    from verl.utils.qat.moe import count_uninitialized_moe_layers as _count_moe

    return _count_moe(model)


def needs_moe_calibration(model: nn.Module) -> bool:
    from verl.utils.qat.moe import needs_moe_calibration as _needs_moe

    return _needs_moe(model)


def needs_w4a4_calibration(model: nn.Module) -> bool:
    linear_total, linear_uninit = count_uninitialized_w4a4_modules(model)
    moe_total, moe_uninit = count_uninitialized_moe_layers(model)
    return (linear_total > 0 and linear_uninit > 0) or (moe_total > 0 and moe_uninit > 0)


def _normalize_device(device: Optional[torch.device | int | str]) -> torch.device:
    if device is None:
        return torch.device(get_device_name())
    return torch.device(device)


def _get_model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _get_model_weight_dtype(model: nn.Module) -> torch.dtype:
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float32


def _is_fsdp_wrapped(model: nn.Module) -> bool:
    from verl.utils.fsdp_utils import fsdp_version

    return fsdp_version(model) > 0


@contextmanager
def _calibration_forward_context(model: nn.Module, device: torch.device) -> Iterator[None]:
    """Use sdpa + bf16 autocast for post-FSDP calibration forwards."""
    config = getattr(model, "config", None)
    original_attn = getattr(config, "_attn_implementation", None) if config is not None else None
    weight_dtype = _get_model_weight_dtype(model)

    if config is not None and original_attn in ("flash_attention_2", "flash_attention_3"):
        config._attn_implementation = "sdpa"
        logger.info(
            "[QAT W4A4 Calib] Switched attention %s -> sdpa (model_dtype=%s, flash-attn needs bf16/fp16)",
            original_attn,
            weight_dtype,
        )

    use_autocast = device.type == "cuda"
    try:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_autocast):
            yield
    finally:
        if config is not None and original_attn is not None:
            config._attn_implementation = original_attn


def _collate_tokenized(samples: list[dict[str, torch.Tensor]], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(sample["input_ids"].shape[1] for sample in samples)
    input_ids = []
    attention_mask = []
    for sample in samples:
        ids = sample["input_ids"][0]
        pad_len = max_len - ids.shape[0]
        if pad_len > 0:
            ids = torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=ids.dtype)])
            mask = torch.cat([torch.ones(ids.shape[0] - pad_len, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
        else:
            mask = torch.ones(ids.shape[0], dtype=torch.long)
        input_ids.append(ids)
        attention_mask.append(mask)
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
    }


def _load_calib_prompts(data_files: str | list[str], num_samples: int, prompt_key: str = "prompt") -> list[Any]:
    import datasets

    if isinstance(data_files, str):
        data_files = [data_files]
    dataset = datasets.load_dataset("parquet", data_files=data_files, split="train")
    if num_samples > 0 and len(dataset) > num_samples:
        dataset = dataset.shuffle(seed=42).select(range(num_samples))

    prompts = []
    for row in dataset:
        prompt = row.get(prompt_key)
        if prompt is not None:
            prompts.append(prompt)
    return prompts


def _tokenize_prompt(tokenizer, prompt: Any, max_seq_len: int) -> dict[str, torch.Tensor]:
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = prompt
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_len)


def _ensure_fsdp_model_on_device(model: nn.Module, device: torch.device) -> tuple[torch.device, bool]:
    """Load an FSDP-wrapped model onto ``device`` for calibration if needed."""
    if not _is_fsdp_wrapped(model):
        raise RuntimeError(
            "[QAT W4A4 Calib] Activation calibration must run on an FSDP-wrapped model after wrap; "
            "got a non-FSDP module."
        )

    model_device = _get_model_device(model)
    if model_device.type == "meta":
        raise RuntimeError("[QAT W4A4 Calib] FSDP model is still on meta device; cannot run activation calibration.")

    if model_device.type == "cuda":
        return model_device, False

    if model_device.type == "cpu" and device.type == "cuda":
        from verl.utils.fsdp_utils import load_fsdp_model_to_gpu

        load_fsdp_model_to_gpu(model)
        logger.info("[QAT W4A4 Calib] Loaded FSDP model to %s for activation calibration", device)
        return torch.device("cpu"), True

    raise RuntimeError(
        f"[QAT W4A4 Calib] Unsupported model device {model_device} for activation calibration on {device}"
    )


def _restore_fsdp_model_device(model: nn.Module, restore_device: torch.device, moved: bool) -> None:
    if not moved:
        return

    if restore_device.type == "cpu":
        from verl.utils.fsdp_utils import offload_fsdp_model_to_cpu

        offload_fsdp_model_to_cpu(model)
    else:
        model.to(restore_device)
    logger.info("[QAT W4A4 Calib] Restored model to %s after activation calibration", restore_device)


def _run_forward_batches(
    model: nn.Module,
    batches: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    max_batches: int,
) -> int:
    restore_device, moved = _ensure_fsdp_model_on_device(model, device)
    run_device = device

    model.train()
    ran = 0
    try:
        with torch.no_grad(), _calibration_forward_context(model, run_device):
            non_blocking = run_device.type != "cpu"
            for batch in batches:
                if ran >= max_batches:
                    break
                input_ids = batch["input_ids"].to(run_device, non_blocking=non_blocking)
                attention_mask = batch["attention_mask"].to(run_device, non_blocking=non_blocking)
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                ran += 1
        model.eval()
    finally:
        _restore_fsdp_model_device(model, restore_device, moved)

    return ran


def _iter_data_batches(
    tokenizer,
    data_files: list[str],
    *,
    num_samples: int,
    max_seq_len: int,
    batch_size: int,
    prompt_key: str,
):
    prompts = _load_calib_prompts(data_files, num_samples=num_samples, prompt_key=prompt_key)
    if not prompts:
        return

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id or 0

    tokenized = [_tokenize_prompt(tokenizer, prompt, max_seq_len) for prompt in prompts]
    for start in range(0, len(tokenized), batch_size):
        chunk = tokenized[start : start + batch_size]
        yield _collate_tokenized(chunk, pad_token_id=pad_token_id)


def _iter_synthetic_batches(
    model: nn.Module,
    *,
    num_samples: int,
    max_seq_len: int,
    batch_size: int,
    seed: int,
):
    vocab_size = model.config.vocab_size
    generator = torch.Generator()
    generator.manual_seed(seed)

    remaining = num_samples
    while remaining > 0:
        current_batch = min(batch_size, remaining)
        input_ids = torch.randint(1, vocab_size, (current_batch, max_seq_len), generator=generator)
        attention_mask = torch.ones(current_batch, max_seq_len, dtype=torch.long)
        yield {"input_ids": input_ids, "attention_mask": attention_mask}
        remaining -= current_batch


def run_w4a4_activation_calibration(
    model: nn.Module,
    *,
    tokenizer=None,
    data_files: Optional[list[str]] = None,
    num_samples: int = 32,
    max_seq_len: int = 2048,
    batch_size: int = 2,
    prompt_key: str = "prompt",
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> int:
    """Populate input_global_scale/input_amax via observer-forward passes on FSDP GPU model."""
    device = _normalize_device(device)

    total_before, uninit_before = count_uninitialized_w4a4_modules(model)
    moe_total, moe_uninit = count_uninitialized_moe_layers(model)
    if uninit_before == 0 and moe_uninit == 0:
        logger.info(
            "[QAT W4A4 Calib] All %d W4A4 linear and %d MoE layers already initialized, skipping",
            total_before,
            moe_total,
        )
        return 0

    logger.info(
        "[QAT W4A4 Calib] %d/%d W4A4 linear + %d/%d MoE layers need activation calibration (post-FSDP GPU forward)",
        uninit_before,
        total_before,
        moe_uninit,
        moe_total,
    )

    max_batches = max(1, (num_samples + batch_size - 1) // batch_size)
    if data_files and tokenizer is not None:
        batches = _iter_data_batches(
            tokenizer,
            data_files,
            num_samples=num_samples,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            prompt_key=prompt_key,
        )
        source = f"parquet ({data_files})"
    else:
        if data_files and tokenizer is None:
            raise RuntimeError(
                "[QAT W4A4 Calib] calib_data_files is set but tokenizer is missing; "
                "cannot run activation forward calibration."
            )
        logger.warning("[QAT W4A4 Calib] No calib_data_files; using synthetic random tokens")
        batches = _iter_synthetic_batches(
            model,
            num_samples=num_samples,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            seed=seed,
        )
        source = "synthetic random tokens"

    ran_batches = _run_forward_batches(model, batches, device=device, max_batches=max_batches)

    from verl.utils.qat.moe import fallback_moe_scales_from_weights

    fallback_moe_scales_from_weights(model)

    _, uninit_after = count_uninitialized_w4a4_modules(model)
    _, moe_uninit_after = count_uninitialized_moe_layers(model)
    calibrated = (uninit_before - uninit_after) + (moe_uninit - moe_uninit_after)

    logger.info(
        "[QAT W4A4 Calib] Finished %d batches from %s; initialized %d scale groups "
        "(%d linear + %d MoE still uninitialized)",
        ran_batches,
        source,
        calibrated,
        uninit_after,
        moe_uninit_after,
    )
    get_torch_device().empty_cache()

    if uninit_after > 0 or moe_uninit_after > 0:
        raise RuntimeError(
            f"[QAT W4A4 Calib] {uninit_after} linear and {moe_uninit_after} MoE layers still have "
            "uninitialized input_global_scale/input_amax after post-FSDP activation forward calibration."
        )
    return calibrated


def maybe_calibrate_w4a4_activations(
    model: nn.Module,
    qat_config: Any,
    *,
    tokenizer=None,
    device: Optional[torch.device] = None,
    post_fsdp: bool = False,
) -> int:
    """Run post-FSDP GPU activation forward calibration when W4A4 scales are missing."""
    if not getattr(qat_config, "enable", False):
        return 0
    if getattr(qat_config, "mode", "").lower() != "w4a4":
        return 0
    if not getattr(qat_config, "calib_enable", True):
        return 0
    if not post_fsdp:
        logger.info(
            "[QAT W4A4 Calib] Skipping pre-FSDP path; activation calibration runs after FSDP wrap on GPU"
        )
        return 0
    if not needs_w4a4_calibration(model):
        return 0

    data_files = getattr(qat_config, "calib_data_files", None)
    if isinstance(data_files, str):
        data_files = [data_files]

    return run_w4a4_activation_calibration(
        model,
        tokenizer=tokenizer,
        data_files=data_files,
        num_samples=getattr(qat_config, "calib_num_samples", 32),
        max_seq_len=getattr(qat_config, "calib_max_seq_len", 2048),
        batch_size=getattr(qat_config, "calib_batch_size", 2),
        prompt_key=getattr(qat_config, "calib_prompt_key", "prompt"),
        seed=getattr(qat_config, "calib_seed", 42),
        device=device,
    )


__all__ = [
    "UNINITIALIZED_SCALE",
    "count_uninitialized_moe_layers",
    "count_uninitialized_w4a4_modules",
    "input_global_scale_from_amax",
    "is_scale_uninitialized",
    "maybe_calibrate_w4a4_activations",
    "needs_moe_calibration",
    "needs_w4a4_calibration",
    "run_w4a4_activation_calibration",
]
