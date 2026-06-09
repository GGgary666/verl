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
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Optional

import torch
import torch.nn as nn
from compressed_tensors.quantization.quant_args import FP4_E2M1_DATA, FP8_E4M3_DATA

from verl.utils.device import get_device_name, get_torch_device

logger = logging.getLogger(__name__)

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


def needs_w4a4_calibration(model: nn.Module) -> bool:
    total, uninitialized = count_uninitialized_w4a4_modules(model)
    return total > 0 and uninitialized > 0


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


@contextmanager
def _calibration_forward_context(model: nn.Module, device: torch.device) -> Iterator[None]:
    """Use sdpa + bf16 autocast for pre-FSDP fp32-weight calibration forwards."""
    config = getattr(model, "config", None)
    original_attn = getattr(config, "_attn_implementation", None) if config is not None else None
    weight_dtype = _get_model_weight_dtype(model)

    if config is not None and original_attn in ("flash_attention_2", "flash_attention_3"):
        config._attn_implementation = "sdpa"
        logger.info(
            "[QAT W4A4 Calib] Switched attention %s -> sdpa (pre-FSDP model_dtype=%s, flash-attn needs bf16/fp16)",
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
            mask = torch.cat([torch.ones(ids.shape[0] - pad_len), torch.zeros(pad_len, dtype=torch.long)])
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


def _run_forward_batches(
    model: nn.Module,
    batches: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    max_batches: int,
) -> int:
    model_device = _get_model_device(model)
    run_device = device
    moved = False
    if model_device.type == "meta":
        model.to(run_device)
        moved = True
        logger.info("[QAT W4A4 Calib] Moved model from meta to %s for calibration", run_device)
    elif model_device != run_device:
        model.to(run_device)
        moved = True
        logger.info(
            "[QAT W4A4 Calib] Moved model from %s to %s for calibration",
            model_device,
            run_device,
        )

    restore_device = model_device if model_device.type != "meta" else torch.device("cpu")
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
        if moved:
            model.to(restore_device)
            logger.info("[QAT W4A4 Calib] Restored model to %s after calibration", restore_device)

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
    pad_token_id = getattr(model.config, "pad_token_id", None) or 0
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
    """Populate input_global_scale/input_amax via observer-forward passes."""
    device = _normalize_device(device)

    total_before, uninit_before = count_uninitialized_w4a4_modules(model)
    if uninit_before == 0:
        logger.info("[QAT W4A4 Calib] All %d W4A4 layers already initialized, skipping", total_before)
        return 0

    logger.info(
        "[QAT W4A4 Calib] %d/%d W4A4 layers need activation calibration",
        uninit_before,
        total_before,
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
            logger.warning("[QAT W4A4 Calib] calib_data_files set but tokenizer missing, using synthetic data")
        batches = _iter_synthetic_batches(
            model,
            num_samples=num_samples,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            seed=seed,
        )
        source = "synthetic random tokens"

    ran_batches = _run_forward_batches(model, batches, device=device, max_batches=max_batches)
    _, uninit_after = count_uninitialized_w4a4_modules(model)
    calibrated = uninit_before - uninit_after

    logger.info(
        "[QAT W4A4 Calib] Finished %d batches from %s; initialized %d layers (%d still uninitialized)",
        ran_batches,
        source,
        calibrated,
        uninit_after,
    )
    get_torch_device().empty_cache()
    return calibrated


def maybe_calibrate_w4a4_activations(
    model: nn.Module,
    qat_config: Any,
    *,
    tokenizer=None,
    device: Optional[torch.device] = None,
) -> int:
    """Run calibration when W4A4 scales are missing and calib_enable is True."""
    if not getattr(qat_config, "enable", False):
        return 0
    if getattr(qat_config, "mode", "").lower() != "w4a4":
        return 0
    if not getattr(qat_config, "calib_enable", True):
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
    "count_uninitialized_w4a4_modules",
    "input_global_scale_from_amax",
    "is_scale_uninitialized",
    "maybe_calibrate_w4a4_activations",
    "needs_w4a4_calibration",
    "run_w4a4_activation_calibration",
]
