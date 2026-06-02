# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Single Process Actor
"""

import logging
import os
from contextlib import contextmanager

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.use_dynamic_bsz = self.config.get("use_dynamic_bsz", False)

        self.use_prefix_grouper = self.config.get("use_prefix_grouper", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_prefix_grouper={self.use_prefix_grouper}")

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        # Sum of squared probabilities computation (for optimal_token_baseline)
        # Only initialize if calculate_sum_pi_squared config is enabled
        if self.config.get("calculate_sum_pi_squared", False):
            self.calculate_sum_pi_squared_from_logits = (
                torch.compile(verl_F.calculate_sum_pi_squared_from_logits, dynamic=True)
                if self.config.get("use_torch_compile", True)
                else verl_F.calculate_sum_pi_squared_from_logits
            )
            assert not (self.use_fused_kernels or self.use_prefix_grouper), (
                "calculate_sum_pi_squared is not supported with "
                f"{self.use_fused_kernels=} or {self.use_prefix_grouper=} for now."
            )

        self._compiled_actor_model_forward_old_log_prob = None
        self._compiled_actor_model_forward_teacher = None
        if self.config.get("compile_forward_for_old_log_prob", False):
            self._compiled_actor_model_forward_old_log_prob = torch.compile(
                self._actor_model_forward_impl, dynamic=True
            )
        if self.config.get("compile_forward_for_self_distill_teacher", False):
            self._compiled_actor_model_forward_teacher = torch.compile(
                self._actor_model_forward_impl, dynamic=True
            )
        self._qat_optimizer_step = 0

    def _set_qat_observer_train_step(self, step: int):
        """Propagate optimizer-step index to all QATLinear modules."""
        for module in self.actor_module.modules():
            setter = getattr(module, "set_activation_observer_train_step", None)
            if callable(setter):
                setter(step)

    def _actor_model_forward_impl(self, **model_kwargs):
        """Forward helper so we can optionally torch.compile no-grad recompute paths."""
        return self.actor_module(**model_kwargs)

    def _actor_model_forward(self, model_kwargs: dict, forward_mode: str):
        """Dispatch actor forward according to recompute mode."""
        if forward_mode == "old_log_prob" and self._compiled_actor_model_forward_old_log_prob is not None:
            return self._compiled_actor_model_forward_old_log_prob(**model_kwargs)
        if forward_mode == "teacher" and self._compiled_actor_model_forward_teacher is not None:
            return self._compiled_actor_model_forward_teacher(**model_kwargs)
        return self._actor_model_forward_impl(**model_kwargs)

    def _forward_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        calculate_entropy: bool = False,
        forward_mode: str = "default",
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict[str, torch.Tensor]:
                log_probs: (bs, response_len)
                if calculate_entropy is True:
                    entropys: (bs, response_len)
                if calculate_sum_pi_squared is False:
                    sum_pi_squared: (bs, response_len)
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
        sum_pi_squared_checkpointing = self.config.get("sum_pi_squared_checkpointing", False)
        # PrefixGrouper path for shared-prefix optimization
        if self.use_prefix_grouper:
            can_use_pg = (
                not self.use_remove_padding
                and not self.use_ulysses_sp
                and not self.use_fused_kernels
                and not self.use_dynamic_bsz
            )
            if can_use_pg and "response_mask" in micro_batch and "uid" in micro_batch:
                from verl.trainer.ppo.prefix_grouper_utils import forward_micro_batch_with_prefix_grouper

                return forward_micro_batch_with_prefix_grouper(
                    micro_batch=micro_batch,
                    model=self.actor_module,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    device_name=self.device_name,
                    param_dtype=self.param_dtype,
                    use_chunking_entropy=self.config.get("entropy_from_logits_with_chunking", False),
                )

        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                model_kwargs = {
                    "input_ids": input_ids_rmpad,
                    "attention_mask": None,
                    "position_ids": position_ids_rmpad,
                    **multi_modal_inputs,
                    "use_cache": False,
                    **extra_args,
                }
                output = self._actor_model_forward(
                    model_kwargs=model_kwargs,
                    forward_mode=forward_mode,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        # ((total_nnz / sp) + pad)
                        entropy_rmpad = (
                            self.compute_entropy_from_logits(logits_rmpad)
                            if not self.config.entropy_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)
                        )

                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = (
                            self.calculate_sum_pi_squared_from_logits(logits_rmpad)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(
                                self.calculate_sum_pi_squared_from_logits, logits_rmpad
                            )
                        )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = gather_outputs_and_unpad(
                            sum_pi_squared_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if calculate_sum_pi_squared:
                    full_sum_pi_squared = pad_input(
                        hidden_states=sum_pi_squared_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if calculate_sum_pi_squared:
                    # (bsz, response_length)
                    sum_pi_squared = full_sum_pi_squared.squeeze(-1)[:, -response_length - 1 : -1]
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                model_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    **multi_modal_inputs,
                    "use_cache": False,
                    **extra_args,
                }
                output = self._actor_model_forward(
                    model_kwargs=model_kwargs,
                    forward_mode=forward_mode,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared = (
                            self.calculate_sum_pi_squared_from_logits(logits)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.calculate_sum_pi_squared_from_logits, logits)
                        )

            outputs = {"log_probs": log_probs}
            if calculate_entropy:
                outputs["entropys"] = entropy
            if calculate_sum_pi_squared:
                outputs["sum_pi_squared"] = sum_pi_squared
            return outputs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()

        # Clear cached QAT weight-scale/amax states after optimizer.step().
        if getattr(self.actor_module, "_qat_enabled", False) or getattr(
            self.actor_module, "_qat_fuse_enabled", False
        ):
            from verl.utils.qat import invalidate_all_scales

            invalidate_all_scales(self.actor_module)
            self._qat_optimizer_step += 1
            self._set_qat_observer_train_step(self._qat_optimizer_step)

        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(
        self, data: DataProto, calculate_entropy: bool = False, forward_mode: str = "old_log_prob"
    ) -> dict[str, torch.Tensor]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            dict[str, torch.Tensor]: a dict containing keys
                - ``log_probs``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``entropys``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``sum_pi_squared``: tensor of shape [batch_size, response_length]. torch.float32.
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)

        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if self.use_prefix_grouper:
            select_keys += [k for k in ["prompts", "response_mask"] if k in data.batch]
            if "uid" in data.non_tensor_batch:
                non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        sum_pi_squared_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            with torch.no_grad():
                outputs = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    forward_mode=forward_mode,
                )
            log_probs_lst.append(outputs["log_probs"])
            if calculate_entropy:
                entropy_lst.append(outputs["entropys"])
            if calculate_sum_pi_squared:
                sum_pi_squared_lst.append(outputs["sum_pi_squared"])

        log_probs = torch.concat(log_probs_lst, dim=0)
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if calculate_sum_pi_squared:
            sum_pi_squared = torch.concat(sum_pi_squared_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if calculate_sum_pi_squared:
                sum_pi_squared = restore_dynamic_batch(sum_pi_squared, batch_idx_list)

        outputs = {"log_probs": log_probs}
        if calculate_entropy:
            outputs["entropys"] = entropys
        if calculate_sum_pi_squared:
            outputs["sum_pi_squared"] = sum_pi_squared
        return outputs

    @contextmanager
    def _temporarily_set_qat_fake_quant(self, enabled: bool):
        """Temporarily set QAT fake-quant runtime mode for teacher forward."""
        try:
            from verl.utils.qat.linear import QATLinear
        except Exception:
            QATLinear = None
        try:
            from verl.utils.qat.fused_rms_norm_fake_quant import FusedRMSNormFakeQuant
        except Exception:
            FusedRMSNormFakeQuant = None

        qat_modules_and_states: list[tuple[nn.Module, bool, bool]] = []
        fused_norm_states: list[tuple[nn.Module, bool]] = []
        for module in self.actor_module.modules():
            if QATLinear is not None and isinstance(module, QATLinear) and hasattr(module, "fake_quant_enabled"):
                old_fake_quant_enabled = bool(module.fake_quant_enabled)
                old_observe_input_scale_only = bool(getattr(module, "observe_input_scale_only", False))
                qat_modules_and_states.append((module, old_fake_quant_enabled, old_observe_input_scale_only))
                module.fake_quant_enabled = bool(enabled)
                if hasattr(module, "observe_input_scale_only"):
                    module.observe_input_scale_only = bool((not enabled) and old_observe_input_scale_only)
            elif FusedRMSNormFakeQuant is not None and isinstance(module, FusedRMSNormFakeQuant):
                fused_norm_states.append((module, bool(module.fake_quant_enabled)))
                module.fake_quant_enabled = bool(enabled)
        try:
            yield
        finally:
            for module, old_fake_quant_enabled, old_observe_input_scale_only in qat_modules_and_states:
                module.fake_quant_enabled = old_fake_quant_enabled
                if hasattr(module, "observe_input_scale_only"):
                    module.observe_input_scale_only = old_observe_input_scale_only
            for module, old_fake_quant_enabled in fused_norm_states:
                module.fake_quant_enabled = old_fake_quant_enabled

    @contextmanager
    def _temporarily_set_eval_mode(self):
        """Temporarily set actor module to eval mode."""
        was_training = self.actor_module.training
        try:
            self.actor_module.eval()
            yield
        finally:
            self.actor_module.train(was_training)

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_teacher_log_prob(self, data: DataProto) -> dict[str, torch.Tensor]:
        """Compute teacher log-probs for self-distill (BF16 or QAT teacher)."""
        teacher_use_qat_forward = bool(self.config.get("self_distill_teacher_use_qat_forward", False))
        with self._temporarily_set_eval_mode():
            with self._temporarily_set_qat_fake_quant(enabled=teacher_use_qat_forward):
                return self.compute_log_prob(data=data, calculate_entropy=False, forward_mode="teacher")

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()
        if getattr(self.actor_module, "_qat_enabled", False) or getattr(self.actor_module, "_qat_fuse_enabled", False):
            self._set_qat_observer_train_step(self._qat_optimizer_step)

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        pad_token_id = data.meta_info.get("pad_token_id", 0)

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.use_prefix_grouper and "prompts" in data.batch.keys():
            select_keys.append("prompts")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # ACR (Adaptive Clipping Range) per-token r_i,t when rollout_acr_enabled
        if "rollout_acr_r" in data.batch.keys():
            select_keys.append("rollout_acr_r")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")
        if "old_bf16_log_probs" in data.batch.keys():
            select_keys.append("old_bf16_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if self.use_prefix_grouper and "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1
        self_distill_enable = bool(self.config.get("self_distill_enable", False))
        self_distill_coef = float(self.config.get("self_distill_coef", 0.001))
        self_distill_loss_type = str(self.config.get("self_distill_loss_type", "low_var_kl")).strip().lower()
        self_distill_teacher_forward_mode = str(
            self.config.get("self_distill_teacher_forward_mode", "per_micro_batch")
        ).strip().lower()
        self_distill_teacher_use_qat_forward = bool(self.config.get("self_distill_teacher_use_qat_forward", False))
        self_distill_observe_only = bool(self.config.get("self_distill_observe_only", False))
        self_distill_conflict_hard_gate_enable = bool(
            self.config.get("self_distill_conflict_hard_gate_enable", False)
        )
        self_distill_conflict_track_without_hard_gate = bool(
            self.config.get("self_distill_conflict_track_without_hard_gate", False)
        )
        self_distill_active = self_distill_enable and (self_distill_coef > 0.0 or self_distill_observe_only)

        if self_distill_loss_type not in {"abs_logprob", "mse", "low_var_kl"}:
            raise ValueError(
                "self_distill_loss_type must be 'abs_logprob', 'mse', or 'low_var_kl', "
                f"got {self_distill_loss_type!r}."
            )
        if self_distill_teacher_forward_mode not in {"per_micro_batch", "per_train_batch"}:
            raise ValueError(
                "self_distill_teacher_forward_mode must be 'per_micro_batch' or 'per_train_batch', "
                f"got {self_distill_teacher_forward_mode!r}."
            )
        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
        }
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    outputs = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )
                    log_prob = outputs["log_probs"]
                    entropy = outputs["entropys"] if calculate_entropy else None

                    # for fully_async_policy
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)
                    rollout_acr_r = model_inputs.get("rollout_acr_r", None)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)
                    tbpo_seq_mismatch_weight = None
                    tbpo_seq_keep_mask = None
                    tbpo_metrics = {"actor/tbpo_active": 0.0}
                    if loss_mode == "tbpo":
                        from verl.trainer.ppo.rollout_corr_helper import compute_tbpo_sequence_stats

                        if rollout_log_prob is None:
                            raise RuntimeError(
                                "TBPO loss requires rollout_log_probs in batch. "
                                "Set actor_rollout_ref.rollout.calculate_log_probs=True."
                            )
                        rollout_corr_cfg = (
                            self.config.policy_loss.get("rollout_correction", None)
                            if hasattr(self.config, "policy_loss")
                            else None
                        )
                        if rollout_corr_cfg is None:
                            raise RuntimeError(
                                "TBPO loss requires policy_loss.rollout_correction config to be present."
                            )
                        tbpo_cap = float(rollout_corr_cfg.get("tbpo_seq_tis_imp_ratio_cap", 2.0))
                        tbpo_tensors, tbpo_stats = compute_tbpo_sequence_stats(
                            old_log_prob=old_log_prob.detach(),
                            rollout_log_prob=rollout_log_prob.detach(),
                            response_mask=response_mask,
                            current_log_prob=log_prob.detach(),
                            seq_mismatch_cap=tbpo_cap,
                        )
                        tbpo_seq_mismatch_weight = tbpo_tensors["tbpo_seq_mismatch_weight_clipped"]
                        tbpo_seq_keep_mask = tbpo_tensors.get("tbpo_seq_keep_mask_w", tbpo_tensors["tbpo_seq_keep_mask"])
                        if rollout_is_weights is not None:
                            tbpo_metrics["actor/tbpo_rollout_is_disabled"] = 1.0
                            rollout_is_weights = None
                        else:
                            tbpo_metrics["actor/tbpo_rollout_is_disabled"] = 0.0
                        if rollout_acr_r is not None:
                            rollout_acr_r = None
                            tbpo_metrics["actor/tbpo_acr_disabled"] = 1.0
                        else:
                            tbpo_metrics["actor/tbpo_acr_disabled"] = 0.0
                        tbpo_metrics.update({f"actor/{k}": v for k, v in tbpo_stats.items()})
                        tbpo_metrics["actor/tbpo_active"] = 1.0

                    # Compute policy loss (any function is expected to return 2 values)
                    policy_loss_kw = dict(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                        rollout_acr_r=rollout_acr_r,
                    )
                    if loss_mode == "tbpo":
                        policy_loss_kw["tbpo_seq_mismatch_weight"] = tbpo_seq_mismatch_weight
                        policy_loss_kw["tbpo_seq_keep_mask"] = tbpo_seq_keep_mask
                    pg_loss, pg_metrics = policy_loss_fn(**policy_loss_kw)
                    micro_batch_metrics.update(pg_metrics)
                    micro_batch_metrics.update(tbpo_metrics)

                    # Skip if using bypass_mode loss (metrics already computed in pg_metrics)
                    if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self_distill_active:
                        teacher_log_prob = None
                        precomputed_teacher_log_prob = model_inputs.get("old_bf16_log_probs", None)
                        if (
                            self_distill_teacher_forward_mode == "per_train_batch"
                            and not self_distill_teacher_use_qat_forward
                            and precomputed_teacher_log_prob is not None
                            and tuple(precomputed_teacher_log_prob.shape) == tuple(log_prob.shape)
                        ):
                            teacher_log_prob = precomputed_teacher_log_prob.detach()
                        else:
                            with torch.no_grad():
                                with self._temporarily_set_eval_mode():
                                    with self._temporarily_set_qat_fake_quant(
                                        enabled=self_distill_teacher_use_qat_forward
                                    ):
                                        teacher_outputs = self._forward_micro_batch(
                                            model_inputs,
                                            temperature=temperature,
                                            calculate_entropy=False,
                                            forward_mode="teacher",
                                        )
                            teacher_log_prob = teacher_outputs["log_probs"].detach()

                        logprob_gap = log_prob - teacher_log_prob
                        if self_distill_loss_type == "abs_logprob":
                            self_distill_token_obj = torch.abs(logprob_gap)
                        elif self_distill_loss_type == "low_var_kl":
                            logprob_delta = teacher_log_prob - log_prob
                            logprob_delta = torch.clamp(logprob_delta, min=-20.0, max=20.0)
                            self_distill_token_obj = torch.exp(logprob_delta) - logprob_delta - 1.0
                            self_distill_token_obj = torch.clamp(self_distill_token_obj, min=-10.0, max=10.0)
                        else:
                            self_distill_token_obj = torch.square(logprob_gap)

                        token_weights = response_mask.to(dtype=log_prob.dtype)
                        response_mask_bool = response_mask.to(dtype=torch.bool)
                        if not self_distill_conflict_hard_gate_enable:
                            micro_batch_metrics["actor/self_distill_conflict_hard_gate_enabled"] = 0.0
                            if response_mask_bool.any():
                                if self_distill_conflict_track_without_hard_gate:
                                    sd_direction = torch.sign(
                                        (teacher_log_prob.detach() - log_prob.detach()).to(torch.float32)
                                    )
                                    pg_direction = torch.sign(advantages.detach().to(torch.float32))
                                    conflict_mask = ((sd_direction * pg_direction) < 0) & response_mask_bool
                                    conflict_frac = float(conflict_mask[response_mask_bool].float().mean().item())
                                    micro_batch_metrics["actor/self_distill_conflict_token_frac"] = conflict_frac
                                    micro_batch_metrics["actor/self_distill_conflict_kept_weight_frac"] = conflict_frac
                                else:
                                    micro_batch_metrics["actor/self_distill_conflict_token_frac"] = 0.0
                                    micro_batch_metrics["actor/self_distill_conflict_kept_weight_frac"] = 1.0
                                scope_mask = response_mask_bool
                                micro_batch_metrics["actor/self_distill_conflict_scope_token_frac"] = float(
                                    scope_mask[response_mask_bool].float().mean().item()
                                )
                            else:
                                micro_batch_metrics["actor/self_distill_conflict_scope_token_frac"] = 0.0
                                micro_batch_metrics["actor/self_distill_conflict_token_frac"] = 0.0
                                micro_batch_metrics["actor/self_distill_conflict_kept_weight_frac"] = 0.0
                        else:
                            micro_batch_metrics["actor/self_distill_conflict_hard_gate_enabled"] = 1.0
                            if response_mask_bool.any():
                                sd_direction = torch.sign((teacher_log_prob.detach() - log_prob.detach()).to(torch.float32))
                                pg_direction = torch.sign(advantages.detach().to(torch.float32))
                                scope_mask = response_mask_bool
                                micro_batch_metrics["actor/self_distill_conflict_scope_token_frac"] = float(
                                    scope_mask[response_mask_bool].float().mean().item()
                                )
                                conflict_mask = ((sd_direction * pg_direction) < 0) & scope_mask
                                valid_conflict = conflict_mask[response_mask_bool]
                                micro_batch_metrics["actor/self_distill_conflict_token_frac"] = float(
                                    valid_conflict.float().mean().item()
                                )
                                weight_sum_pre = float(token_weights.detach()[response_mask_bool].sum().item())
                                token_weights = token_weights * (~conflict_mask).to(token_weights.dtype)
                                weight_sum_post = float(token_weights.detach()[response_mask_bool].sum().item())
                                if weight_sum_pre > 0.0:
                                    micro_batch_metrics["actor/self_distill_conflict_kept_weight_frac"] = (
                                        weight_sum_post / weight_sum_pre
                                    )
                                else:
                                    micro_batch_metrics["actor/self_distill_conflict_kept_weight_frac"] = 0.0
                            else:
                                micro_batch_metrics["actor/self_distill_conflict_scope_token_frac"] = 0.0
                                micro_batch_metrics["actor/self_distill_conflict_token_frac"] = 0.0
                                micro_batch_metrics["actor/self_distill_conflict_kept_weight_frac"] = 0.0
                        eps = torch.finfo(log_prob.dtype).eps
                        self_distill_loss = (self_distill_token_obj * token_weights).sum() / (token_weights.sum() + eps)
                        self_distill_applied_coef = 0.0 if self_distill_observe_only else self_distill_coef
                        if self_distill_applied_coef != 0.0:
                            policy_loss = policy_loss + self_distill_applied_coef * self_distill_loss
                        micro_batch_metrics["actor/self_distill_loss"] = self_distill_loss.detach().item()
                        micro_batch_metrics["actor/self_distill_logprob_gap_mean"] = (
                            (logprob_gap * token_weights).sum() / (token_weights.sum() + eps)
                        ).detach().item()
                        micro_batch_metrics["actor/self_distill_coef"] = self_distill_coef
                        micro_batch_metrics["actor/self_distill_applied_coef"] = self_distill_applied_coef
                        micro_batch_metrics["actor/self_distill_teacher_use_qat_forward"] = float(
                            self_distill_teacher_use_qat_forward
                        )
                        micro_batch_metrics["actor/self_distill_observe_only"] = float(self_distill_observe_only)

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    if getattr(self.actor_module, "_qat_batched_amax_sync", False):
                        from verl.utils.qat import sync_activation_observer_amax

                        sync_activation_observer_amax(self.actor_module)

                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
