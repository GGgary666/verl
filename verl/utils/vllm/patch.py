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

import logging
import re

import torch

logger = logging.getLogger(__name__)

# To support different vLLM versions, we add the model into SUPPORTED_MOE_MODELS separately to avoid triggering
# unsupported issues.
SUPPORTED_MOE_MODELS = []

try:
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM, DeepseekV3ForCausalLM

    SUPPORTED_MOE_MODELS.append(DeepseekV2ForCausalLM)
    SUPPORTED_MOE_MODELS.append(DeepseekV3ForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.mixtral import MixtralForCausalLM

    SUPPORTED_MOE_MODELS.append(MixtralForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen2MoeForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen3MoeForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen3_vl_moe import Qwen3MoeLLMForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen3MoeLLMForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen3_next import Qwen3NextForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen3NextForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.kimi_vl import KimiVLForConditionalGeneration

    SUPPORTED_MOE_MODELS.append(KimiVLForConditionalGeneration)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen3_5 import Qwen3_5MoeForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen3_5MoeForCausalLM)
except ImportError:
    pass


_EXPERT_ID_RE = re.compile(r"experts\.(\d+)\.")


def _make_moe_weight_loader_wrapper(original_weight_loader, is_w13=False):
    """Wrap ``FusedMoE.weight_loader`` to robustly handle per-expert 2D weights.

    vLLM >= 0.22 stores routed expert weights as fused 3-D tensors
    (``w13_weight``: ``[E, 2*inter, hidden]``, ``w2_weight``: ``[E, hidden, inter]``).
    The canonical ``FusedMoE.weight_loader`` requires an explicit ``expert_id``
    to slice the correct 2-D expert slab out of the fused parameter.  When the
    caller (e.g. ``AutoWeightsLoader._load_param`` or a model-level
    ``load_weights`` that does not use the ``expert_mapping``) omits
    ``expert_id`` or passes a 3-D single-expert tensor (``[1, ...]``), the loader
    falls through with ``expert_data = param.data`` (full 3-D) and crashes in
    ``_get_hidden_dim``.

    This wrapper intercepts those cases and fixes the arguments before
    delegating to the original loader.
    """

    # Get FusedMoE instance from the bound method for TP info
    _fused_moe = getattr(original_weight_loader, "__self__", None)

    def wrapper(param, loaded_weight, weight_name=None, shard_id=None, expert_id=None, return_success=False):
        # ------------------------------------------------------------------
        # Case 1: Called with only 2 positional args (AutoWeightsLoader path).
        # ------------------------------------------------------------------
        if weight_name is None:
            if shard_id is None:
                shard_id = "w1" if is_w13 else "w2"
            return original_weight_loader(
                param,
                loaded_weight,
                weight_name="",
                shard_id=shard_id,
                expert_id=expert_id if expert_id is not None else 0,
                return_success=return_success,
            )

        if isinstance(weight_name, torch.Tensor):
            actual_weight = weight_name
            inferred_shard = "w1" if is_w13 else "w2"
            return original_weight_loader(
                param,
                actual_weight,
                weight_name="",
                shard_id=inferred_shard,
                expert_id=0,
                return_success=return_success,
            )

        # ------------------------------------------------------------------
        # Case 2: Extract expert_id from weight_name if not provided.
        # ------------------------------------------------------------------
        has_per_expert_name = False
        if isinstance(weight_name, str):
            m = _EXPERT_ID_RE.search(weight_name)
            if m is not None:
                has_per_expert_name = True
                if expert_id is None:
                    expert_id = int(m.group(1))

        # ------------------------------------------------------------------
        # Case 3: If loaded_weight is 3-D, squeeze to 2-D.
        # ------------------------------------------------------------------
        if isinstance(loaded_weight, torch.Tensor) and loaded_weight.dim() == 3:
            loaded_weight = loaded_weight[0].contiguous()

        # ------------------------------------------------------------------
        # Case 4: Infer shard_id from the weight name when not provided.
        # ------------------------------------------------------------------
        if shard_id is None and isinstance(weight_name, str):
            if "gate_proj" in weight_name:
                shard_id = "w1"
            elif "up_proj" in weight_name:
                shard_id = "w3"
            elif "down_proj" in weight_name:
                shard_id = "w2"
            else:
                shard_id = "w1" if is_w13 else "w2"

        if expert_id is None:
            expert_id = 0

        # ------------------------------------------------------------------
        # DIRECT COPY PATH: For per-expert 2D weights into fused 3D/4D
        # params, bypass the original weight_loader to avoid the
        # shard_dim/ndim mismatch.  This handles TP sharding manually.
        #
        # vLLM >= 0.22 may store fused MoE params as:
        #   3D: [num_experts, 2*inter_per_tp, hidden_per_tp]
        #   4D: [num_experts, 2*inter_per_tp, tp_size, hidden_per_tp]
        #       or other packed layouts.
        # For 4D params, expert_data = param.data[expert_id] is 3D, and
        # the original weight_loader cannot handle it with shard_dim=0.
        # We handle both cases by directly copying weight data.
        # ------------------------------------------------------------------
        if (
            isinstance(loaded_weight, torch.Tensor)
            and loaded_weight.dim() == 2
            and isinstance(param, (torch.nn.Parameter, torch.Tensor))
            and param.dim() >= 3
            and shard_id in ("w1", "w2", "w3")
        ):
            try:
                tp_rank = getattr(_fused_moe, "tp_rank", 0) if _fused_moe else 0
                tp_size = getattr(_fused_moe, "tp_size", 1) if _fused_moe else 1

                # Map global expert_id to local
                if _fused_moe is not None:
                    local_expert_id = _fused_moe._map_global_expert_id_to_local_expert_id(expert_id)
                    if local_expert_id == -1:
                        return False if return_success else None
                else:
                    local_expert_id = expert_id

                expert_data = param.data[local_expert_id]  # 2D for 3D param, 3D for 4D param

                if expert_data.dim() == 2:
                    # Standard 3D fused param: [E, X, Y] → expert_data is [X, Y]
                    if shard_id in ("w1", "w3"):
                        shard_size = expert_data.shape[0] // 2
                        if shard_id == "w1":
                            target = expert_data[:shard_size]
                        else:
                            target = expert_data[shard_size:]

                        target_inter = target.shape[0]
                        target_hidden = target.shape[1]
                        if loaded_weight.shape == (target_inter * tp_size, target_hidden):
                            is_transposed = False
                        elif loaded_weight.shape == (target_hidden, target_inter * tp_size):
                            is_transposed = True
                        elif loaded_weight.shape == target.shape:
                            target.copy_(loaded_weight)
                            return True if return_success else None
                        else:
                            target.copy_(loaded_weight)
                            return True if return_success else None

                        if not is_transposed:
                            if tp_size > 1:
                                tp_shard = target_inter
                                start = tp_shard * tp_rank
                                loaded_shard = loaded_weight[start : start + tp_shard]
                            else:
                                loaded_shard = loaded_weight
                            target.copy_(loaded_shard)
                        else:
                            if tp_size > 1:
                                tp_shard = target_inter
                                start = tp_shard * tp_rank
                                loaded_shard = loaded_weight[:, start : start + tp_shard]
                            else:
                                loaded_shard = loaded_weight
                            target.copy_(loaded_shard.t().contiguous())
                    else:  # w2
                        target = expert_data
                        target_hidden = target.shape[0]
                        target_inter = target.shape[1]
                        if loaded_weight.shape == (target_hidden, target_inter * tp_size):
                            is_transposed = False
                        elif loaded_weight.shape == (target_inter * tp_size, target_hidden):
                            is_transposed = True
                        elif loaded_weight.shape == target.shape:
                            target.copy_(loaded_weight)
                            return True if return_success else None
                        else:
                            target.copy_(loaded_weight)
                            return True if return_success else None

                        if not is_transposed:
                            if tp_size > 1:
                                tp_shard = target_inter
                                start = tp_shard * tp_rank
                                loaded_shard = loaded_weight[:, start : start + tp_shard]
                            else:
                                loaded_shard = loaded_weight
                            target.copy_(loaded_shard)
                        else:
                            if tp_size > 1:
                                tp_shard = target_inter
                                start = tp_shard * tp_rank
                                loaded_shard = loaded_weight[start : start + tp_shard]
                            else:
                                loaded_shard = loaded_weight
                            target.copy_(loaded_shard.t().contiguous())

                elif expert_data.dim() == 3:
                    # 4D fused param: [E, X, Y, Z] → expert_data is [X, Y, Z]
                    # Reshape the 2D loaded_weight to match the 3D expert slice.
                    # For w1/w3: first half of dim 0 is w1, second half is w3.
                    # For w2: the full expert slice.
                    target_numel = expert_data.numel()
                    loaded_numel = loaded_weight.numel()

                    if shard_id in ("w1", "w3"):
                        # w1 and w3 each occupy half of the fused dim
                        half_numel = target_numel // 2
                        if loaded_numel == half_numel:
                            # loaded_weight matches exactly one half
                            reshaped = loaded_weight.reshape(expert_data.shape[0] // 2, *expert_data.shape[1:])
                            if shard_id == "w1":
                                expert_data[:expert_data.shape[0] // 2].copy_(reshaped)
                            else:
                                expert_data[expert_data.shape[0] // 2:].copy_(reshaped)
                        elif loaded_numel == target_numel:
                            # loaded_weight covers both halves — take the right half
                            full_reshaped = loaded_weight.reshape(expert_data.shape)
                            if shard_id == "w1":
                                expert_data.copy_(full_reshaped[:expert_data.shape[0] // 2])
                            else:
                                expert_data.copy_(full_reshaped[expert_data.shape[0] // 2:])
                        else:
                            # Best effort: flatten and copy
                            flat_target = expert_data.view(-1)
                            if loaded_numel <= flat_target.numel():
                                flat_target[:loaded_numel].copy_(loaded_weight.view(-1))
                            else:
                                flat_target.copy_(loaded_weight.view(-1)[:flat_target.numel()])
                    else:  # w2
                        if loaded_numel == target_numel:
                            expert_data.copy_(loaded_weight.reshape(expert_data.shape))
                        else:
                            flat_target = expert_data.view(-1)
                            if loaded_numel <= flat_target.numel():
                                flat_target[:loaded_numel].copy_(loaded_weight.view(-1))
                            else:
                                flat_target.copy_(loaded_weight.view(-1)[:flat_target.numel()])

                return True if return_success else None
            except Exception as e:
                import traceback
                print(
                    f"[MOE_PATCH] direct copy FAILED for {weight_name!r} "
                    f"shard_id={shard_id} expert_id={expert_id} "
                    f"lw_shape={tuple(loaded_weight.shape)} "
                    f"param_shape={tuple(param.shape)} "
                    f"expert_data_shape={tuple(param.data[local_expert_id].shape) if local_expert_id is not None else '?'}: {e}",
                    flush=True,
                )
                traceback.print_exc()
                # Fall through to original weight_loader as fallback

        return original_weight_loader(
            param,
            loaded_weight,
            weight_name=weight_name,
            shard_id=shard_id,
            expert_id=expert_id,
            return_success=return_success,
        )

    # Preserve the original method's metadata for debugging.
    wrapper.__name__ = "patched_moe_weight_loader"
    wrapper.__qualname__ = "patched_moe_weight_loader"
    return wrapper


def patch_vllm_moe_model_weight_loader(model):
    # this is a work around to load the weight of vllm fused moe model
    # it is from a bug from vllm 0.8.2
    # all the weights are supposed to have a weight_loader, but the moe weights
    # do not have a weight_loader, so we need to patch it
    # (True, 'model.embed_tokens.weight')
    # (True, 'model.layers.0.self_attn.qkv_proj.weight')
    # (True, 'model.layers.0.self_attn.qkv_proj.bias')
    # (True, 'model.layers.0.self_attn.o_proj.weight')
    # (True, 'model.layers.0.mlp.gate.weight')
    # (True, 'model.layers.0.mlp.shared_expert.gate_up_proj.weight')
    # (True, 'model.layers.0.mlp.shared_expert.down_proj.weight')
    # (False, 'model.layers.0.mlp.shared_expert_gate.weight')   use default
    # (False, 'model.layers.0.input_layernorm.weight')          use default
    # (False, 'model.layers.0.post_attention_layernorm.weight') use default
    # (False, 'model.layers.0.mlp.experts.w13_weight')          use mlp.experts.weight_loader
    # (False, 'model.layers.0.mlp.experts.w2_weight')          use mlp.experts.weight_loader

    # Early return if no MOE models are supported
    if not SUPPORTED_MOE_MODELS:
        return

    original_model_type = type(model)
    if hasattr(model, "runnable") and "ACLGraphWrapper" in str(original_model_type):
        model = model.runnable
        original_model_type = type(model)

    # Define MLP attribute mapping for different model types
    MLP_ATTR_MAPPING = {}
    try:
        from vllm.model_executor.models.mixtral import MixtralForCausalLM

        MLP_ATTR_MAPPING[MixtralForCausalLM] = "block_sparse_moe"
    except ImportError:
        pass

    DEFAULT_MLP_ATTR = "mlp"

    # Get inner model (either model.model or model.language_model)
    inner_model = getattr(model, "model", None) or getattr(model, "language_model", None)
    if inner_model is None:
        raise ValueError("The provided model does not have a valid 'model' or 'language_model' attribute.")

    if not isinstance(model, tuple(SUPPORTED_MOE_MODELS)) and not isinstance(inner_model, tuple(SUPPORTED_MOE_MODELS)):
        return

    # TODO(@leisuzz): class Qwen3MoeLLMForCausalLM is not available if VLLM version < 0.11.0,
    # will update the 'if statement' with 'isinstance' when verl commonly use VLLM version >= 0.11.0
    if type(inner_model).__name__ in ("Qwen3MoeLLMForCausalLM", "Qwen3_5MoeForCausalLM"):
        inner_model = inner_model.model  # Reassign inner_model in Qwen3-vl

    for layer_idx, layer in enumerate(inner_model.layers):
        mlp_attr = MLP_ATTR_MAPPING.get(original_model_type, DEFAULT_MLP_ATTR)

        mlp = getattr(layer, mlp_attr, None)
        if not mlp:
            continue

        experts = getattr(mlp, "experts", None)
        if not experts or not hasattr(experts, "weight_loader"):
            continue

        # Patch the weight loaders with a wrapper that handles per-expert
        # 2-D weights being loaded into fused 3-D parameters (vLLM >= 0.22).
        for name, param in mlp.named_parameters():
            if "w13_weight" in name or "w2_weight" in name:
                param.weight_loader = _make_moe_weight_loader_wrapper(
                    experts.weight_loader, is_w13=("w13_weight" in name)
                )
