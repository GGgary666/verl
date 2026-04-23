# Self-Distill Quick Start

This document describes how to enable and configure the self-distill auxiliary loss in `verl`.

## What It Does

Self-distill adds a teacher-student alignment term on response tokens during actor update:

- `L_total = L_pg + self_distill_coef * L_sd`
- Token weights are uniform over `response_mask` only.
- Teacher path is fixed to `bf16` behavior (QAT fake quant disabled).

## Config Parameters

All parameters are under actor config (`actor_rollout_ref.actor`):

- `self_distill_enable` (`bool`, default: `false`)
  - Enables/disables self-distill.
- `self_distill_coef` (`float`, default: `0.001`)
  - Coefficient for self-distill loss.
  - Recommended range: `0.001` ~ `0.01`.
- `self_distill_loss_type` (`str`, default: `low_var_kl`)
  - Supported: `abs_logprob`, `mse`, `low_var_kl`.
- `self_distill_teacher_forward_mode` (`str`, default: `per_micro_batch`)
  - Supported: `per_micro_batch`, `per_train_batch`.
  - `per_train_batch` can reuse trainer-precomputed `old_bf16_log_probs`.

## Example Config

Use this block in your actor config:

```yaml
self_distill_enable: true
self_distill_coef: 0.003
self_distill_loss_type: low_var_kl
self_distill_teacher_forward_mode: per_train_batch
```

## Example Override (Hydra Style)

```bash
python -m verl.trainer.main_ppo \
  actor_rollout_ref.actor.self_distill_enable=true \
  actor_rollout_ref.actor.self_distill_coef=0.003 \
  actor_rollout_ref.actor.self_distill_loss_type=low_var_kl \
  actor_rollout_ref.actor.self_distill_teacher_forward_mode=per_train_batch
```

