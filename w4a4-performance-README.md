# W4A4 QAT Performance Quick Start

W4A4 训练与 rollout 同步的性能开关。QAT 基础用法见 [`recipe/qat/README.md`](recipe/qat/README.md)，设计与 benchmark 见 [`recipe/qat/W4A4_PERFORMANCE.md`](recipe/qat/W4A4_PERFORMANCE.md)。

Micro-bench（1.7B，interval=8 + fuse）：相对 upstream-like W4A4 约 **1.37×**。

配置路径：`actor_rollout_ref.actor.qat.*`（默认见 `verl/trainer/config/actor/dp_actor.yaml`）。

## 参数说明

| 参数 | 默认值 | 通俗解释 |
|------|--------|----------|
| `activation_observer_update_interval` | `8` | **隔几步更新一次激活 scale**。设为 1 表示每步都更新（最准但最慢）；8 表示每 8 个 optimizer step 更新一次，是主要加速项。 |
| `activation_observer_sync_interval` | `8` | **隔几步在多卡之间同步 scale**。多卡训练时，各卡各自观测到的最大值需要对齐；间隔越大，通信越少。 |
| `activation_observer_freeze_after_steps` | `-1` | 训练到第 N 步后**不再更新**激活 scale。`-1` 表示一直更新；设成具体数字可在 warmup 后省掉 observer 开销。 |
| `activation_observer` | `static_minmax` | scale 怎么统计：`static_minmax` 取历史最大值（常用）；`memoryless_minmax` 只看当前 batch；`minmax` 用 EMA 平滑。 |
| `fuse_w4a4_rms_norm_activation` | `true` | **把 RMSNorm 和激活 fake-quant 合成一步**。Norm 之后 quant 一次，q/k/v、gate/up 不再各自 quant，减少重复计算。 |
| `fake_quant_kernel_impl` | `nvfp4` | fake-quant 用哪套算子：`nvfp4` 对齐 NVFP4 语义（B200 等推荐）；`legacy` 旧 Triton；`torchao_real` 实验路径。 |
| `train_fake_quant_enable` | `true` | 训练 forward 是否走 fake-quant。`false` 时 actor 用 BF16 算，但仍保留 QAT 模块供 rollout 导出。 |
| `observe_w4a4_input_scale_in_bf16_forward` | `true` | 上项为 `false` 时，BF16 forward 里是否**仍更新**激活 scale observer。 |
| `rollout.use_shm` | 未设 | 权重同步到 vLLM 时，量化结果放哪：`false` → 留在 **GPU**（NVIDIA IPC 推荐，少一次 CPU 拷贝）；否则默认走 CPU/共享内存。 |
| `enable_batched_amax_sync` | worker 自动 | 无 yaml 配置项。W4A4 多卡训练时，worker 自动把各层 amax 的 `all_reduce` 合并成一批。 |


## 怎么开（训练脚本）

**省事做法：** 直接用 [`dapo_qat_trainer.yaml`](recipe/qat/config/dapo_qat_trainer.yaml)，里面已是生产默认（interval=8、fuse=true、nvfp4）：

```bash
MODEL_PATH=/path/to/model bash recipe/qat/run_qwen3_30b_w4a4.sh
```

**手动写 Hydra override**（贴进任意 launch 脚本）：

```bash
python3 -m recipe.dapo.main_dapo \   # 或 verl.trainer.main_ppo
  --config-path recipe/qat/config \
  --config-name dapo_qat_trainer \
  actor_rollout_ref.actor.qat.activation_observer_update_interval=8 \
  actor_rollout_ref.actor.qat.activation_observer_sync_interval=8 \
  actor_rollout_ref.actor.qat.fuse_w4a4_rms_norm_activation=true \
  actor_rollout_ref.actor.qat.fake_quant_kernel_impl=nvfp4 \
  actor_rollout_ref.rollout.use_shm=false
```

**对比 baseline（消融用）：** interval/sync 改 **1**，fuse 改 **false**，kernel 改 **legacy**。

```bash
MODEL_PATH=/path/to/checkpoint bash exp_scripts/profile_b200/test_qat_opt_vs_upstream_like.sh
```

## 验证

```bash
python scripts/profile_qat_train_step.py --stack hf --model-path "${MODEL_PATH}" \
  --mode w4a4 --hf-enable-qat --hf-fuse-w4a4-rms-norm-activation \
  --fake-quant-kernel-impl nvfp4 \
  --activation-observer-update-interval 8 --activation-observer-sync-interval 8 \
  --batch-size 2 --seq-len 1024 --steps 24 --warmup 4
```
