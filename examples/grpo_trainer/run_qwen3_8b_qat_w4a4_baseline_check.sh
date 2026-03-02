set -x

# 解析当前脚本的绝对路径，方便后面拷贝
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

export VLLM_LOGGING_LEVEL=WARNING
export VERL_LOGGING_LEVEL=INFO
export VLLM_CONFIGURE_LOGGING=0

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCHDYNAMO_VERBOSE=1
export TORCH_COMPILE_DISABLE=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

ROOT_DIR=/workspace/gg/logs/
EXP_NAME=qwen3_8B_qat_w4a4_baseline_check

# QAT Configuration - w4a4 mode (ref: recipe/qat/config/nvfp4_w4a4.json)
VERL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
qat_config_path="${qat_config_path:-"${VERL_ROOT}/recipe/qat/config/nvfp4_w4a4.json"}"
export TENSORBOARD_DIR=/root/data/gg/tmp_logs/$EXP_NAME/tensorboard
mkdir -p $ROOT_DIR
mkdir -p $ROOT_DIR/$EXP_NAME
mkdir -p $TENSORBOARD_DIR

# 将当前训练脚本和运行命令记录到 tensorboard 目录，便于之后溯源
cp "$SCRIPT_PATH" "$TENSORBOARD_DIR/$(basename "$SCRIPT_PATH")"
{
  echo "Run time: $(date)"
  echo "Script : $SCRIPT_PATH"
  echo "Command: $SCRIPT_PATH $*"
} > "$TENSORBOARD_DIR/run_command.txt"


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    data.train_files=/xpfs/fp4/dpj/data/gsm8k/train.parquet \
    data.val_files=/xpfs/fp4/dpj/data/gsm8k/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/xpfs/fp4/dpj/la_models/Qwen3-8B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.qat.enable=True \
    actor_rollout_ref.actor.qat.mode=w4a4 \
    actor_rollout_ref.actor.qat.quantization_config_path="${qat_config_path}" \
    'actor_rollout_ref.actor.qat.ignore_patterns=["lm_head", "embed_tokens", "re:.*mlp.gate$"]' \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=$EXP_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.default_local_dir=$ROOT_DIR/$EXP_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=0 \
    trainer.test_freq=10 \
    trainer.log_val_generations=10 \
    trainer.total_epochs=15 $@ | tee -a $ROOT_DIR/$EXP_NAME.log