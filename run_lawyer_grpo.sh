#!/bin/bash
set -euo pipefail

# ========== 配置部分 ==========
# export PYTHONPATH=/home/wangyc/deeplawyer0:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_API_KEY=b040deeeb597f481c025b7c7db998c32d1736333

# 自动计算 GPU 数量与 TMP size
N_GPUS=$(nvidia-smi -L | wc -l)
export N_GPUS
export TMP_SIZE=$((N_GPUS / 4))

# 日志配置
mkdir -p logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/train_${TIMESTAMP}.log"

# ========== 启动训练 ==========
echo "🚀 Starting training with ${N_GPUS} GPUs, TMP size ${TMP_SIZE}"
echo "📄 Logging to $LOG_FILE"

nohup python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=verl/data/jec-qa-1-multi-choice/train.parquet \
    data.val_files=verl/data/jec-qa-1-multi-choice/test.parquet \
    data.train_batch_size=64 \
    data.val_batch_size=1312 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=/home/wangyc/deeplawyer0/models/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    trainer.logger="['console', 'wandb']" \
    trainer.project_name='Lawyer-Zero' \
    trainer.experiment_name='qwen2.5-7b-grpo-mcq' \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.default_local_dir=checkpoints/qwen2.5-7b-grpo-hard-mcq \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=2 \
    "$@" > "$LOG_FILE" 2>&1 &
