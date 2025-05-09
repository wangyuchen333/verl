#!/bin/bash
set -euo pipefail

# 检查是否有训练进程在运行
check_training_running() {
    pgrep -f "python3 -m verl.trainer.main_ppo" > /dev/null
    return $?
}

# 自动检测空闲GPU的函数
find_free_gpus() {
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | while IFS=, read -r index used total; do
        if [ "$used" -lt "$((total * 20 / 100))" ]; then
            echo -n "$index,"
        fi
    done | sed 's/,$//'
}

# 检查是否有足够的空闲GPU
check_available_gpus() {
    FREE_GPUS=$(find_free_gpus)
    NUM_FREE=$(echo $FREE_GPUS | tr ',' '\n' | wc -l)
    if [ $NUM_FREE -gt 4 ]; then
        return 1
    else
        return 0
    fi
}

# 启动训练的函数
start_training() {
    # ========== 配置部分 ==========
    export TOKENIZERS_PARALLELISM=false
    export WANDB_API_KEY=b040deeeb597f481c025b7c7db998c32d1736333

    # 自动计算 GPU 数量与 TMP size
    N_GPUS=$(nvidia-smi -L | wc -l)
    export N_GPUS
    export TMP_SIZE=$((N_GPUS / 4))
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

    if  [ $NUM_GPUS -eq 4 ]; then
        TENSOR_PARALLEL_SIZE=2
        BATCH_SIZE=4
        echo "检测到4张GPU，调整 tensor_parallel_size=2, batch_size=4"
    elif [ $NUM_GPUS -eq 6 ]; then
        TENSOR_PARALLEL_SIZE=2
        BATCH_SIZE=6
        echo "检测到6张GPU，调整 tensor_parallel_size=2, batch_size=6"
    elif [ $NUM_GPUS -eq 8 ]; then
        TENSOR_PARALLEL_SIZE=2
        BATCH_SIZE=8
        echo "检测到8张GPU，调整 tensor_parallel_size=2, batch_size=8"
    else
        echo "警告：未识别的GPU数量 $NUM_GPUS，使用默认设置"
        TENSOR_PARALLEL_SIZE=2
        BATCH_SIZE=2
    fi

    # 日志配置
    mkdir -p logs
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="logs/train_${TIMESTAMP}.log"

    # ========== 启动训练 ==========
    echo "🚀 Starting training with ${NUM_GPUS} GPUs, TMP size ${TMP_SIZE}"
    echo "📄 Logging to $LOG_FILE"

    nohup python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files=/home/wangyc/verl/data/jec-qa-1-multi-choice/train.parquet \
        data.val_files=/home/wangyc/verl/data/jec-qa-1-multi-choice/test.parquet\
        data.train_batch_size=64 \
        data.val_batch_size=1312 \
        data.max_prompt_length=1024 \
        data.max_response_length=1024 \
        actor_rollout_ref.model.path=/home/wangyc/verl/Qwen/Qwen2.5-7B-Instruct \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=32 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-sum-norm" \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.n=6 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        algorithm.norm_adv_by_std_in_grpo=False \
        trainer.critic_warmup=0 \
        trainer.logger="['console','wandb']" \
        trainer.project_name='Lawyer-Zero' \
        trainer.experiment_name='qwen2.5-7b-grpo-mcq' \
        trainer.n_gpus_per_node=$NUM_GPUS \
        trainer.nnodes=1 \
        trainer.default_local_dir=checkpoints/qwen2.5-7b-grpo-hard-mcq \
        trainer.save_freq=50 \
        trainer.test_freq=10 \
        trainer.total_epochs=2 \
        "$@" > "$LOG_FILE" 2>&1 &
}

last_status=""

while true; do
    if ! check_training_running; then
        FREE_GPUS=$(find_free_gpus)
        NUM_FREE=$(echo $FREE_GPUS | tr ',' '\n' | wc -l)
        # 取小于等于NUM_FREE的最大偶数，且至少为4
        if [ $NUM_FREE -ge 4 ]; then
            if [ $((NUM_FREE % 2)) -eq 1 ]; then
                USE_GPUS=$((NUM_FREE - 1))
            else
                USE_GPUS=$NUM_FREE
            fi
            SELECTED_GPUS=$(echo $FREE_GPUS | tr ',' '\n' | head -n $USE_GPUS | paste -sd "," -)
            export CUDA_VISIBLE_DEVICES=$SELECTED_GPUS
            if [ "$last_status" != "start" ]; then
                echo "找到$NUM_FREE个空闲GPU，使用$USE_GPUS个: $SELECTED_GPUS，开始训练..."
                last_status="start"
            fi
            start_training
            break
        else
            if [ "$last_status" != "wait" ]; then
                echo "空闲GPU不足4张，等待中... (当前时间: $(date))"
                last_status="wait"
            fi
        fi
    else
        if [ "$last_status" != "running" ]; then
            echo "训练正在进行中... (当前时间: $(date))"
            last_status="running"
        fi
    fi
    sleep 1
done