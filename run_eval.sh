#!/bin/bash

model="/home/wangyc/verl/models/Qwen2.5-7B-Instruct-GRPO"

# 设置环境变量以使用所有 8 个 GPU (0-7)
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" # <-- 修改这里
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python evaluation/eval.py \
    --benchmarks jec-qa-1-multi-choice \
    --model_path $model \
    --max_tokens 4096 \
    --tensor_parallel_size 4 \ # <-- 修改这里
    --system_message r1-lawyer \
    --force_generate
