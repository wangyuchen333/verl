#!/bin/bash

# --- Configuration ---

# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error when substituting.
set -u
# The return value of a pipeline is the status of the last command to exit with a non-zero status,
# or zero if no command exited with a non-zero status.
set -o pipefail

# Directory containing the sharded model checkpoints (input for fsdp2hf.py)
SHARDED_MODEL_DIR="/home/wangyc/verl/checkpoints/qwen2.5-7b-grpo-hard-mcq/global_step_50/actor"

# Directory for intermediate converted model
INTERMEDIATE_DIR="/home/wangyc/verl/new"

# !!! IMPORTANT: Set this to the directory containing the original base model's
# config.json and tokenizer files (e.g., /home/wangyc/verl/Qwen/Qwen2.5-7B-Instruct)
BASE_MODEL_DIR="/home/wangyc/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" # <--- SET THIS PATH

# --- Environment Setup ---

# Set environment variables for CUDA
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "-------------------------------------"
echo "Starting Model Conversion..."
echo "Sharded Model Input: ${SHARDED_MODEL_DIR}"
echo "Intermediate Output: ${INTERMEDIATE_DIR}"
echo "Base Model Config/Tokenizer Source: ${BASE_MODEL_DIR}"
echo "Using GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "-------------------------------------"

# --- Step 1: Convert FSDP to HF format ---

# Check if conversion is already done

echo "Converting FSDP model to HF format..."
python fsdp2hf.py \
    --local_dir "$SHARDED_MODEL_DIR" \
    --save_dir "$INTERMEDIATE_DIR"
echo "-------------------------------------"
echo "FSDP to HF Conversion Complete."
echo "-------------------------------------"


# --- Step 2: Run Evaluation ---

echo "Starting Evaluation..."
echo "Evaluating Model: ${INTERMEDIATE_DIR}"
echo "Benchmarks: jec-qa-1-multi-choice"
echo "Tensor Parallel Size: 4"
echo "-------------------------------------"

python evaluation/eval.py \
    --benchmarks jec-qa-1-multi-choice \
    --model_path "$INTERMEDIATE_DIR" \
    --max_tokens 4096 \
    --tensor_parallel_size 4 \
    --system_message r1-lawyer \
    --force_generate

echo "-------------------------------------"
echo "Evaluation Complete."
echo "-------------------------------------"

