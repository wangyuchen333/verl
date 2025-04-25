#!/bin/bash

# --- Configuration ---

# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error when substituting.
set -u
# The return value of a pipeline is the status of the last command to exit with a non-zero status,
# or zero if no command exited with a non-zero status.
set -o pipefail

# Directory containing the sharded model checkpoints (input for hf.py)
SHARDED_MODEL_DIR="/home/wangyc/verl/ck4/global_step_132/actor"

# !!! IMPORTANT: Set this to the directory containing the original base model's
# config.json and tokenizer files (e.g., /home/wangyc/verl/Qwen/Qwen2.5-7B-Instruct)
BASE_MODEL_DIR="/home/wangyc/verl/Qwen/Qwen2.5-7B-Instruct" # <--- SET THIS PATH

# Calculate the output directory where the merged model will be saved by hf.py
# (This logic matches the output path generation in the provided hf.py script)
PARENT_DIR=$(dirname "$SHARDED_MODEL_DIR")
MODEL_IDENTIFIER=$(dirname "$PARENT_DIR")
MERGED_MODEL_PATH="${MODEL_IDENTIFIER}/hf"

# --- Environment Setup ---

# Set environment variables for CUDA
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "-------------------------------------"
echo "Starting Model Merging..."
echo "Sharded Model Input: ${SHARDED_MODEL_DIR}"
echo "Base Model Config/Tokenizer Source: ${BASE_MODEL_DIR}"
echo "Merged Model Output Target: ${MERGED_MODEL_PATH}"
echo "Using GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "-------------------------------------"

# --- Step 1: Merge Distributed Checkpoints ---

# Check if BASE_MODEL_DIR is set
# if [ "$BASE_MODEL_DIR" == "/path/to/your/base/model/files" ]; then
#   echo "Error: Please set the BASE_MODEL_DIR variable in the script."
#   exit 1
# fi

python hf.py \
    --local_dir "$SHARDED_MODEL_DIR" \
    # --model_src_dir "$BASE_MODEL_DIR"

echo "-------------------------------------"
echo "Model Merging Complete."
echo "-------------------------------------"

# --- Step 2: Run Evaluation ---

echo "Starting Evaluation..."
echo "Evaluating Model: ${MERGED_MODEL_PATH}"
echo "Benchmarks: jec-qa-1-multi-choice"
echo "Tensor Parallel Size: 4"
echo "-------------------------------------"

python evaluation/eval.py \
    --benchmarks jec-qa-1-multi-choice \
    --model_path "$MERGED_MODEL_PATH" \
    --max_tokens 4096 \
    --tensor_parallel_size 4 \
    --system_message r1-lawyer \
    --force_generate

echo "-------------------------------------"
echo "Evaluation Complete."
echo "-------------------------------------"

