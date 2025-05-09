python fsdp2hf.py \
    --local_dir "/home/wangyc/verl/checkpoints/qwen2.5-7b-grpo-hard-mcq/global_step_132/actor" \
    --save_dir "/home/wangyc/verl/no_format"

python evaluation/eval.py \
    --benchmarks jec-qa-1-multi-choice \
    --model_path "/home/wangyc/verl/no_format" \
    --max_tokens 4096 \
    --tensor_parallel_size 4 \
    --system_message r1-lawyer \
    --force_generate