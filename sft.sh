set -x

nproc_per_node=8
save_path=/home/wangyc/verl/checkpoints/law-sft-qwen-2.5-7b-instruct-sp2-liger

# Shift the arguments so $@ refers to the rest
shift 2
    # data.prompt_dict_keys=['instruction'] \
    # +data.response_dict_keys=['output'] \

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/home/wangyc/verl/processed_data/train.parquet \
    data.val_files=/home/wangyc/verl/processed_data/test.parquet \
    data.prompt_key=instruction \
    data.response_key=output \
    optim.lr=1e-4 \
    data.micro_batch_size=4 \
    model.partial_pretrain=/home/wangyc/verl/Qwen/Qwen2.5-7B-Instruct \
    model.use_liger=True \
    trainer.default_local_dir=$save_path \
    trainer.project_name=law-sft \
    trainer.experiment_name=law-sft-qwen-2.5-7b-instruct-sp2-liger \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null $@ \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true
