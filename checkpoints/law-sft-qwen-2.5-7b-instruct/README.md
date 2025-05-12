---
library_name: transformers
license: other
base_model: /home/wangyc/verl/Qwen/Qwen2.5-7B-Instruct
tags:
- llama-factory
- full
- generated_from_trainer
model-index:
- name: law-sft-qwen-2.5-7b-instruct
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# law-sft-qwen-2.5-7b-instruct

This model is a fine-tuned version of [/home/wangyc/verl/Qwen/Qwen2.5-7B-Instruct](https://huggingface.co//home/wangyc/verl/Qwen/Qwen2.5-7B-Instruct) on the legal_multi_choice dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 1
- eval_batch_size: 8
- seed: 42
- distributed_type: multi-GPU
- num_devices: 8
- gradient_accumulation_steps: 8
- total_train_batch_size: 64
- total_eval_batch_size: 64
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 2.0

### Training results



### Framework versions

- Transformers 4.51.3
- Pytorch 2.6.0+cu124
- Datasets 3.5.0
- Tokenizers 0.21.1
