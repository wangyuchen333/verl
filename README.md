<h1 style="text-align: center;">veRL: Volcano Engine Reinforcement Learning for LLM</h1>

veRL is a flexible, efficient and production-ready RL training framework designed for large language models (LLMs). veRL is the open-source version of [HybridFlow](https://arxiv.org/abs/2409.19256v2) paper.

veRL is flexible and easy to use with:

- **Easy extension of diverse RL algorithms**: The Hybrid programming model combines the strengths of single-controller and multi-controller paradigms to enable flexible representation and efficient execution of complex Post-Training dataflows. Allowing users to build RL dataflows in a few lines of code.

- **Seamless integration of existing LLM infra with modular APIs**: Decouples computation and data dependencies, enabling seamless integration with existing LLM frameworks, such as PyTorch FSDP, Megatron-LM and vLLM. Moreover, users can easily extend to other LLM training and inference frameworks.

- **Flexible device mapping**: Supports various placement of models onto different sets of GPUs for efficient resource utilization and scalability across different cluster sizes.

- Readily integration with popular HuggingFace models


veRL is fast with:

- **State-of-the-art throughput**: By seamlessly integrating existing SOTA LLM training and inference frameworks, veRL achieves high generation and training throughput.

- **Efficient actor model resharding with 3D-HybridEngine**: Eliminates memory redundancy and significantly reduces communication overhead during transitions between training and generation phases.


<p align="center">
| <a href="https://verl.readthedocs.io/en/latest/index.html"><b>Documentation</b></a> | <a href="https://arxiv.org/abs/2409.19256v2"><b>Paper</b></a> | <a href="https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA"><b>Slack</b></a> | <a href="https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/WeChat.JPG"><b>Wechat</b></a> | 

<!-- <a href=""><b>Slides</b></a> | -->
</p>

## News

- [2024/12] The team presented <a href="https://neurips.cc/Expo/Conferences/2024/workshop/100677">Post-training LLMs: From Algorithms to Infrastructure</a> at NeurIPS 2024.
  - [Slides](https://github.com/eric-haibin-lin/verl-data/tree/neurips), [notebooks](https://lightning.ai/eric-haibin-lin/studios/verl-neurips~01je0d1benfjb9grmfjxqahvkn?view=public&section=featured), and [video](https://neurips.cc/Expo/Conferences/2024/workshop/100677) available.
- [2024/08] HybridFlow (verl) is accepted to EuroSys 2025.

## Installation Guide

Below are the steps to install veRL in your environment.

### Requirements
- **Python**: Version >= 3.9
- **CUDA**: Version >= 12.1

veRL supports various backends. Currently, the following configurations are available:
- **FSDP** and **Megatron-LM** for training.
- **vLLM** for rollout generation, **SGLang** support coming soon.

**Training backends**

We recommend using **FSDP** backend to investigate, research and prototype different models, datasets and RL algorithms. The guide for using FSDP backend can be found in [PyTorch FSDP Backend](https://verl.readthedocs.io/en/latest/workers/fsdp_workers.html)

For users who pursue better scalability, we recommend using **Megatron-LM** backend. Currently, we support Megatron-LM@core_v0.4.0 with some internal patches (soon be updated to latest version directly relying on upstream Megatron-LM). The guide for using Megatron-LM backend can be found in [Megatron-LM Backend](https://verl.readthedocs.io/en/latest/workers/megatron_workers.html)

### Installation Options

#### 1. From Docker Image

We provide pre-built Docker images for quick setup.

Image and tag: `verlai/verl:vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te1.7-v0.0.3`

1. Launch the desired Docker image:

```bash
docker run --runtime=nvidia -it --rm --shm-size="10g" --cap-add=SYS_ADMIN -v <image:tag> 
```

2.	Inside the container, install veRL:

```bash
# install the nightly version
git clone https://github.com/volcengine/verl && cd verl && pip3 install -e .
# or install from pypi via `pip3 install verl`
```

<details><summary> 3. Setup Megatron (optional) </summary>

If you want to enable training with Megatron, Megatron code must be added to PYTHONPATH:

```bash
cd ..
git clone -b core_v0.4.0 https://github.com/NVIDIA/Megatron-LM.git
cp verl/patches/megatron_v4.patch Megatron-LM/
cd Megatron-LM && git apply megatron_v4.patch
pip3 install -e .
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

You can also get the Megatron code after verl's patch via
```bash
git clone -b core_v0.4.0_verl https://github.com/eric-haibin-lin/Megatron-LM
```
</details>

#### 2. From Custom Environments

<details><summary>If you prefer setting up veRL in your custom environment, expand this section and follow the steps below.</summary>

Using **conda** is recommended for managing dependencies.

1. Create a conda environment:

```bash
conda create -n verl python==3.9
conda activate verl
```

2. Install common dependencies (required for all backends)

```bash
# install torch [or you can skip this step and let vllm to install the correct version for you]
pip3 install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# install vllm
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1
pip3 install ray

# flash attention 2
pip3 install flash-attn --no-build-isolation
```

3. Install veRL

```bash
# install the nightly version
git clone https://github.com/volcengine/verl && cd verl && pip3 install -e .
# or install from pypi via `pip3 install verl`
```

4. Setup Megatron (optional)

```bash
# FOR Megatron-LM Backend
# apex
pip3 install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
         --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" \
         git+https://github.com/NVIDIA/apex

# transformer engine
pip3 install git+https://github.com/NVIDIA/TransformerEngine.git@v1.7

# megatron core v0.4.0
cd ..
git clone -b core_v0.4.0 https://github.com/NVIDIA/Megatron-LM.git
cp verl/patches/megatron_v4.patch Megatron-LM/
cd Megatron-LM && git apply megatron_v4.patch
pip3 install -e .
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

</details>

## Getting Started
Visit our [documentation](https://verl.readthedocs.io/en/latest/index.html) to learn more.

**Quickstart:**
- [Installation](https://verl.readthedocs.io/en/latest/preparation/install.html)
- [Quickstart](https://verl.readthedocs.io/en/latest/start/quickstart.html)

**Running an PPO example step-by-step:**
- Data and Reward Preparation
  - [Prepare Data (Parquet) for Post-Training](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html)
  - [Implement Reward Function for Dataset](https://verl.readthedocs.io/en/latest/preparation/reward_function.html)
- Understanding the PPO Example
  - [PPO Example Architecture](https://verl.readthedocs.io/en/latest/examples/ppo_code_architecture.html)
  - [Config Explanation](https://verl.readthedocs.io/en/latest/examples/config.html)
  - [Run GSM8K Example](https://verl.readthedocs.io/en/latest/examples/gsm8k_example.html)

**Reproducible algorithm baselines:**
- [PPO](https://verl.readthedocs.io/en/latest/experiment/ppo.html)

**For code explanation and advance usage (extension):**
- PPO Trainer and Workers
  - [PPO Ray Trainer](https://verl.readthedocs.io/en/latest/workers/ray_trainer.html)
  - [PyTorch FSDP Backend](https://verl.readthedocs.io/en/latest/workers/fsdp_workers.html)
  - [Megatron-LM Backend](https://verl.readthedocs.io/en/latest/index.html)
- Advance Usage and Extension
  - [Ray API Design Tutorial](https://verl.readthedocs.io/en/latest/advance/placement.html)
  - [Extend to other RL(HF) algorithms](https://verl.readthedocs.io/en/latest/advance/dpo_extension.html)
  - [Add models with the FSDP backend](https://verl.readthedocs.io/en/latest/advance/fsdp_extension.html)
  - [Add models with the Megatron-LM backend](https://verl.readthedocs.io/en/latest/advance/megatron_extension.html)


## Citation

```tex
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2409.19256}
}

@inproceedings{zhang2024framework,
  title={A Framework for Training Large Language Models for Code Generation via Proximal Policy Optimization},
  author={Zhang, Chi and Sheng, Guangming and Liu, Siyao and Li, Jiahao and Feng, Ziyuan and Liu, Zherui and Liu, Xin and Jia, Xiaoying and Peng, Yanghua and Lin, Haibin and Wu, Chuan},
  booktitle={In NL2Code Workshop of ACM KDD},
  year={2024}
}
```

## Publications Using veRL
- [Enhancing Multi-Step Reasoning Abilities of Language Models through Direct Q-Function Optimization](https://arxiv.org/abs/2410.09302)
- [Flaming-hot Initiation with Regular Execution Sampling for Large Language Models](https://arxiv.org/abs/2410.21236)
- [Process Reinforcement Through Implicit Rewards](https://github.com/PRIME-RL/PRIME/)