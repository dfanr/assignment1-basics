# CS336 Spring 2025 Assignment 1: Basics

## 1. 常用链接
- assignment1作业说明pdf
  - [作业1.pdf(英语版)](./cs336_spring2025_assignment1_basics.pdf)
  - [作业1.pdf(双语版)(MinerU翻译版)](./cs336_spring2025_assignment1_basics.no_watermark.zh-CN.dual.pdf)
- 5个assignment对应的git仓库
  - [https://github.com/stanford-cs336/assignment1-basics](https://github.com/stanford-cs336/assignment1-basics)
  - [https://github.com/stanford-cs336/assignment2-systems](https://github.com/stanford-cs336/assignment2-systems)
  - [https://github.com/stanford-cs336/assignment3-scaling](https://github.com/stanford-cs336/assignment3-scaling)
  - [https://github.com/stanford-cs336/assignment4-data](https://github.com/stanford-cs336/assignment4-data)
  - [https://github.com/stanford-cs336/assignment5-alignment](https://github.com/stanford-cs336/assignment5-alignment)

## 2. 作业内容和要求

本节对pdf中的作业具体要求进行了整理。

### 2.1 作业内容

1. BPE tokenizer
2. Transformer模型定义
3. 交叉熵损失和AdamW优化器
4. 训练loop，支持保存和加载模型和优化器状态

### 2.2 流程

1. 在TinyStories数据集上训练一个BPE tokenizer
2. 使用训练好的tokenizer堆数据集进行操作，将其转化为整数id序列
3. 在TinyStories数据集上训练一个Transformer模型
4. 使用训练好的模型生成样本，并评估困惑度
5. 在OpenWebText上训练模型，并将获得的困惑度提交到排行榜

### 2.3 限制

- 期望您从头开始构建组件
- 禁止直接让AI生成解决方案代码
- 允许让AI解释高级概念问题和低级编程问题
- 不能使用：
  - torch.nn
  - torch.nn.functional
  - torch.optim
- 可以使用：
  - torch.nn.Parameter
  - torch.nn.Module
  - torch.nn.ModuleList
  - torch.nn.Sequential
  - torch.optim.Optimizer基类

### 2.4 项目结构

- 在`cs336_basics/`路径下，从头编写实现具体功能的代码
- 简单修改`tests/adapters.py`，以调用上面的具体功能代码，使之通过对应的单元测试
- 运行`tests/test_*.py`，以运行各个部分的单元测试

## 3 作业完成过程备忘

### 3.0 环境配置和准备过程
```bash
# 环境配置
cd path/to/assignment1-basics/

# 假设已经提前安装了uv，指定使用python3.11，以便利用全局的package下载缓存
uv venv --python 3.11
uv sync
source .venv/bin/activate

# 下载和解压数据集
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

### 3.1 BPE tokenizer
#### 3.1.1 Problem (unicode1): Understanding Unicode (1 point)
#### 3.1.2 Problem (unicode2): Unicode Encodings (3 points)
#### 3.1.3 Problem (train_bpe): BPE Tokenizer Training (15 points)
#### 3.1.4 Problem (train_bpe_tinystories): BPE Training on TinyStories (2 points)
#### 3.1.5 Problem (train_bpe_expts_owt): BPE Training on OpenWebText (2 points)
#### 3.1.6 Problem (tokenizer): Implementing the tokenizer (15 points)
#### 3.1.7 Problem (tokenizer_experiments): Experiments with tokenizers (4 points)

### 3.2 Transformer Language Model Architecture
#### 3.2.1 Problem (linear): Implementing the linear module (1 point)

- 测试命令：`uv run pytest -k test_linear`
- 相关代码：
  - cs336_basics/custom_layers/linear.py
  - tests/adapters.py::run_linear()

#### 3.2.2 Problem (embedding): Implement the embedding module (1 point)

- 测试命令： `uv run pytest -k test_embedding`
- 相关代码：
  - cs336_basics/custom_layers/embedding.py
  - tests/adapters.py::run_embedding()

#### 3.2.3 Problem (rmsnorm): Root Mean Square Layer Normalization (1 point)

- 测试命令： `uv run pytest -k test_rmsnorm`
- 相关代码：
  - cs336_basics/custom_layers/rms_norm.py
  - tests/adapters.py::run_rmsnorm()

#### 3.2.4 Problem (positionwise_feedforward): Implement the position-wise feed-forward network (2 points)

- 测试命令： 
  - `uv run pytest -k test_swiglu`
  - `uv run pytest -k test_silu`
- 相关代码：
  - cs336_basics/custom_layers/feed_forward.py
  - cs336_basics/custom_layers/silu.py
  - tests/adapters.py::run_swiglu()
  - tests/adapters.py::run_silu()

#### 3.2.5 Problem (rope): Implement RoPE (2 points)
#### 3.2.6 Problem (softmax): Implement softmax (1 point)
#### 3.2.7 Problem (scaled_dot_product_attention): Implement scaled dot-product attention (5 points)
#### 3.2.8 Problem (multihead_self_attention): Implement causal multi-head self-attention (5 points)
#### 3.2.9 Problem (transformer_block): Implement the Transformer block (3 points)
#### 3.2.10 Problem (transformer_lm): Implementing the Transformer LM (3 points)
#### 3.2.11 Problem (transformer_accounting): Transformer LM resource accounting (5 points)
#### 3.2.12 Problem (cross_entropy): Implement Cross entropy
#### 3.2.13 Problem (learning_rate_tuning): Tuning the learning rate (1 point)
#### 3.2.14 Problem (adamw): Implement AdamW (2 points)
#### 3.2.15 Problem (adamwAccounting): Resource accounting for training with AdamW (2 points)
#### 3.2.16 Problem (learning_rate_schedule): Implement cosine learning rate schedule with warmup
#### 3.2.17 Problem (gradient_clipping): Implement gradient clipping (1 point)


### 3.3 Training loop
#### 3.3.1 Problem (data_loading): Implement data loading (2 points)
#### 3.3.2 Problem (checkpointing): Implement model checkpointing (1 point)
#### 3.3.3 Problem (training_together): Put it together (4 points)


### 3.4 Generating text
#### 3.4.1 Problem (decoding): Decoding (3 points)


### 3.5 Experiments
#### 3.5.1 Problem (experiment_log): Experiment logging (3 points)
#### 3.5.2 Problem (learning_rate): Tune the learning rate (3 points) (4 H100 hrs)
#### 3.5.3 Problem (batch_size_experiment): Batch size variations (1 point) (2 H100 hrs)
#### 3.5.4 Problem (generate): Generate text (1 point)
#### 3.5.5 Problem (layer_norm_ablation): Remove RMSNorm and train (1 point) (1 H100 hr)
#### 3.5.6 Problem (pre_norm_ablation): Implement post-norm and train (1 point) (1 H100 hr)
#### 3.5.7 Problem (no_pos_emb): Implement NoPE (1 point) (1 H100 hr)
#### 3.5.8 Problem (swiglu_ablation): SwiGLU vs. SiLU (1 point) (1 H100 hr)
#### 3.5.9 Problem (main_experiment): Experiment on OWT (2 points) (3 H100 hrs)
#### 3.5.10 Problem (leaderboard): Leaderboard (6 points) (10 H100 hrs)



