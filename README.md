# Khalil Unlearning Framework

统一的机器遗忘学习（Machine Unlearning）框架，整合了 open-unlearning、PISTOL、FLAT 等项目的设计思路。

## 项目结构

```
Khalil/
├── src/
│   ├── finetune.py          # 模型微调脚本
│   ├── baselines/           # Unlearning baselines
│   │   ├── main.py          # Unlearning 主入口
│   │   ├── trainer/         # 各种 unlearning 方法
│   │   ├── utils/           # 工具函数
│   │   └── data/            # 数据集类
│   └── evaluate/            # 评估模块
│       ├── main.py          # 评估主入口
│       ├── evaluator.py     # 评估器
│       └── metrics.py       # 评估指标
├── config/
│   ├── model_config.yaml    # 模型配置文件（包含各种模型的配置）
│   ├── finetune.yaml        # Finetune 配置参考
│   └── evaluate.yaml        # Evaluate 配置参考
└── scripts/
    ├── run_finetune.sh      # Finetune 运行脚本
    └── run_evaluate.sh      # Evaluate 运行脚本
```

## 功能模块

### 1. Finetune（模型微调）

用于在特定数据集上微调模型，为 unlearning 做准备。

#### 基本用法

**方式 1：使用模型配置（推荐）**

```bash
python src/finetune.py \
  --model_family llama2-7b-chat \
  --data_path data/train.jsonl \
  --output_dir outputs/finetune \
  --use_lora \
  --lr 1e-4 \
  --batch_size 4 \
  --num_epochs 10
```

使用 `--model_family` 参数，系统会自动从 `config/model_config.yaml` 读取模型配置，包括：
- 模型路径（`hf_key`）
- Flash Attention 设置
- 梯度检查点设置
- 问答模板标签

**方式 2：直接指定模型名称**

```bash
python src/finetune.py \
  --model_name meta-llama/Llama-2-7b-chat-hf \
  --data_path data/train.jsonl \
  --output_dir outputs/finetune \
  --use_lora \
  --use_flash_attention \
  --gradient_checkpointing \
  --lr 1e-4 \
  --batch_size 4 \
  --num_epochs 10
```

#### 主要参数

**模型配置**：
- `--model_family`: 模型家族名称（从 `config/model_config.yaml` 读取配置，推荐使用）
- `--model_name`: 基础模型名称或路径（如果未指定 `model_family` 则必须指定）
- `--use_flash_attention`: 是否使用 flash attention（如果指定了 `model_family` 则从配置读取）
- `--gradient_checkpointing`: 是否启用梯度检查点（如果指定了 `model_family` 则从配置读取）

**LoRA 配置**：
- `--use_lora`: 是否使用 LoRA
- `--lora_r`, `--lora_alpha`, `--lora_dropout`: LoRA 参数

**训练配置**：
- `--data_path`: 训练数据路径（JSON/JSONL）
- `--output_dir`: 输出目录
- `--lr`: 学习率
- `--batch_size`: 批次大小
- `--num_epochs`: 训练轮数
- `--merge_lora`: 训练后是否合并 LoRA

#### 数据格式

支持 JSON/JSONL 格式，每行或每个对象包含：
```json
{"question": "What is...", "answer": "The answer is..."}
```

### 2. Unlearning Baselines

提供多种 unlearning 方法，详见 `src/baselines/README.md`。

### 3. Evaluate（评估）

用于评估 unlearning 的效果，包括遗忘质量和模型实用性。

#### 支持的方法

- **GradAscent**: 梯度上升
- **GradDiff**: 梯度差异
- **KL Divergence**: KL 散度
- **DPO**: Direct Preference Optimization
- **NPO**: Negative Preference Optimization

#### 使用示例

```bash
python src/baselines/main.py \
  --model_name meta-llama/Llama-2-7b-chat-hf \
  --forget_data_path data/forget.jsonl \
  --retain_data_path data/retain.jsonl \
  --method grad_diff \
  --output_dir outputs/unlearn \
  --use_lora
```

## 完整工作流程

### 1. 准备数据

准备训练数据（JSON/JSONL 格式）：
```json
{"question": "...", "answer": "..."}
```

### 2. 微调模型

```bash
python src/finetune.py \
  --model_name meta-llama/Llama-2-7b-chat-hf \
  --data_path data/train.jsonl \
  --output_dir outputs/finetune \
  --use_lora \
  --lr 1e-4 \
  --num_epochs 10
```

### 3. 执行 Unlearning

```bash
python src/baselines/main.py \
  --model_name meta-llama/Llama-2-7b-chat-hf \
  --model_path outputs/finetune \
  --forget_data_path data/forget.jsonl \
  --retain_data_path data/retain.jsonl \
  --method grad_diff \
  --output_dir outputs/unlearn \
  --use_lora
```

### 4. 评估结果

```bash
python src/evaluate/main.py \
  --model_path outputs/unlearn/checkpoint-1000 \
  --forget_data_path data/forget.jsonl \
  --retain_data_path data/retain.jsonl \
  --output_dir outputs/eval_results \
  --eval_forget \
  --eval_utility \
  --eval_probability
```

#### 评估指标

**Forget Quality（遗忘质量）**：
- ROUGE-1/L Recall（越低越好，表示模型无法生成原始答案）
- BLEU（越低越好）
- Perplexity

**Model Utility（模型实用性）**：
- ROUGE-1/L Recall（越高越好，表示模型仍能正确回答保留集问题）
- BLEU（越高越好）
- Perplexity（越低越好）

**Probability（概率）**：
- 模型对遗忘集和保留集的平均概率

## 配置说明

### 模型配置

`config/model_config.yaml` 包含了各种模型的配置信息，包括：
- 模型路径（HuggingFace 名称或本地路径）
- 问答模板标签（用于格式化数据）
- Flash Attention 设置
- 梯度检查点设置

支持的模型包括：
- `llama2-7b-chat`, `llama2-7b`
- `llama3-8b-instruct`, `llama3.2-3b-instruct`
- `gemma-2-2b-it`
- `qwen2.5-7b-instruct`, `qwen2.5-3b-instruct`
- `tofu-llama2-7b`, `tofu-llama3-8b`, `tofu-gemma-2-2b-it`
- `kud-llama2-7b`, `kud-llama3-8b`, `kud-gemma-2-2b-it`
- 以及其他模型

使用 `--model_family` 参数时，系统会自动从配置文件读取这些设置。

### Finetune 配置

配置文件位于 `config/finetune.yaml`，但实际使用命令行参数。配置文件仅作为参考。

### 不使用 Hydra

本框架不使用 Hydra，所有配置通过命令行参数传递，更加灵活和直观。

## 依赖

主要依赖：
- `torch`
- `transformers`
- `peft` (用于 LoRA)
- `datasets` (可选)

## 参考项目

- [open-unlearning](https://github.com/facebookresearch/open-unlearning)
- PISTOL
- FLAT
- LUNAR

## 许可证

请参考各子项目的许可证。

