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

```bash
python src/finetune.py \
  --model_name meta-llama/Llama-2-7b-chat-hf \
  --data_path data/train.jsonl \
  --output_dir outputs/finetune \
  --use_lora \
  --lr 1e-4 \
  --batch_size 4 \
  --num_epochs 10
```

#### 主要参数

- `--model_name`: 基础模型名称或路径
- `--data_path`: 训练数据路径（JSON/JSONL）
- `--output_dir`: 输出目录
- `--use_lora`: 是否使用 LoRA
- `--lora_r`, `--lora_alpha`, `--lora_dropout`: LoRA 参数
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

