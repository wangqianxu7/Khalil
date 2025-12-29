# Unlearning Baselines

统一的遗忘学习（Machine Unlearning）baseline 框架，整合了 open-unlearning、PISTOL、FLAT 等项目的设计思路。

## 特性

- **统一接口**：所有方法使用相同的 Trainer 基类
- **注册器模式**：易于扩展新方法
- **支持多种方法**：
  - GradAscent: 梯度上升
  - GradDiff: 梯度差异
  - KL Divergence: KL 散度
  - DPO: Direct Preference Optimization
  - NPO: Negative Preference Optimization
- **支持 LoRA**：高效参数微调
- **支持 Oracle/Reference Model**：用于 KL、DPO 等方法
- **灵活的数据格式**：支持 JSON/JSONL

## 文件结构

```
baselines/
├── main.py                 # 主入口
├── trainer/
│   ├── __init__.py
│   ├── registry.py         # 注册器
│   └── unlearn/
│       ├── __init__.py
│       ├── base.py         # 基类
│       ├── grad_ascent.py
│       ├── grad_diff.py
│       ├── kl_divergence.py
│       ├── dpo.py
│       └── npo.py
├── utils/
│   ├── __init__.py
│   ├── loss_utils.py       # 损失函数
│   └── model_utils.py      # 模型工具
└── data/
    ├── __init__.py
    └── dataset.py          # 数据集类
```

## 使用方法

### 基本用法

```bash
python main.py \
  --model_name meta-llama/Llama-2-7b-chat-hf \
  --forget_data_path data/forget.jsonl \
  --retain_data_path data/retain.jsonl \
  --method grad_diff \
  --output_dir outputs/grad_diff \
  --use_lora \
  --lr 1e-5 \
  --batch_size 4 \
  --num_epochs 3
```

### 使用 DPO

```bash
python main.py \
  --model_name meta-llama/Llama-2-7b-chat-hf \
  --forget_data_path data/forget.jsonl \
  --retain_data_path data/retain.jsonl \
  --method dpo \
  --output_dir outputs/dpo \
  --beta 1.0 \
  --use_lora \
  --lr 1e-5
```

### 使用 KL Divergence

```bash
python main.py \
  --model_name meta-llama/Llama-2-7b-chat-hf \
  --forget_data_path data/forget.jsonl \
  --retain_data_path data/retain.jsonl \
  --method kl \
  --output_dir outputs/kl \
  --oracle_model_path path/to/oracle_model \
  --temperature 1.0 \
  --use_lora
```

## 数据格式

### JSONL 格式

每行一个 JSON 对象：

```json
{"question": "What is...", "answer": "The answer is..."}
{"question": "...", "answer": "...", "alternate_answer": "I don't know."}
```

### JSON 格式

包含列表的 JSON：

```json
[
  {"question": "...", "answer": "..."},
  {"question": "...", "answer": "..."}
]
```

## 参数说明

### 模型参数
- `--model_name`: 基础模型名称或路径
- `--model_path`: 已训练模型的路径（可选）
- `--use_lora`: 是否使用 LoRA
- `--lora_r`, `--lora_alpha`, `--lora_dropout`: LoRA 参数

### 数据参数
- `--forget_data_path`: 遗忘集数据路径
- `--retain_data_path`: 保留集数据路径（可选）
- `--max_length`: 最大序列长度

### 训练参数
- `--method`: Unlearning 方法（grad_ascent, grad_diff, kl, dpo, npo）
- `--lr`: 学习率
- `--batch_size`: 批次大小
- `--num_epochs`: 训练轮数

### Unlearning 特定参数
- `--forget_weight`: 遗忘集权重
- `--retain_weight`: 保留集权重
- `--oracle_model_path`: Oracle 模型路径（用于 KL/DPO）
- `--beta`: DPO/NPO 的 beta 参数
- `--temperature`: KL 散度的温度参数

## 扩展新方法

1. 在 `trainer/unlearn/` 下创建新文件
2. 继承 `UnlearnTrainer` 基类
3. 实现 `compute_loss` 方法
4. 在 `trainer/registry.py` 中注册

示例：

```python
from .base import UnlearnTrainer

class MyMethod(UnlearnTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # 实现你的方法
        ...
        return loss
```

然后在 `registry.py` 中注册：

```python
from .unlearn.my_method import MyMethod
register_trainer("my_method")(MyMethod)
```

## 参考

- [open-unlearning](https://github.com/facebookresearch/open-unlearning)
- PISTOL
- FLAT

