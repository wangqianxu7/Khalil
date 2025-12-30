#!/bin/bash
# Finetune 运行脚本示例

# 设置参数
MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
DATA_PATH="data/train.jsonl"
OUTPUT_DIR="outputs/finetune_llama2_7b"

# 运行训练
python src/finetune.py \
  --model_name "${MODEL_NAME}" \
  --data_path "${DATA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --use_lora \
  --lora_r 8 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lr 1e-4 \
  --batch_size 8 \
  --gradient_accumulation_steps 2 \
  --num_epochs 10 \
  --weight_decay 0 \
  --bf16 \
  --gradient_checkpointing \
  --merge_lora \
  --seed 42

echo "训练完成！模型保存在: ${OUTPUT_DIR}"

