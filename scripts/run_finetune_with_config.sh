#!/bin/bash
# 使用模型配置的 Finetune 运行脚本示例

# 设置参数
MODEL_FAMILY="llama2-7b-chat"  # 从 config/model_config.yaml 读取配置
DATA_PATH="data/train.jsonl"
OUTPUT_DIR="outputs/finetune_llama2_7b_config"

# 运行训练（使用模型配置）
python src/finetune.py \
  --model_family "${MODEL_FAMILY}" \
  --data_path "${DATA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --use_lora \
  --lora_r 8 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lr 1e-4 \
  --batch_size 4 \
  --gradient_accumulation_steps 2 \
  --num_epochs 10 \
  --weight_decay 0.01 \
  --bf16 \
  --merge_lora \
  --seed 42

echo "训练完成！模型保存在: ${OUTPUT_DIR}"

