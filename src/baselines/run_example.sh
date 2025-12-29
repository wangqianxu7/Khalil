#!/bin/bash
# Unlearning Baseline 运行示例脚本

# 设置参数
MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
FORGET_DATA="data/forget.jsonl"
RETAIN_DATA="data/retain.jsonl"
OUTPUT_DIR="outputs/grad_diff_example"
METHOD="grad_diff"

# 运行训练
python main.py \
  --model_name "${MODEL_NAME}" \
  --forget_data_path "${FORGET_DATA}" \
  --retain_data_path "${RETAIN_DATA}" \
  --method "${METHOD}" \
  --output_dir "${OUTPUT_DIR}" \
  --use_lora \
  --lora_r 8 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lr 1e-5 \
  --batch_size 4 \
  --gradient_accumulation_steps 2 \
  --num_epochs 3 \
  --forget_weight 1.0 \
  --retain_weight 1.0 \
  --bf16 \
  --seed 42

echo "训练完成！模型保存在: ${OUTPUT_DIR}"

