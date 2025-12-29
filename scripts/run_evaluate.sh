#!/bin/bash
# 评估脚本示例

# 设置参数
MODEL_PATH="outputs/unlearn/checkpoint-1000"
FORGET_DATA="data/forget.jsonl"
RETAIN_DATA="data/retain.jsonl"
OUTPUT_DIR="outputs/eval_results"

# 运行评估
python src/evaluate/main.py \
  --model_path "${MODEL_PATH}" \
  --forget_data_path "${FORGET_DATA}" \
  --retain_data_path "${RETAIN_DATA}" \
  --output_dir "${OUTPUT_DIR}" \
  --eval_forget \
  --eval_utility \
  --eval_probability \
  --batch_size 4 \
  --max_new_tokens 128

echo "评估完成！结果保存在: ${OUTPUT_DIR}/eval_results.json"

