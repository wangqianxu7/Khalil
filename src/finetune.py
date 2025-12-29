#!/usr/bin/env python
"""Finetune 主入口

仿照 FLAT、PISTOL 的设计思路，不使用 hydra
"""
import argparse
import os
import json
import random
import numpy as np
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, PeftModel

# 导入 baselines 中的工具函数
from baselines.utils.model_utils import find_all_linear_names, setup_lora, load_model
from baselines.data.dataset import UnlearnDataset


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune - 模型微调脚本"
    )
    
    # 模型配置
    parser.add_argument("--model_name", type=str, required=True,
                       help="基础模型名称或路径")
    parser.add_argument("--use_lora", action="store_true",
                       help="是否使用 LoRA")
    parser.add_argument("--lora_r", type=int, default=8,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                       help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=None,
                       help="LoRA target modules（None 则自动查找）")
    parser.add_argument("--use_flash_attention", action="store_true",
                       help="是否使用 flash attention")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="是否启用梯度检查点")
    
    # 数据配置
    parser.add_argument("--data_path", type=str, required=True,
                       help="训练数据路径（JSON/JSONL）")
    parser.add_argument("--max_length", type=int, default=512,
                       help="最大序列长度")
    parser.add_argument("--question_key", type=str, default="question",
                       help="问题字段名")
    parser.add_argument("--answer_key", type=str, default="answer",
                       help="答案字段名")
    
    # 训练配置
    parser.add_argument("--output_dir", type=str, required=True,
                       help="输出目录")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="学习率")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="梯度累积步数")
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="训练轮数")
    parser.add_argument("--warmup_steps", type=int, default=None,
                       help="预热步数（None 则自动计算）")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="权重衰减")
    parser.add_argument("--bf16", action="store_true",
                       help="使用 bf16")
    parser.add_argument("--fp16", action="store_true",
                       help="使用 fp16")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--logging_steps", type=int, default=50,
                       help="日志步数")
    parser.add_argument("--save_steps", type=int, default=None,
                       help="保存步数（None 则每轮保存一次）")
    parser.add_argument("--save_total_limit", type=int, default=2,
                       help="保存的 checkpoint 数量限制")
    parser.add_argument("--merge_lora", action="store_true",
                       help="训练后是否合并 LoRA（保存完整模型）")
    parser.add_argument("--eval_strategy", type=str, default="no",
                       choices=["no", "steps", "epoch"],
                       help="评估策略")
    parser.add_argument("--eval_steps", type=int, default=None,
                       help="评估步数（如果 eval_strategy=steps）")
    
    return parser.parse_args()


class FinetuneDataset:
    """Finetune 数据集类
    
    支持 JSON/JSONL 格式，用于模型微调
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        question_key: str = "question",
        answer_key: str = "answer",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.question_key = question_key
        self.answer_key = answer_key
        
        # 加载数据
        self.data = self._load_data(data_path)
    
    def _load_data(self, path: str):
        """加载数据文件"""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            if path.endswith('.jsonl'):
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            else:  # JSON 格式
                data = json.load(f)
                if isinstance(data, dict):
                    # 如果是字典，尝试找到数据列表
                    for key in ['data', 'examples', 'samples', 'train']:
                        if key in data:
                            data = data[key]
                            break
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """获取一个样本"""
        item = self.data[idx]
        question = item.get(self.question_key, "")
        answer = item.get(self.answer_key, "")
        
        # 格式化文本
        text = self._format_text(question, answer)
        
        # Tokenize
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "labels": encodings["input_ids"].squeeze(0),
        }
    
    def _format_text(self, question: str, answer: str) -> str:
        """格式化文本"""
        # 对于 chat 模型，使用 chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            # 简单格式
            return f"Question: {question}\nAnswer: {answer}"


def main():
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    config_dict = vars(args)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if args.use_flash_attention else None,
    )
    
    # 启用梯度检查点
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # 应用 LoRA
    if args.use_lora:
        if args.lora_target_modules is None:
            # 自动查找
            target_modules = find_all_linear_names(model)
        else:
            target_modules = args.lora_target_modules
        
        model = setup_lora(
            model,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
        )
    
    # 加载数据
    train_dataset = FinetuneDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        question_key=args.question_key,
        answer_key=args.answer_key,
    )
    
    # 计算训练步数
    num_samples = len(train_dataset)
    steps_per_epoch = num_samples // (args.batch_size * args.gradient_accumulation_steps)
    max_steps = steps_per_epoch * args.num_epochs
    
    if args.warmup_steps is None:
        args.warmup_steps = max(1, max_steps // 10)
    
    if args.save_steps is None:
        args.save_steps = steps_per_epoch  # 每轮保存一次
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        max_steps=max_steps,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        bf16=args.bf16,
        fp16=args.fp16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps if args.eval_strategy == "steps" else None,
        load_best_model_at_end=False,
        report_to="tensorboard" if os.getenv("WANDB_DISABLED") != "true" else "none",
        seed=args.seed,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 因果语言模型，不是 MLM
    )
    
    # 创建 trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # 训练
    print(f"开始训练")
    print(f"模型: {args.model_name}")
    print(f"数据: {args.data_path} ({num_samples} 样本)")
    print(f"使用 LoRA: {args.use_lora}")
    if args.use_lora:
        print(f"LoRA r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"总步数: {max_steps}, 每轮步数: {steps_per_epoch}")
    print(f"输出目录: {args.output_dir}")
    
    model.config.use_cache = False
    trainer.train()
    
    # 保存模型
    if args.use_lora and args.merge_lora:
        # 合并 LoRA 并保存完整模型
        model = model.merge_and_unload()
        print(f"LoRA 已合并")
        model.save_pretrained(args.output_dir)
    else:
        # 只保存 LoRA 适配器或完整模型
        trainer.save_model(args.output_dir)
    
    tokenizer.save_pretrained(args.output_dir)
    print(f"模型已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()

