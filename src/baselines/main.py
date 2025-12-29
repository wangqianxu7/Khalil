#!/usr/bin/env python
"""Unlearning Baseline 主入口

整合了 open-unlearning, PISTOL, FLAT 的设计思路
"""
import argparse
import os
import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
)
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from trainer.registry import get_trainer, list_trainers
from utils.model_utils import load_model
from data.dataset import UnlearnDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unlearning Baseline - 统一的遗忘学习框架"
    )
    
    # 模型配置
    parser.add_argument("--model_name", type=str, required=True,
                       help="基础模型名称或路径")
    parser.add_argument("--model_path", type=str, default=None,
                       help="已训练模型的路径（可选）")
    parser.add_argument("--use_lora", action="store_true",
                       help="是否使用 LoRA")
    parser.add_argument("--lora_r", type=int, default=8,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                       help="LoRA dropout")
    parser.add_argument("--lora_path", type=str, default=None,
                       help="已有 LoRA 适配器路径")
    parser.add_argument("--use_flash_attention", action="store_true",
                       help="是否使用 flash attention")
    
    # 数据配置
    parser.add_argument("--forget_data_path", type=str, required=True,
                       help="遗忘集数据路径")
    parser.add_argument("--retain_data_path", type=str, default=None,
                       help="保留集数据路径（可选）")
    parser.add_argument("--max_length", type=int, default=512,
                       help="最大序列长度")
    
    # 训练配置
    parser.add_argument("--method", type=str, default="grad_ascent",
                       choices=list_trainers(),
                       help=f"Unlearning 方法，可选: {', '.join(list_trainers())}")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="输出目录")
    parser.add_argument("--lr", type=float, default=1e-5,
                       help="学习率")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="梯度累积步数")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="训练轮数")
    parser.add_argument("--warmup_steps", type=int, default=None,
                       help="预热步数（None 则自动计算）")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="权重衰减")
    parser.add_argument("--bf16", action="store_true",
                       help="使用 bf16")
    parser.add_argument("--fp16", action="store_true",
                       help="使用 fp16")
    
    # Unlearning 特定参数
    parser.add_argument("--forget_weight", type=float, default=1.0,
                       help="遗忘集权重")
    parser.add_argument("--retain_weight", type=float, default=1.0,
                       help="保留集权重")
    parser.add_argument("--oracle_model_path", type=str, default=None,
                       help="Oracle 模型路径（用于 KL/DPO 等）")
    parser.add_argument("--beta", type=float, default=1.0,
                       help="DPO/NPO 的 beta 参数")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="KL 散度的温度参数")
    parser.add_argument("--retain_loss_type", type=str, default="NLL",
                       choices=["NLL", "KL"],
                       help="保留集损失类型")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="日志步数")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="保存步数")
    parser.add_argument("--save_total_limit", type=int, default=2,
                       help="保存的 checkpoint 数量限制")
    parser.add_argument("--eval_steps", type=int, default=None,
                       help="评估步数（None 则不评估）")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # 对于 decoder-only 模型
    
    # 加载模型
    lora_config = None
    if args.use_lora:
        lora_config = {
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
        }
    
    model = load_model(
        model_name=args.model_name,
        model_path=args.model_path,
        use_lora=args.use_lora,
        lora_path=args.lora_path,
        lora_config=lora_config,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32,
        trust_remote_code=True,
        use_flash_attention=args.use_flash_attention,
    )
    
    # 加载 oracle model（如果需要）
    oracle_model = None
    if args.oracle_model_path:
        oracle_model = load_model(
            model_name=args.model_name,
            model_path=args.oracle_model_path,
            use_lora=False,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32,
            trust_remote_code=True,
            use_flash_attention=args.use_flash_attention,
        )
        oracle_model.eval()
        oracle_model.requires_grad_(False)
    
    # 加载数据
    train_dataset = UnlearnDataset(
        forget_path=args.forget_data_path,
        retain_path=args.retain_data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    
    # 计算训练步数
    num_samples = len(train_dataset)
    steps_per_epoch = num_samples // (args.batch_size * args.gradient_accumulation_steps)
    max_steps = steps_per_epoch * args.num_epochs
    
    if args.warmup_steps is None:
        args.warmup_steps = max(1, max_steps // 10)
    
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
        eval_steps=args.eval_steps,
        evaluation_strategy="steps" if args.eval_steps else "no",
        load_best_model_at_end=False,
        report_to="tensorboard" if os.getenv("WANDB_DISABLED") != "true" else "none",
        seed=args.seed,
    )
    
    # 获取 trainer 类
    trainer_cls = get_trainer(args.method)
    
    # 准备 trainer 参数
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "tokenizer": tokenizer,
        "forget_weight": args.forget_weight,
        "retain_weight": args.retain_weight,
    }
    
    # 添加方法特定参数
    if args.method in ["dpo", "npo"]:
        trainer_kwargs["beta"] = args.beta
        trainer_kwargs["ref_model"] = None  # 会在 trainer 内部创建
    if args.method == "kl" or args.method == "kl_divergence":
        trainer_kwargs["temperature"] = args.temperature
    if args.method == "grad_diff":
        trainer_kwargs["retain_loss_type"] = args.retain_loss_type
    
    # 添加 oracle model
    if oracle_model is not None:
        trainer_kwargs["oracle_model"] = oracle_model
    
    # 创建 trainer
    trainer = trainer_cls(**trainer_kwargs)
    
    # 训练
    print(f"开始训练，使用方法: {args.method}")
    print(f"训练样本数: {num_samples}")
    print(f"总步数: {max_steps}")
    print(f"每轮步数: {steps_per_epoch}")
    
    trainer.train()
    
    # 保存
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"模型已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()

