#!/usr/bin/env python
"""评估主入口

用于评估 unlearning 的效果
"""
import argparse
import json
from pathlib import Path
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluate.evaluator import Evaluator


def load_data(data_path: str) -> list:
    """加载数据文件"""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        if data_path.endswith('.jsonl'):
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="评估 Unlearning 效果"
    )
    
    # 模型配置
    parser.add_argument("--model_path", type=str, required=True,
                       help="模型路径")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                       help="Tokenizer 路径（如果与 model_path 不同）")
    parser.add_argument("--use_lora", action="store_true",
                       help="是否使用 LoRA")
    parser.add_argument("--lora_path", type=str, default=None,
                       help="LoRA 适配器路径")
    
    # 数据配置
    parser.add_argument("--forget_data_path", type=str, default=None,
                       help="遗忘集数据路径")
    parser.add_argument("--retain_data_path", type=str, default=None,
                       help="保留集数据路径")
    
    # 评估配置
    parser.add_argument("--output_dir", type=str, required=True,
                       help="输出目录")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="批次大小")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                       help="最大生成 token 数")
    parser.add_argument("--device", type=str, default="cuda",
                       help="设备")
    parser.add_argument("--eval_forget", action="store_true",
                       help="评估遗忘质量")
    parser.add_argument("--eval_utility", action="store_true",
                       help="评估模型实用性")
    parser.add_argument("--eval_probability", action="store_true",
                       help="评估概率")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 创建评估器
    evaluator = Evaluator(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        use_lora=args.use_lora,
        lora_path=args.lora_path,
        device=args.device,
    )
    
    results = {}
    
    # 评估遗忘质量
    if args.eval_forget and args.forget_data_path:
        print("评估遗忘质量...")
        forget_data = load_data(args.forget_data_path)
        forget_results = evaluator.evaluate_forget_quality(
            forget_data=forget_data,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
        results["forget_quality"] = forget_results
        print(f"Forget Quality - ROUGE-L Recall: {forget_results['rougeL_recall']:.4f}")
        print(f"Forget Quality - BLEU: {forget_results['bleu']:.4f}")
    
    # 评估模型实用性
    if args.eval_utility and args.retain_data_path:
        print("评估模型实用性...")
        retain_data = load_data(args.retain_data_path)
        utility_results = evaluator.evaluate_model_utility(
            retain_data=retain_data,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
        results["model_utility"] = utility_results
        print(f"Model Utility - ROUGE-L Recall: {utility_results['rougeL_recall']:.4f}")
        print(f"Model Utility - BLEU: {utility_results['bleu']:.4f}")
    
    # 评估概率
    if args.eval_probability:
        if args.forget_data_path:
            print("评估遗忘集概率...")
            forget_data = load_data(args.forget_data_path)
            forget_prob = evaluator.evaluate_probability(
                data=forget_data,
                batch_size=args.batch_size,
            )
            results["forget_probability"] = forget_prob
            print(f"Forget Mean Probability: {forget_prob['mean_probability']:.4f}")
        
        if args.retain_data_path:
            print("评估保留集概率...")
            retain_data = load_data(args.retain_data_path)
            retain_prob = evaluator.evaluate_probability(
                data=retain_data,
                batch_size=args.batch_size,
            )
            results["retain_probability"] = retain_prob
            print(f"Retain Mean Probability: {retain_prob['mean_probability']:.4f}")
    
    # 保存结果
    output_path = Path(args.output_dir) / "eval_results.json"
    evaluator.save_results(results, str(output_path))
    print(f"\n评估结果已保存到: {output_path}")


if __name__ == "__main__":
    main()

