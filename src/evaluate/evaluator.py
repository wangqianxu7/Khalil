"""评估器

统一的评估接口，支持多种评估指标
"""
import json
import os
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from peft import PeftModel

import numpy as np
from .metrics import (
    compute_rouge,
    compute_bleu,
    compute_perplexity,
    compute_batch_probability,
)


class Evaluator:
    """统一的评估器
    
    支持评估：
    - Forget Quality（遗忘质量）
    - Model Utility（模型实用性）
    """
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        use_lora: bool = False,
        lora_path: Optional[str] = None,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Args:
            model_path: 模型路径
            tokenizer_path: Tokenizer 路径（如果与 model_path 不同）
            use_lora: 是否使用 LoRA
            lora_path: LoRA 适配器路径
            device: 设备
            torch_dtype: 数据类型
        """
        self.device = device
        self.torch_dtype = torch_dtype
        
        # 加载 tokenizer
        tokenizer_path = tokenizer_path or model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map="auto" if device == "cuda" else None,
        )
        
        # 如果使用 LoRA
        if use_lora and lora_path:
            self.model = PeftModel.from_pretrained(self.model, lora_path)
        
        self.model.eval()
    
    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
        batch_size: int = 4,
    ) -> List[str]:
        """生成文本
        
        Args:
            prompts: 提示列表
            max_new_tokens: 最大生成 token 数
            temperature: 温度
            do_sample: 是否采样
            top_p: nucleus sampling 参数
            batch_size: 批次大小
            
        Returns:
            生成的文本列表
        """
        self.model.eval()
        generated_texts = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
                batch_prompts = prompts[i:i + batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(self.device)
                
                # Generate
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
                # Decode
                input_length = inputs["input_ids"].shape[1]
                generated_ids = outputs[:, input_length:]
                generated_batch = self.tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )
                generated_texts.extend(generated_batch)
        
        return generated_texts
    
    def evaluate_forget_quality(
        self,
        forget_data: List[Dict[str, str]],
        batch_size: int = 4,
        max_new_tokens: int = 128,
    ) -> Dict[str, Any]:
        """评估遗忘质量
        
        指标：
        - ROUGE-1/L recall（越低越好）
        - BLEU（越低越好）
        - Perplexity
        
        Args:
            forget_data: 遗忘集数据，每个元素包含 "question" 和 "answer"
            batch_size: 批次大小
            max_new_tokens: 最大生成 token 数
            
        Returns:
            评估结果字典
        """
        # 准备 prompts 和 ground truth
        prompts = [item["question"] for item in forget_data]
        ground_truths = [item["answer"] for item in forget_data]
        
        # 生成
        generated_texts = self.generate(
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
        )
        
        # 计算指标
        rouge_scores = compute_rouge(generated_texts, ground_truths)
        bleu_score = compute_bleu(generated_texts, ground_truths)
        
        # 计算困惑度（在生成的文本上）
        perplexity = compute_perplexity(
            self.model,
            self.tokenizer,
            generated_texts,
            device=self.device
        )
        
        results = {
            "rouge1_recall": rouge_scores["rouge1_recall"],
            "rougeL_recall": rouge_scores["rougeL_recall"],
            "rouge1_f1": rouge_scores["rouge1_f1"],
            "rougeL_f1": rouge_scores["rougeL_f1"],
            "bleu": bleu_score,
            "perplexity": perplexity,
            "num_samples": len(forget_data),
        }
        
        return results
    
    def evaluate_model_utility(
        self,
        retain_data: List[Dict[str, str]],
        batch_size: int = 4,
        max_new_tokens: int = 128,
    ) -> Dict[str, Any]:
        """评估模型实用性
        
        指标：
        - ROUGE-1/L recall（越高越好）
        - BLEU（越高越好）
        - Perplexity（越低越好）
        
        Args:
            retain_data: 保留集数据，每个元素包含 "question" 和 "answer"
            batch_size: 批次大小
            max_new_tokens: 最大生成 token 数
            
        Returns:
            评估结果字典
        """
        # 准备 prompts 和 ground truth
        prompts = [item["question"] for item in retain_data]
        ground_truths = [item["answer"] for item in retain_data]
        
        # 生成
        generated_texts = self.generate(
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
        )
        
        # 计算指标
        rouge_scores = compute_rouge(generated_texts, ground_truths)
        bleu_score = compute_bleu(generated_texts, ground_truths)
        
        # 计算困惑度
        perplexity = compute_perplexity(
            self.model,
            self.tokenizer,
            generated_texts,
            device=self.device
        )
        
        results = {
            "rouge1_recall": rouge_scores["rouge1_recall"],
            "rougeL_recall": rouge_scores["rougeL_recall"],
            "rouge1_f1": rouge_scores["rouge1_f1"],
            "rougeL_f1": rouge_scores["rougeL_f1"],
            "bleu": bleu_score,
            "perplexity": perplexity,
            "num_samples": len(retain_data),
        }
        
        return results
    
    def evaluate_probability(
        self,
        data: List[Dict[str, str]],
        batch_size: int = 4,
    ) -> Dict[str, Any]:
        """评估模型对数据的概率
        
        Args:
            data: 数据列表，每个元素包含 "question" 和 "answer"
            batch_size: 批次大小
            
        Returns:
            包含平均概率的结果字典
        """
        all_probs = []
        
        # 准备数据
        texts = []
        for item in data:
            question = item["question"]
            answer = item["answer"]
            # 格式化文本
            if hasattr(self.tokenizer, 'apply_chat_template'):
                messages = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
            else:
                text = f"Question: {question}\nAnswer: {answer}"
            texts.append(text)
        
        # 批量计算概率
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing probabilities"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs["labels"] = inputs["input_ids"].clone()
            
            # 计算概率
            probs = compute_batch_probability(
                self.model,
                inputs,
                device=self.device
            )
            all_probs.extend(probs)
        
        results = {
            "mean_probability": np.mean(all_probs),
            "std_probability": np.std(all_probs),
            "num_samples": len(data),
        }
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """保存评估结果
        
        Args:
            results: 评估结果
            output_path: 输出路径
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

