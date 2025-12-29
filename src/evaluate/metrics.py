"""评估指标计算

整合了 FLAT、LUNAR、open-unlearning 中的评估指标
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional
from rouge_score import rouge_scorer
import sacrebleu
from transformers import PreTrainedModel, PreTrainedTokenizer


def compute_rouge(
    generated_texts: List[str],
    reference_texts: List[str],
    rouge_types: List[str] = ["rouge1", "rougeL"]
) -> Dict[str, float]:
    """计算 ROUGE 分数
    
    Args:
        generated_texts: 生成的文本列表
        reference_texts: 参考文本列表
        rouge_types: ROUGE 类型列表
        
    Returns:
        包含各种 ROUGE 分数的字典
    """
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    rouge_scores = {f"{rt}_recall": [] for rt in rouge_types}
    rouge_scores.update({f"{rt}_f1": [] for rt in rouge_types})
    
    for gen, ref in zip(generated_texts, reference_texts):
        scores = scorer.score(ref, gen)
        for rt in rouge_types:
            rouge_scores[f"{rt}_recall"].append(scores[rt].recall)
            rouge_scores[f"{rt}_f1"].append(scores[rt].fmeasure)
    
    # 计算平均值
    result = {}
    for key, values in rouge_scores.items():
        result[key] = np.mean(values) if values else 0.0
    
    return result


def compute_bleu(
    generated_texts: List[str],
    reference_texts: List[str]
) -> float:
    """计算 BLEU 分数
    
    Args:
        generated_texts: 生成的文本列表
        reference_texts: 参考文本列表
        
    Returns:
        BLEU 分数
    """
    bleu = sacrebleu.corpus_bleu(generated_texts, [reference_texts])
    return bleu.score / 100.0  # 转换为 0-1 范围


def compute_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    device: str = "cuda",
    max_length: int = 512
) -> float:
    """计算困惑度（Perplexity）
    
    Args:
        model: 模型
        tokenizer: Tokenizer
        texts: 文本列表
        device: 设备
        max_length: 最大长度
        
    Returns:
        平均困惑度
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            ).to(device)
            
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            # 计算有效 token 数
            num_tokens = (inputs["input_ids"] != tokenizer.pad_token_id).sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity


def compute_probability(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    device: str = "cuda"
) -> float:
    """计算模型对标签序列的概率
    
    Args:
        model: 模型
        tokenizer: Tokenizer
        input_ids: 输入 ID
        attention_mask: 注意力掩码
        labels: 标签
        device: 设备
        
    Returns:
        平均概率（exp(-loss)）
    """
    model.eval()
    
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
    
    # 概率 = exp(-loss)
    prob = torch.exp(-loss).item()
    return prob


def compute_truth_ratio(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    correct_inputs: Dict[str, torch.Tensor],
    wrong_inputs: Dict[str, torch.Tensor],
    device: str = "cuda"
) -> float:
    """计算 Truth Ratio
    
    Truth Ratio = P(wrong) / P(correct)
    对于遗忘集：越接近 1 越好（表示模型无法区分正确和错误答案）
    对于保留集：越小越好（表示模型偏好正确答案）
    
    Args:
        model: 模型
        tokenizer: Tokenizer
        correct_inputs: 正确答案的输入
        wrong_inputs: 错误答案的输入
        device: 设备
        
    Returns:
        Truth Ratio
    """
    model.eval()
    
    # 计算正确答案的概率
    correct_prob = compute_probability(
        model, tokenizer,
        correct_inputs["input_ids"],
        correct_inputs["attention_mask"],
        correct_inputs["labels"],
        device
    )
    
    # 计算错误答案的概率
    wrong_prob = compute_probability(
        model, tokenizer,
        wrong_inputs["input_ids"],
        wrong_inputs["attention_mask"],
        wrong_inputs["labels"],
        device
    )
    
    # Truth Ratio = wrong_prob / correct_prob
    truth_ratio = wrong_prob / (correct_prob + 1e-10)
    return truth_ratio


def compute_batch_probability(
    model: PreTrainedModel,
    inputs: Dict[str, torch.Tensor],
    device: str = "cuda"
) -> List[float]:
    """批量计算概率
    
    Args:
        model: 模型
        inputs: 输入字典，包含 input_ids, attention_mask, labels
        device: 设备
        
    Returns:
        每个样本的概率列表
    """
    model.eval()
    
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    labels = inputs["labels"].to(device)
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # 计算每个 token 的损失
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        losses = losses.view(shift_labels.shape)
        
        # 对每个序列求平均损失
        mask = (shift_labels != -100).float()
        seq_losses = (losses * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-10)
        
        # 转换为概率
        probs = torch.exp(-seq_losses).cpu().tolist()
    
    return probs

