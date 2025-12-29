"""损失函数工具模块

整合了 open-unlearning 中的各种损失计算函数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional


def compute_kl_divergence(
    model: nn.Module,
    target_model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    temperature: float = 1.0
) -> Tuple[torch.Tensor, Any]:
    """计算 KL 散度损失
    
    Args:
        model: 当前模型（学生模型）
        target_model: 目标模型（教师/参考模型）
        inputs: 输入数据
        temperature: 温度参数，用于平滑分布
        
    Returns:
        (kl_loss, outputs): KL 损失和模型输出
    """
    with torch.no_grad():
        ref_outputs = target_model(**inputs)
    
    # 应用温度缩放
    ref_logits = ref_outputs.logits / temperature
    ref_probs = F.log_softmax(ref_logits, dim=-1)
    ref_probs = ref_probs.view(-1, ref_logits.shape[-1])
    
    outputs = model(**inputs)
    current_logits = outputs.logits / temperature
    current_probs = F.log_softmax(current_logits, dim=-1)
    current_probs = current_probs.view(-1, current_logits.shape[-1])
    
    # KL 散度：KL(current || ref)
    kl_loss = F.kl_div(
        current_probs,
        ref_probs,
        reduction="batchmean",
        log_target=True
    )
    
    return kl_loss, outputs


def compute_batch_nll(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, Any]:
    """计算每个序列的负对数似然（NLL）损失
    
    注意：这里返回的是每个序列的总损失，而不是平均损失
    
    Args:
        model: 模型
        inputs: 输入数据，包含 input_ids, attention_mask, labels
        
    Returns:
        (loss, outputs): 每个序列的损失（shape: [batch_size]）和模型输出
    """
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]
    
    # 移位以对齐预测和标签
    shifted_labels = labels[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    
    # 计算每个 token 的损失
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    loss = loss_function(
        shift_logits.view(-1, shift_logits.size(-1)),
        shifted_labels.view(-1)
    )
    
    # 重塑为 [batch_size, seq_len-1]
    loss = loss.view(shifted_labels.shape)
    
    # 对每个序列求和（忽略 padding）
    mask = (shifted_labels != -100).float()
    loss = (loss * mask).sum(dim=-1)
    
    return loss, outputs


def compute_dpo_loss(
    model: nn.Module,
    ref_model: nn.Module,
    win_inputs: Optional[Dict[str, torch.Tensor]] = None,
    lose_inputs: Optional[Dict[str, torch.Tensor]] = None,
    beta: float = 1.0
) -> Tuple[torch.Tensor, Tuple[Any, Any]]:
    """计算 DPO (Direct Preference Optimization) 损失
    
    DPO 通过对比"更好"和"更差"的响应来优化模型
    
    Args:
        model: 当前模型
        ref_model: 参考模型（通常是原始模型）
        win_inputs: "更好"的输入（例如：拒绝回答、通用回答）
        lose_inputs: "更差"的输入（例如：原始遗忘数据）
        beta: DPO 温度参数
        
    Returns:
        (loss, (win_outputs, lose_outputs)): DPO 损失和模型输出
    """
    if win_inputs is None and lose_inputs is None:
        raise ValueError("Both win_inputs and lose_inputs can't be None")
    
    win_log_ratio, lose_log_ratio = 0.0, 0.0
    win_outputs, lose_outputs = None, None
    
    if win_inputs is not None:
        win_loss, win_outputs = compute_batch_nll(model, win_inputs)
        with torch.no_grad():
            win_ref_loss, _ = compute_batch_nll(ref_model, win_inputs)
        # log π_θ(win) - log π_ref(win)
        win_log_ratio = -(win_loss - win_ref_loss)
    
    if lose_inputs is not None:
        lose_loss, lose_outputs = compute_batch_nll(model, lose_inputs)
        with torch.no_grad():
            lose_ref_loss, _ = compute_batch_nll(ref_model, lose_inputs)
        # log π_θ(lose) - log π_ref(lose)
        lose_log_ratio = -(lose_loss - lose_ref_loss)
    
    # DPO loss: -log σ(β * (log_ratio_win - log_ratio_lose))
    # 我们希望 win_log_ratio > lose_log_ratio
    log_diff = win_log_ratio - lose_log_ratio
    loss = -F.logsigmoid(beta * log_diff).mean()
    
    return loss, (win_outputs, lose_outputs)


def compute_npo_loss(
    model: nn.Module,
    ref_model: nn.Module,
    lose_inputs: Dict[str, torch.Tensor],
    beta: float = 1.0
) -> Tuple[torch.Tensor, Any]:
    """计算 NPO (Negative Preference Optimization) 损失
    
    NPO 是 DPO 的简化版本，只使用"更差"的输入（lose_inputs）
    
    Args:
        model: 当前模型
        ref_model: 参考模型
        lose_inputs: "更差"的输入（要遗忘的数据）
        beta: 温度参数
        
    Returns:
        (loss, outputs): NPO 损失和模型输出
    """
    lose_loss, lose_outputs = compute_batch_nll(model, lose_inputs)
    with torch.no_grad():
        lose_ref_loss, _ = compute_batch_nll(ref_model, lose_inputs)
    
    # log π_θ(lose) - log π_ref(lose)
    log_ratio = -(lose_loss - lose_ref_loss)
    
    # NPO loss: -log σ(β * log_ratio)
    # 我们希望降低 lose 的概率，所以 log_ratio 应该为负
    loss = -F.logsigmoid(beta * log_ratio).mean()
    
    return loss, lose_outputs

