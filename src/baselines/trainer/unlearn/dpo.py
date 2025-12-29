"""DPO (Direct Preference Optimization) Unlearning

使用 DPO 来优化模型，使其偏好"更好"的响应（如拒绝回答）而不是原始遗忘数据
"""
from .grad_diff import GradDiff
from ...utils.loss_utils import compute_dpo_loss


class DPO(GradDiff):
    """DPO 遗忘方法
    
    DPO 通过对比学习：
    - win_inputs: 更好的响应（如拒绝回答、通用回答）
    - lose_inputs: 更差的响应（原始遗忘数据）
    
    让模型学会偏好 win_inputs 而不是 lose_inputs
    """
    
    def __init__(self, beta=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        
        # 准备参考模型
        if self.ref_model is None:
            if self.oracle_model is not None:
                self.ref_model = self.oracle_model
            else:
                self.ref_model = self._prepare_ref_model(self.model)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """计算 DPO 损失"""
        # 获取 forget 数据
        forget_inputs = inputs.get("forget", {})
        
        # DPO 需要 original 和 alternate 两种输入
        # original: 原始遗忘数据（lose）
        # alternate: 替代响应（win，如拒绝回答）
        if isinstance(forget_inputs, dict) and "original" in forget_inputs:
            lose_inputs = {
                "input_ids": forget_inputs["original"]["input_ids"],
                "attention_mask": forget_inputs["original"]["attention_mask"],
                "labels": forget_inputs["original"]["labels"],
            }
            win_inputs = {
                "input_ids": forget_inputs["alternate"]["input_ids"],
                "attention_mask": forget_inputs["alternate"]["attention_mask"],
                "labels": forget_inputs["alternate"]["labels"],
            }
        else:
            # 如果没有 alternate，只使用 original 作为 lose
            lose_inputs = {
                "input_ids": forget_inputs["input_ids"],
                "attention_mask": forget_inputs["attention_mask"],
                "labels": forget_inputs["labels"],
            }
            win_inputs = None
        
        # 计算 DPO 损失
        forget_loss, (win_outputs, lose_outputs) = compute_dpo_loss(
            model=model,
            ref_model=self.ref_model,
            win_inputs=win_inputs,
            lose_inputs=lose_inputs,
            beta=self.beta
        )
        
        # 保留集损失
        total_loss = self.forget_weight * forget_loss
        
        if "retain" in inputs:
            retain_inputs = inputs["retain"]
            retain_inputs = {
                "input_ids": retain_inputs["input_ids"],
                "attention_mask": retain_inputs["attention_mask"],
                "labels": retain_inputs["labels"],
            }
            retain_loss = self.compute_retain_loss(model, retain_inputs)
            total_loss += self.retain_weight * retain_loss
        
        forget_outputs = lose_outputs if lose_outputs is not None else win_outputs
        return (total_loss, forget_outputs) if return_outputs else total_loss

