"""KL Divergence Unlearning

使用 KL 散度来让模型在遗忘集上的分布远离原始分布
"""
from .base import UnlearnTrainer
from ...utils.loss_utils import compute_kl_divergence


class KLDivergence(UnlearnTrainer):
    """KL 散度遗忘方法
    
    通过最小化模型在遗忘集上的输出与参考模型输出的 KL 散度，
    同时最大化与原始模型的差异来实现遗忘。
    """
    
    def __init__(self, temperature=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        
        # 如果没有提供 ref_model，使用 oracle_model 或创建参考模型
        if self.ref_model is None:
            if self.oracle_model is not None:
                self.ref_model = self.oracle_model
            else:
                self.ref_model = self._prepare_ref_model(self.model)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """计算 KL 散度损失"""
        # 遗忘集：最小化与参考模型的 KL（让模型输出远离原始）
        forget_inputs = inputs.get("forget", {})
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        
        # 计算 KL 散度
        forget_kl_loss, forget_outputs = compute_kl_divergence(
            model=model,
            target_model=self.ref_model,
            inputs=forget_inputs,
            temperature=self.temperature
        )
        
        # 保留集：如果有保留集，保持模型性能
        total_loss = self.forget_weight * forget_kl_loss
        
        if "retain" in inputs:
            retain_inputs = inputs["retain"]
            retain_inputs = {
                "input_ids": retain_inputs["input_ids"],
                "attention_mask": retain_inputs["attention_mask"],
                "labels": retain_inputs["labels"],
            }
            retain_loss = self.compute_retain_loss(model, retain_inputs)
            total_loss += self.retain_weight * retain_loss
        
        return (total_loss, forget_outputs) if return_outputs else total_loss

