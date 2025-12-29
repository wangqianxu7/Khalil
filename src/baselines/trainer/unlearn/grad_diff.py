"""Gradient Difference Unlearning

梯度差异方法：遗忘集梯度上升 + 保留集梯度下降
"""
from .base import UnlearnTrainer
from ...utils.loss_utils import compute_kl_divergence


class GradDiff(UnlearnTrainer):
    """梯度差异：遗忘集梯度上升 + 保留集梯度下降
    
    这是最常用的 unlearning 方法之一：
    - 在遗忘集上最大化损失（梯度上升）
    - 在保留集上最小化损失（梯度下降）
    - 通过权重平衡两个目标
    """
    
    def __init__(self, retain_loss_type="NLL", *args, **kwargs):
        """
        Args:
            retain_loss_type: 保留集损失类型
                - "NLL": 负对数似然（标准交叉熵）
                - "KL": KL 散度（相对于参考模型）
        """
        super().__init__(*args, **kwargs)
        self.retain_loss_type = retain_loss_type
        
        # 如果使用 KL，需要准备参考模型
        if retain_loss_type == "KL" and self.ref_model is None:
            if self.oracle_model is not None:
                self.ref_model = self.oracle_model
            else:
                self.ref_model = self._prepare_ref_model(self.model)
    
    def compute_retain_loss(self, model, retain_inputs):
        """计算保留集损失
        
        支持两种损失类型：
        - NLL: 标准负对数似然
        - KL: 相对于参考模型的 KL 散度
        """
        if self.retain_loss_type == "NLL":
            outputs = model(**retain_inputs)
            return outputs.loss
        elif self.retain_loss_type == "KL":
            if self.ref_model is None:
                if self.oracle_model is not None:
                    self.ref_model = self.oracle_model
                else:
                    self.ref_model = self._prepare_ref_model(self.model)
            kl_loss, _ = compute_kl_divergence(
                model=model,
                target_model=self.ref_model,
                inputs=retain_inputs
            )
            return kl_loss
        else:
            raise ValueError(f"Unknown retain_loss_type: {self.retain_loss_type}")
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """计算梯度差异损失"""
        # 遗忘集：梯度上升（最大化损失）
        forget_inputs = inputs.get("forget", {})
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        forget_outputs = model(**forget_inputs)
        forget_loss = -forget_outputs.loss  # 负损失 = 梯度上升
        
        # 保留集：梯度下降（最小化损失）
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
        
        return (total_loss, forget_outputs) if return_outputs else total_loss

