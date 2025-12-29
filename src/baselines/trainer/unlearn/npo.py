"""NPO (Negative Preference Optimization) Unlearning

NPO 是 DPO 的简化版本，只使用"更差"的输入来降低其概率
"""
from .grad_diff import GradDiff
from ...utils.loss_utils import compute_npo_loss


class NPO(GradDiff):
    """NPO 遗忘方法
    
    NPO 是 DPO 的简化版本，只需要 lose_inputs（要遗忘的数据），
    通过降低这些数据的概率来实现遗忘。
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
        """计算 NPO 损失"""
        # 获取遗忘数据（作为 lose_inputs）
        forget_inputs = inputs.get("forget", {})
        
        # 如果 forget_inputs 有 original/alternate 结构，使用 original
        if isinstance(forget_inputs, dict) and "original" in forget_inputs:
            lose_inputs = {
                "input_ids": forget_inputs["original"]["input_ids"],
                "attention_mask": forget_inputs["original"]["attention_mask"],
                "labels": forget_inputs["original"]["labels"],
            }
        else:
            lose_inputs = {
                "input_ids": forget_inputs["input_ids"],
                "attention_mask": forget_inputs["attention_mask"],
                "labels": forget_inputs["labels"],
            }
        
        # 计算 NPO 损失
        forget_loss, forget_outputs = compute_npo_loss(
            model=model,
            ref_model=self.ref_model,
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
        
        return (total_loss, forget_outputs) if return_outputs else total_loss

