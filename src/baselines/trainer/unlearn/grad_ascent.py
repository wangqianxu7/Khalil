"""Gradient Ascent Unlearning

梯度上升方法：在遗忘集上最大化损失
"""
from .base import UnlearnTrainer


class GradAscent(UnlearnTrainer):
    """梯度上升：最大化遗忘集损失
    
    最简单直接的 unlearning 方法：
    通过在遗忘集上最大化损失（梯度上升）来实现遗忘。
    
    注意：这种方法可能会影响模型在其他数据上的性能，
    建议配合保留集使用（使用 GradDiff）。
    """
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """计算梯度上升损失"""
        forget_inputs = inputs.get("forget", inputs)
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        outputs = model(**forget_inputs)
        loss = -outputs.loss  # 负损失 = 梯度上升
        return (loss, outputs) if return_outputs else loss

