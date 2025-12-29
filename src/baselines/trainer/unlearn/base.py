import torch
import copy
from transformers import Trainer
from typing import Dict, Any, Optional


class UnlearnTrainer(Trainer):
    """统一的 Unlearning Trainer 基类
    
    整合了 open-unlearning, PISTOL, FLAT 的设计思路：
    - 支持 oracle/reference model
    - 支持 forget/retain 权重平衡
    - 统一的接口便于扩展新方法
    """
    
    def __init__(
        self,
        oracle_model=None,
        ref_model=None,
        forget_weight=1.0,
        retain_weight=1.0,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.oracle_model = oracle_model
        self.ref_model = ref_model
        self.forget_weight = forget_weight
        self.retain_weight = retain_weight
        
        # 准备 oracle/ref model
        if self.oracle_model is not None:
            self.oracle_model.eval()
            self.oracle_model.requires_grad_(False)
            if hasattr(self, 'accelerator'):
                self.oracle_model = self.accelerator.prepare_model(
                    self.oracle_model, evaluation_mode=True
                )
        
        if self.ref_model is not None:
            self.ref_model.eval()
            self.ref_model.requires_grad_(False)
            if hasattr(self, 'accelerator'):
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True
                )
    
    def _prepare_ref_model(self, model):
        """准备参考模型（用于 KL/DPO 等）
        
        Args:
            model: 要复制的模型
            
        Returns:
            深拷贝的参考模型，设置为 eval 模式
        """
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        ref_model.requires_grad_(False)
        
        # 移动到相同设备
        if hasattr(model, 'device'):
            ref_model = ref_model.to(model.device)
        
        if hasattr(self, 'accelerator'):
            ref_model = self.accelerator.prepare_model(
                ref_model, evaluation_mode=True
            )
        return ref_model
    
    def compute_retain_loss(self, model, retain_inputs):
        """计算保留集损失（子类可重写）
        
        Args:
            model: 当前模型
            retain_inputs: 保留集输入
            
        Returns:
            保留集损失值
        """
        outputs = model(**retain_inputs)
        return outputs.loss
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """计算损失（子类必须实现）
        
        Args:
            model: 当前模型
            inputs: 输入数据，通常包含 "forget" 和 "retain" 键
            return_outputs: 是否返回模型输出
            
        Returns:
            损失值，如果 return_outputs=True 则返回 (loss, outputs)
        """
        raise NotImplementedError("Subclass must implement compute_loss")

