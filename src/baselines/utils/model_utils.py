"""模型工具函数"""
import torch
from typing import List, Optional
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel


def find_all_linear_names(model) -> List[str]:
    """找到所有线性层名称，用于 LoRA target_modules
    
    Args:
        model: 模型
        
    Returns:
        线性层名称列表
    """
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    
    # 移除 lm_head（通常不需要 LoRA）
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    
    return list(lora_module_names)


def setup_lora(
    model,
    r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    task_type: str = "CAUSAL_LM"
) -> torch.nn.Module:
    """设置 LoRA 适配器
    
    Args:
        model: 基础模型
        r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        target_modules: 目标模块列表，如果为 None 则自动查找
        task_type: 任务类型
        
    Returns:
        应用了 LoRA 的模型
    """
    if target_modules is None:
        target_modules = find_all_linear_names(model)
    
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=task_type
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_model(
    model_name: str,
    model_path: Optional[str] = None,
    use_lora: bool = False,
    lora_path: Optional[str] = None,
    lora_config: Optional[dict] = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
    use_flash_attention: bool = False,
) -> torch.nn.Module:
    """加载模型（支持 LoRA）
    
    Args:
        model_name: 模型名称或路径
        model_path: 已训练模型的路径（可选）
        use_lora: 是否使用 LoRA
        lora_path: LoRA 适配器路径（如果要从已有 LoRA 加载）
        lora_config: LoRA 配置字典
        torch_dtype: 数据类型
        trust_remote_code: 是否信任远程代码
        use_flash_attention: 是否使用 flash attention
        
    Returns:
        加载的模型
    """
    # 加载基础模型
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
    }
    
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    if model_path:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    # 如果要从 LoRA 加载
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
        if not use_lora:  # 如果要合并 LoRA
            model = model.merge_and_unload()
    
    # 如果要用新的 LoRA
    if use_lora and not lora_path:
        if lora_config is None:
            lora_config = {
                "r": 8,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
            }
        model = setup_lora(model, **lora_config)
    
    return model

