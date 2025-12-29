"""配置工具函数

用于读取和管理模型配置
"""
import yaml
from pathlib import Path
from typing import Dict, Optional


def load_model_config(
    model_family: str,
    config_path: str = "config/model_config.yaml"
) -> Dict:
    """从 YAML 文件加载模型配置
    
    Args:
        model_family: 模型家族名称（如 "llama2-7b-chat"）
        config_path: 配置文件路径
        
    Returns:
        模型配置字典，包含：
        - hf_key: HuggingFace 模型名称或路径
        - question_start_tag: 问题开始标签
        - question_end_tag: 问题结束标签
        - answer_tag: 答案标签
        - flash_attention2: 是否使用 flash attention
        - gradient_checkpointing: 是否启用梯度检查点
        
    Raises:
        FileNotFoundError: 配置文件不存在
        KeyError: 模型家族不存在
    """
    config_path = Path(config_path)
    if not config_path.exists():
        # 尝试相对路径
        config_path = Path(__file__).parent.parent.parent / config_path
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        configs = yaml.safe_load(f)
    
    if model_family not in configs:
        available = ", ".join(configs.keys())
        raise KeyError(
            f"Model family '{model_family}' not found in config. "
            f"Available: {available}"
        )
    
    return configs[model_family]


def format_qa_text(
    question: str,
    answer: str,
    model_config: Dict,
    use_chat_template: bool = True
) -> str:
    """根据模型配置格式化问答文本
    
    Args:
        question: 问题
        answer: 答案
        model_config: 模型配置字典
        use_chat_template: 是否优先使用 tokenizer 的 chat template
        
    Returns:
        格式化后的文本
    """
    # 如果模型配置中没有模板标签，返回简单格式
    if not model_config.get("question_start_tag"):
        return f"Question: {question}\nAnswer: {answer}"
    
    question_start = model_config.get("question_start_tag", "")
    question_end = model_config.get("question_end_tag", "")
    answer_tag = model_config.get("answer_tag", "")
    
    formatted = question_start + question + question_end + answer_tag + answer
    return formatted


def get_model_config_summary(model_family: str) -> str:
    """获取模型配置的简要信息
    
    Args:
        model_family: 模型家族名称
        
    Returns:
        配置信息字符串
    """
    try:
        config = load_model_config(model_family)
        return (
            f"Model: {config['hf_key']}\n"
            f"Flash Attention: {config['flash_attention2']}\n"
            f"Gradient Checkpointing: {config['gradient_checkpointing']}"
        )
    except Exception as e:
        return f"Error loading config: {e}"

