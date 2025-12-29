"""Trainer 注册器

使用注册器模式管理不同的 unlearning 方法，便于扩展
"""
from typing import Dict, Type
from .unlearn.base import UnlearnTrainer


TRAINER_REGISTRY: Dict[str, Type[UnlearnTrainer]] = {}


def register_trainer(name: str):
    """注册 trainer 的装饰器
    
    Usage:
        @register_trainer("grad_ascent")
        class GradAscent(UnlearnTrainer):
            ...
    """
    def decorator(cls: Type[UnlearnTrainer]):
        if not issubclass(cls, UnlearnTrainer):
            raise ValueError(f"{cls.__name__} must be a subclass of UnlearnTrainer")
        TRAINER_REGISTRY[name] = cls
        return cls
    return decorator


def get_trainer(name: str) -> Type[UnlearnTrainer]:
    """根据名称获取 trainer 类
    
    Args:
        name: trainer 名称
        
    Returns:
        Trainer 类
        
    Raises:
        ValueError: 如果名称不存在
    """
    if name not in TRAINER_REGISTRY:
        available = ", ".join(TRAINER_REGISTRY.keys())
        raise ValueError(
            f"Unknown trainer: '{name}'. Available trainers: {available}"
        )
    return TRAINER_REGISTRY[name]


def list_trainers() -> list:
    """列出所有已注册的 trainer"""
    return list(TRAINER_REGISTRY.keys())


# 自动注册所有 trainer
from .unlearn import (
    GradAscent,
    GradDiff,
    KLDivergence,
    DPO,
    NPO,
)

register_trainer("grad_ascent")(GradAscent)
register_trainer("grad_diff")(GradDiff)
register_trainer("kl")(KLDivergence)
register_trainer("kl_divergence")(KLDivergence)
register_trainer("dpo")(DPO)
register_trainer("npo")(NPO)

