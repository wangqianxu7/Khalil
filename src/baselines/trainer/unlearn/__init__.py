"""Unlearning Trainers

整合了多种 unlearning 方法：
- GradAscent: 梯度上升
- GradDiff: 梯度差异
- KLDivergence: KL 散度
- DPO: Direct Preference Optimization
- NPO: Negative Preference Optimization
"""

from .base import UnlearnTrainer
from .grad_ascent import GradAscent
from .grad_diff import GradDiff
from .kl_divergence import KLDivergence
from .dpo import DPO
from .npo import NPO

__all__ = [
    "UnlearnTrainer",
    "GradAscent",
    "GradDiff",
    "KLDivergence",
    "DPO",
    "NPO",
]

