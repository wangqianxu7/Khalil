"""评估模块

用于评估 unlearning 的效果，包括：
- Forget Quality（遗忘质量）
- Model Utility（模型实用性）
"""
from .evaluator import Evaluator
from .metrics import (
    compute_rouge,
    compute_bleu,
    compute_perplexity,
    compute_probability,
    compute_truth_ratio,
)

__all__ = [
    "Evaluator",
    "compute_rouge",
    "compute_bleu",
    "compute_perplexity",
    "compute_probability",
    "compute_truth_ratio",
]

