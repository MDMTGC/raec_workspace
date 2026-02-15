"""
RAEC Self-Evaluation System - Performance tracking and confidence calibration

Tracks:
- Task outcomes (success/failure and why)
- Prediction accuracy (was RAEC's confidence calibrated?)
- Improvement over time
- Weak areas needing attention
"""

from .self_evaluator import SelfEvaluator, Evaluation, EvaluationType

__all__ = [
    'SelfEvaluator',
    'Evaluation',
    'EvaluationType',
]
