"""
RAEC Curiosity Engine

Gives RAEC the drive to wonder, question, and explore.
Both directed (user-relevant) and ambient (just interesting).
"""

from .engine import CuriosityEngine
from .questions import QuestionQueue, Question, QuestionType
from .idle_loop import IdleLoop

__all__ = [
    'CuriosityEngine',
    'QuestionQueue',
    'Question',
    'QuestionType',
    'IdleLoop',
]
