"""
RAEC Context Awareness - Perceiving the situation

Context includes:
- Time (time of day, day of week, time since last interaction)
- User state (apparent urgency, mood signals)
- Environment (what project, what files, what history)
- Conversational (topic continuity, question depth)
"""

from .context_manager import ContextManager, Context, UserState

__all__ = [
    'ContextManager',
    'Context',
    'UserState',
]
