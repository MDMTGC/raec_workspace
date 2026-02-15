"""
RAEC Goals System - Persistent objectives that guide behavior

Goals give RAEC direction. Not just reacting to requests,
but pursuing objectives over time.

Goal types:
- User-given: Explicit objectives from the user
- Inferred: Goals derived from user behavior/interests
- Self-improvement: RAEC's own growth objectives
- Maintenance: Ongoing system health goals
"""

from .goal_manager import GoalManager, Goal, GoalType, GoalStatus

__all__ = [
    'GoalManager',
    'Goal',
    'GoalType',
    'GoalStatus',
]
