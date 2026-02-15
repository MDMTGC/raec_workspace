"""
RAEC Preferences System - Learned user and context preferences

Preferences are patterns RAEC learns over time:
- User preferences: How the user likes things done
- Context preferences: What works best in different situations
- Style preferences: Communication and output formatting
"""

from .preference_manager import PreferenceManager, Preference, PreferenceType

__all__ = [
    'PreferenceManager',
    'Preference',
    'PreferenceType',
]
