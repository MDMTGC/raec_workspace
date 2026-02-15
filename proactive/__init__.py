"""
RAEC Proactive Communication - Reaching out when relevant

RAEC doesn't just respond - it can initiate contact when:
- It learned something relevant to a past conversation
- A goal deadline is approaching
- It noticed something the user should know
- It has a suggestion based on patterns
"""

from .notifier import Notifier, Notification, NotificationType

__all__ = [
    'Notifier',
    'Notification',
    'NotificationType',
]
