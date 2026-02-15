"""
Notifier - RAEC's proactive communication system

This enables RAEC to initiate contact rather than just respond.
Notifications are queued and delivered appropriately.
"""

import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
from enum import Enum


class NotificationType(Enum):
    """Types of proactive notifications"""
    LEARNING = "learning"         # "I learned something relevant"
    REMINDER = "reminder"         # "Don't forget about X"
    SUGGESTION = "suggestion"     # "You might want to consider..."
    ALERT = "alert"               # "Something needs your attention"
    UPDATE = "update"             # "Status update on X"
    QUESTION = "question"         # "I have a question about..."


class NotificationPriority(Enum):
    """How urgently should this be delivered"""
    LOW = 1       # Mention when convenient
    MEDIUM = 2    # Bring up at session start
    HIGH = 3      # Interrupt if possible
    CRITICAL = 4  # Must see immediately


class NotificationStatus(Enum):
    """Notification lifecycle"""
    PENDING = "pending"       # Waiting to be delivered
    DELIVERED = "delivered"   # Shown to user
    READ = "read"             # User acknowledged
    DISMISSED = "dismissed"   # User dismissed without reading
    EXPIRED = "expired"       # No longer relevant


@dataclass
class Notification:
    """A proactive message from RAEC to the user"""
    id: Optional[int]
    notification_type: str
    priority: int
    status: str

    # Content
    title: str
    message: str
    context: str  # Why is RAEC sending this?

    # Metadata
    created_at: str
    deliver_after: Optional[str]  # Don't deliver before this time
    expires_at: Optional[str]     # Don't deliver after this time
    delivered_at: Optional[str]
    read_at: Optional[str]

    # Linking
    related_goal: Optional[int]
    related_question: Optional[int]
    related_session: Optional[str]

    # Response
    requires_response: bool
    response: Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_row(cls, row: tuple) -> 'Notification':
        return cls(
            id=row[0],
            notification_type=row[1],
            priority=row[2],
            status=row[3],
            title=row[4],
            message=row[5],
            context=row[6],
            created_at=row[7],
            deliver_after=row[8],
            expires_at=row[9],
            delivered_at=row[10],
            read_at=row[11],
            related_goal=row[12],
            related_question=row[13],
            related_session=row[14],
            requires_response=bool(row[15]),
            response=row[16]
        )

    def is_deliverable(self) -> bool:
        """Can this notification be delivered now?"""
        if self.status != NotificationStatus.PENDING.value:
            return False

        now = datetime.now()

        if self.deliver_after:
            if datetime.fromisoformat(self.deliver_after) > now:
                return False

        if self.expires_at:
            if datetime.fromisoformat(self.expires_at) < now:
                return False

        return True


class Notifier:
    """
    Manages RAEC's proactive communications.

    RAEC can:
    - Queue notifications for later delivery
    - Deliver based on priority and timing
    - Track user responses
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        on_notification: Optional[Callable[[Notification], None]] = None
    ):
        if db_path is None:
            db_path = Path(__file__).parent.parent / "data" / "notifications.db"

        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.on_notification = on_notification

        self._init_db()

    def _init_db(self):
        """Initialize the notifications database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    notification_type TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 2,
                    status TEXT NOT NULL DEFAULT 'pending',
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    context TEXT,
                    created_at TEXT NOT NULL,
                    deliver_after TEXT,
                    expires_at TEXT,
                    delivered_at TEXT,
                    read_at TEXT,
                    related_goal INTEGER,
                    related_question INTEGER,
                    related_session TEXT,
                    requires_response INTEGER NOT NULL DEFAULT 0,
                    response TEXT
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_notif_status
                ON notifications(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_notif_priority
                ON notifications(priority DESC)
            """)
            conn.commit()

    def notify(
        self,
        title: str,
        message: str,
        notification_type: NotificationType,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        context: str = "",
        deliver_after: Optional[datetime] = None,
        expires_after: Optional[timedelta] = None,
        related_goal: Optional[int] = None,
        related_question: Optional[int] = None,
        related_session: Optional[str] = None,
        requires_response: bool = False
    ) -> Notification:
        """Create a notification"""
        now = datetime.now()

        expires_at = None
        if expires_after:
            expires_at = (now + expires_after).isoformat()

        notification = Notification(
            id=None,
            notification_type=notification_type.value,
            priority=priority.value,
            status=NotificationStatus.PENDING.value,
            title=title,
            message=message,
            context=context,
            created_at=now.isoformat(),
            deliver_after=deliver_after.isoformat() if deliver_after else None,
            expires_at=expires_at,
            delivered_at=None,
            read_at=None,
            related_goal=related_goal,
            related_question=related_question,
            related_session=related_session,
            requires_response=requires_response,
            response=None
        )

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO notifications
                (notification_type, priority, status, title, message, context,
                 created_at, deliver_after, expires_at, related_goal,
                 related_question, related_session, requires_response)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                notification.notification_type,
                notification.priority,
                notification.status,
                notification.title,
                notification.message,
                notification.context,
                notification.created_at,
                notification.deliver_after,
                notification.expires_at,
                notification.related_goal,
                notification.related_question,
                notification.related_session,
                1 if notification.requires_response else 0
            ))
            notification.id = cursor.lastrowid
            conn.commit()

        return notification

    def notify_learning(
        self,
        what_learned: str,
        relevance: str,
        priority: NotificationPriority = NotificationPriority.MEDIUM
    ) -> Notification:
        """Notify about something RAEC learned"""
        return self.notify(
            title="I learned something",
            message=what_learned,
            notification_type=NotificationType.LEARNING,
            priority=priority,
            context=f"This might be relevant because: {relevance}"
        )

    def notify_suggestion(
        self,
        suggestion: str,
        reasoning: str,
        priority: NotificationPriority = NotificationPriority.LOW
    ) -> Notification:
        """Notify with a suggestion"""
        return self.notify(
            title="Suggestion",
            message=suggestion,
            notification_type=NotificationType.SUGGESTION,
            priority=priority,
            context=reasoning
        )

    def notify_reminder(
        self,
        reminder: str,
        goal_id: Optional[int] = None,
        deliver_at: Optional[datetime] = None
    ) -> Notification:
        """Create a reminder notification"""
        return self.notify(
            title="Reminder",
            message=reminder,
            notification_type=NotificationType.REMINDER,
            priority=NotificationPriority.MEDIUM,
            deliver_after=deliver_at,
            related_goal=goal_id
        )

    def notify_question(
        self,
        question: str,
        context: str,
        priority: NotificationPriority = NotificationPriority.MEDIUM
    ) -> Notification:
        """Notify with a question for the user"""
        return self.notify(
            title="Question",
            message=question,
            notification_type=NotificationType.QUESTION,
            priority=priority,
            context=context,
            requires_response=True
        )

    def get_pending(self) -> List[Notification]:
        """Get all pending notifications ready for delivery"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT * FROM notifications
                WHERE status = 'pending'
                ORDER BY priority DESC, created_at ASC
            """).fetchall()

            notifications = [Notification.from_row(row) for row in rows]
            return [n for n in notifications if n.is_deliverable()]

    def get_for_session_start(self) -> List[Notification]:
        """Get notifications to show at session start"""
        pending = self.get_pending()
        # Return high priority and learning notifications
        return [
            n for n in pending
            if n.priority >= NotificationPriority.MEDIUM.value
            or n.notification_type == NotificationType.LEARNING.value
        ]

    def deliver(self, notification_id: int) -> bool:
        """Mark a notification as delivered"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE notifications
                SET status = ?, delivered_at = ?
                WHERE id = ?
            """, (
                NotificationStatus.DELIVERED.value,
                datetime.now().isoformat(),
                notification_id
            ))
            conn.commit()
        return True

    def mark_read(self, notification_id: int, response: Optional[str] = None) -> bool:
        """Mark a notification as read"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE notifications
                SET status = ?, read_at = ?, response = ?
                WHERE id = ?
            """, (
                NotificationStatus.READ.value,
                datetime.now().isoformat(),
                response,
                notification_id
            ))
            conn.commit()
        return True

    def dismiss(self, notification_id: int) -> bool:
        """Dismiss a notification"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE notifications
                SET status = ?
                WHERE id = ?
            """, (NotificationStatus.DISMISSED.value, notification_id))
            conn.commit()
        return True

    def expire_old(self) -> int:
        """Expire notifications past their expiry date"""
        now = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                UPDATE notifications
                SET status = ?
                WHERE status = 'pending' AND expires_at IS NOT NULL AND expires_at < ?
            """, (NotificationStatus.EXPIRED.value, now))
            conn.commit()
            return cursor.rowcount

    def get_stats(self) -> dict:
        """Get notification statistics"""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM notifications").fetchone()[0]
            pending = conn.execute(
                "SELECT COUNT(*) FROM notifications WHERE status = 'pending'"
            ).fetchone()[0]
            delivered = conn.execute(
                "SELECT COUNT(*) FROM notifications WHERE status = 'delivered'"
            ).fetchone()[0]
            read = conn.execute(
                "SELECT COUNT(*) FROM notifications WHERE status = 'read'"
            ).fetchone()[0]

            # By type
            by_type = {}
            for nt in NotificationType:
                count = conn.execute(
                    "SELECT COUNT(*) FROM notifications WHERE notification_type = ?",
                    (nt.value,)
                ).fetchone()[0]
                if count > 0:
                    by_type[nt.value] = count

            return {
                "total": total,
                "pending": pending,
                "delivered": delivered,
                "read": read,
                "read_rate": read / delivered if delivered > 0 else 0.0,
                "by_type": by_type
            }

    def format_pending(self) -> str:
        """Format pending notifications for display"""
        pending = self.get_pending()
        if not pending:
            return "No pending notifications."

        lines = [f"Pending Notifications ({len(pending)}):", "=" * 40]

        for n in pending:
            priority_marker = "!" * n.priority
            lines.append(f"\n[{priority_marker}] {n.title}")
            lines.append(f"    {n.message[:60]}...")
            lines.append(f"    Type: {n.notification_type}")
            if n.context:
                lines.append(f"    Context: {n.context[:40]}...")

        return "\n".join(lines)

    def format_session_greeting(self) -> str:
        """
        Format notifications as a greeting for session start.

        Returns a natural message incorporating notifications.
        """
        notifications = self.get_for_session_start()

        if not notifications:
            return ""

        parts = []

        # Group by type
        learnings = [n for n in notifications if n.notification_type == "learning"]
        reminders = [n for n in notifications if n.notification_type == "reminder"]
        suggestions = [n for n in notifications if n.notification_type == "suggestion"]
        questions = [n for n in notifications if n.notification_type == "question"]

        if learnings:
            if len(learnings) == 1:
                parts.append(f"While you were away, I learned something: {learnings[0].message}")
            else:
                parts.append(f"I learned {len(learnings)} things while you were away.")

        if reminders:
            parts.append(f"Reminder: {reminders[0].message}")

        if suggestions:
            parts.append(f"I have a suggestion: {suggestions[0].message}")

        if questions:
            parts.append(f"I have a question: {questions[0].message}")

        # Mark as delivered
        for n in notifications:
            self.deliver(n.id)

        return " ".join(parts)
