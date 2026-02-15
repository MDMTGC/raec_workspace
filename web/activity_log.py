"""
Activity Log - Transparent record of all web activity

Every fetch, every search, every reason why. Nothing hidden.
"""

import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from enum import Enum


class ActivityType(Enum):
    """Types of web activity"""
    FETCH = "fetch"
    SEARCH = "search"


@dataclass
class WebActivity:
    """A single web activity record"""
    id: Optional[int]
    timestamp: str
    activity_type: str
    url: Optional[str]
    query: Optional[str]
    reason: str  # Why RAEC did this
    success: bool
    result_summary: str  # Brief description of what was found
    triggered_by: str  # "user" or "autonomous"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_row(cls, row: tuple) -> 'WebActivity':
        return cls(
            id=row[0],
            timestamp=row[1],
            activity_type=row[2],
            url=row[3],
            query=row[4],
            reason=row[5],
            success=row[6],
            result_summary=row[7],
            triggered_by=row[8]
        )


class ActivityLog:
    """
    Persistent log of all web activity.

    Transparency is non-negotiable. Every web access is recorded
    with full context: what, when, why, and what was found.
    """

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = Path(__file__).parent.parent / "data" / "web_activity.db"

        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the activity database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS web_activity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    activity_type TEXT NOT NULL,
                    url TEXT,
                    query TEXT,
                    reason TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    result_summary TEXT NOT NULL,
                    triggered_by TEXT NOT NULL
                )
            """)

            # Index for efficient querying
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_activity_timestamp
                ON web_activity(timestamp DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_activity_type
                ON web_activity(activity_type)
            """)
            conn.commit()

    def log_fetch(
        self,
        url: str,
        reason: str,
        success: bool,
        result_summary: str,
        triggered_by: str = "user"
    ) -> WebActivity:
        """Log a URL fetch"""
        activity = WebActivity(
            id=None,
            timestamp=datetime.now().isoformat(),
            activity_type=ActivityType.FETCH.value,
            url=url,
            query=None,
            reason=reason,
            success=success,
            result_summary=result_summary,
            triggered_by=triggered_by
        )
        return self._save(activity)

    def log_search(
        self,
        query: str,
        reason: str,
        success: bool,
        result_summary: str,
        triggered_by: str = "user"
    ) -> WebActivity:
        """Log a web search"""
        activity = WebActivity(
            id=None,
            timestamp=datetime.now().isoformat(),
            activity_type=ActivityType.SEARCH.value,
            url=None,
            query=query,
            reason=reason,
            success=success,
            result_summary=result_summary,
            triggered_by=triggered_by
        )
        return self._save(activity)

    def _save(self, activity: WebActivity) -> WebActivity:
        """Save activity to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO web_activity
                (timestamp, activity_type, url, query, reason, success, result_summary, triggered_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                activity.timestamp,
                activity.activity_type,
                activity.url,
                activity.query,
                activity.reason,
                1 if activity.success else 0,
                activity.result_summary,
                activity.triggered_by
            ))
            activity.id = cursor.lastrowid
            conn.commit()
        return activity

    def get_recent(self, limit: int = 20) -> List[WebActivity]:
        """Get recent activity"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM web_activity
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            return [WebActivity.from_row(row) for row in cursor.fetchall()]

    def get_autonomous(self, limit: int = 50) -> List[WebActivity]:
        """Get autonomous (non-user-triggered) activity"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM web_activity
                WHERE triggered_by = 'autonomous'
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            return [WebActivity.from_row(row) for row in cursor.fetchall()]

    def get_by_type(self, activity_type: ActivityType, limit: int = 50) -> List[WebActivity]:
        """Get activity by type"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM web_activity
                WHERE activity_type = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (activity_type.value, limit))
            return [WebActivity.from_row(row) for row in cursor.fetchall()]

    def get_stats(self) -> dict:
        """Get activity statistics"""
        with sqlite3.connect(self.db_path) as conn:
            # Total counts
            total = conn.execute("SELECT COUNT(*) FROM web_activity").fetchone()[0]
            fetches = conn.execute(
                "SELECT COUNT(*) FROM web_activity WHERE activity_type = 'fetch'"
            ).fetchone()[0]
            searches = conn.execute(
                "SELECT COUNT(*) FROM web_activity WHERE activity_type = 'search'"
            ).fetchone()[0]
            autonomous = conn.execute(
                "SELECT COUNT(*) FROM web_activity WHERE triggered_by = 'autonomous'"
            ).fetchone()[0]
            successful = conn.execute(
                "SELECT COUNT(*) FROM web_activity WHERE success = 1"
            ).fetchone()[0]

            return {
                "total_activities": total,
                "fetches": fetches,
                "searches": searches,
                "autonomous_actions": autonomous,
                "success_rate": successful / total if total > 0 else 0,
                "user_triggered": total - autonomous
            }

    def format_recent(self, limit: int = 10) -> str:
        """Format recent activity for display"""
        activities = self.get_recent(limit)
        if not activities:
            return "No web activity recorded."

        lines = ["Recent Web Activity:", "=" * 40]
        for a in activities:
            time_str = datetime.fromisoformat(a.timestamp).strftime("%H:%M:%S")
            trigger = "🤖" if a.triggered_by == "autonomous" else "👤"
            status = "✓" if a.success else "✗"

            if a.activity_type == "fetch":
                lines.append(f"{trigger} {time_str} {status} FETCH: {a.url[:50]}...")
            else:
                lines.append(f"{trigger} {time_str} {status} SEARCH: {a.query}")

            lines.append(f"   Reason: {a.reason}")
            lines.append(f"   Result: {a.result_summary[:60]}...")
            lines.append("")

        return "\n".join(lines)
