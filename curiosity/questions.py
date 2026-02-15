"""
Question Queue - Persistent storage for things RAEC wonders about

Questions come from:
- Uncertainty in responses ("I'm not sure about...")
- Gaps in knowledge during tasks
- Interesting tangents from conversations
- User topics worth exploring deeper
"""

import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from enum import Enum


class QuestionType(Enum):
    """Types of curiosity"""
    UNCERTAINTY = "uncertainty"      # RAEC wasn't sure about something
    KNOWLEDGE_GAP = "knowledge_gap"  # Needed info it didn't have
    TANGENT = "tangent"              # Interesting but off-topic
    USER_INTEREST = "user_interest"  # Something the user cares about
    FOLLOW_UP = "follow_up"          # Natural next question
    AMBIENT = "ambient"              # Just curious, no specific trigger


class QuestionPriority(Enum):
    """How urgently should this be investigated"""
    LOW = 1       # Whenever there's time
    MEDIUM = 2    # Worth looking into soon
    HIGH = 3      # Directly relevant to user needs
    URGENT = 4    # Blocking current work


@dataclass
class Question:
    """A single question RAEC wants to investigate"""
    id: Optional[int]
    question: str
    question_type: str
    priority: int
    context: str           # Why this question arose
    source_conversation: Optional[str]  # Session ID where it came from
    created_at: str
    investigated_at: Optional[str]
    resolved: bool
    resolution: Optional[str]  # What was learned

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_row(cls, row: tuple) -> 'Question':
        return cls(
            id=row[0],
            question=row[1],
            question_type=row[2],
            priority=row[3],
            context=row[4],
            source_conversation=row[5],
            created_at=row[6],
            investigated_at=row[7],
            resolved=bool(row[8]),
            resolution=row[9]
        )


class QuestionQueue:
    """
    Persistent queue of questions RAEC wants to explore.

    Questions are prioritized by:
    1. Priority level (user-relevant > ambient)
    2. Age (older questions shouldn't be forgotten)
    3. Type (uncertainties should be resolved)
    """

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = Path(__file__).parent.parent / "data" / "curiosity.db"

        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the questions database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS questions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    question_type TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 2,
                    context TEXT NOT NULL,
                    source_conversation TEXT,
                    created_at TEXT NOT NULL,
                    investigated_at TEXT,
                    resolved INTEGER NOT NULL DEFAULT 0,
                    resolution TEXT
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_questions_priority
                ON questions(priority DESC, created_at ASC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_questions_resolved
                ON questions(resolved)
            """)
            conn.commit()

    def add(
        self,
        question: str,
        question_type: QuestionType,
        context: str,
        priority: QuestionPriority = QuestionPriority.MEDIUM,
        source_conversation: Optional[str] = None
    ) -> Question:
        """Add a new question to the queue"""
        q = Question(
            id=None,
            question=question,
            question_type=question_type.value,
            priority=priority.value,
            context=context,
            source_conversation=source_conversation,
            created_at=datetime.now().isoformat(),
            investigated_at=None,
            resolved=False,
            resolution=None
        )

        with sqlite3.connect(self.db_path) as conn:
            # Check for duplicate questions
            existing = conn.execute(
                "SELECT id FROM questions WHERE question = ? AND resolved = 0",
                (question,)
            ).fetchone()

            if existing:
                # Update priority if new one is higher
                conn.execute(
                    "UPDATE questions SET priority = MAX(priority, ?) WHERE id = ?",
                    (priority.value, existing[0])
                )
                q.id = existing[0]
            else:
                cursor = conn.execute("""
                    INSERT INTO questions
                    (question, question_type, priority, context, source_conversation, created_at, resolved)
                    VALUES (?, ?, ?, ?, ?, ?, 0)
                """, (
                    q.question,
                    q.question_type,
                    q.priority,
                    q.context,
                    q.source_conversation,
                    q.created_at
                ))
                q.id = cursor.lastrowid
            conn.commit()

        return q

    def get_next(self, count: int = 1) -> List[Question]:
        """Get the next questions to investigate (highest priority, oldest first)"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM questions
                WHERE resolved = 0
                ORDER BY priority DESC, created_at ASC
                LIMIT ?
            """, (count,))
            return [Question.from_row(row) for row in cursor.fetchall()]

    def get_by_type(self, question_type: QuestionType, limit: int = 10) -> List[Question]:
        """Get questions of a specific type"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM questions
                WHERE question_type = ? AND resolved = 0
                ORDER BY priority DESC, created_at ASC
                LIMIT ?
            """, (question_type.value, limit))
            return [Question.from_row(row) for row in cursor.fetchall()]

    def mark_investigating(self, question_id: int):
        """Mark a question as currently being investigated"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE questions
                SET investigated_at = ?
                WHERE id = ?
            """, (datetime.now().isoformat(), question_id))
            conn.commit()

    def resolve(self, question_id: int, resolution: str):
        """Mark a question as resolved with what was learned"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE questions
                SET resolved = 1, resolution = ?, investigated_at = ?
                WHERE id = ?
            """, (resolution, datetime.now().isoformat(), question_id))
            conn.commit()

    def get_recent_resolutions(self, limit: int = 10) -> List[Question]:
        """Get recently resolved questions (for 'what I learned' summaries)"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM questions
                WHERE resolved = 1
                ORDER BY investigated_at DESC
                LIMIT ?
            """, (limit,))
            return [Question.from_row(row) for row in cursor.fetchall()]

    def get_unresolved_count(self) -> int:
        """Get count of unresolved questions"""
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM questions WHERE resolved = 0"
            ).fetchone()[0]

    def get_stats(self) -> dict:
        """Get queue statistics"""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
            unresolved = conn.execute(
                "SELECT COUNT(*) FROM questions WHERE resolved = 0"
            ).fetchone()[0]
            resolved = total - unresolved

            # By type
            by_type = {}
            for qt in QuestionType:
                count = conn.execute(
                    "SELECT COUNT(*) FROM questions WHERE question_type = ? AND resolved = 0",
                    (qt.value,)
                ).fetchone()[0]
                if count > 0:
                    by_type[qt.value] = count

            # By priority
            by_priority = {}
            for qp in QuestionPriority:
                count = conn.execute(
                    "SELECT COUNT(*) FROM questions WHERE priority = ? AND resolved = 0",
                    (qp.value,)
                ).fetchone()[0]
                if count > 0:
                    by_priority[qp.name.lower()] = count

            return {
                "total_questions": total,
                "unresolved": unresolved,
                "resolved": resolved,
                "by_type": by_type,
                "by_priority": by_priority
            }

    def format_pending(self, limit: int = 5) -> str:
        """Format pending questions for display"""
        questions = self.get_next(limit)
        if not questions:
            return "No pending questions."

        lines = [f"Pending Questions ({self.get_unresolved_count()} total):", "=" * 40]
        for q in questions:
            priority_marker = "!" * q.priority
            lines.append(f"\n[{priority_marker}] {q.question}")
            lines.append(f"    Type: {q.question_type}")
            lines.append(f"    Context: {q.context[:60]}...")

        return "\n".join(lines)
