"""
Goal Manager - Persistent objectives that guide RAEC's behavior

Goals are different from tasks:
- Tasks are immediate actions to complete
- Goals are ongoing directions to pursue

Goals influence:
- What RAEC is curious about
- How it prioritizes requests
- What it proactively works on
- How it evaluates its own performance
"""

import json
import sqlite3
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from enum import Enum


class GoalType(Enum):
    """Types of goals RAEC can pursue"""
    USER_GIVEN = "user_given"        # Explicit user objective
    INFERRED = "inferred"            # Derived from user behavior
    SELF_IMPROVEMENT = "self_improvement"  # RAEC's growth
    MAINTENANCE = "maintenance"      # System health
    PROJECT = "project"              # Ongoing project work


class GoalStatus(Enum):
    """Goal lifecycle states"""
    ACTIVE = "active"           # Currently pursuing
    PAUSED = "paused"           # Temporarily on hold
    COMPLETED = "completed"     # Successfully achieved
    ABANDONED = "abandoned"     # No longer relevant
    BLOCKED = "blocked"         # Can't progress


class GoalPriority(Enum):
    """Goal priority levels"""
    CRITICAL = 4    # Must address immediately
    HIGH = 3        # Important, address soon
    MEDIUM = 2      # Normal priority
    LOW = 1         # Nice to have
    BACKGROUND = 0  # Work on when nothing else


@dataclass
class GoalProgress:
    """Progress update on a goal"""
    timestamp: str
    description: str
    progress_delta: float  # Change in progress (0.0 to 1.0)
    context: Optional[str] = None


@dataclass
class Goal:
    """A persistent objective RAEC is pursuing"""
    id: Optional[int]
    name: str
    description: str
    goal_type: str
    status: str
    priority: int
    progress: float  # 0.0 to 1.0

    # Context
    created_at: str
    updated_at: str
    target_date: Optional[str]  # When should this be done?

    # Tracking
    success_criteria: List[str]  # How do we know it's done?
    progress_history: List[Dict]  # Progress updates
    related_skills: List[str]    # Skills that serve this goal
    related_questions: List[int]  # Curiosity questions related to this

    # Metadata
    source: str  # Where did this goal come from?
    tags: List[str]
    notes: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_row(cls, row: tuple, progress_history: List[Dict] = None) -> 'Goal':
        return cls(
            id=row[0],
            name=row[1],
            description=row[2],
            goal_type=row[3],
            status=row[4],
            priority=row[5],
            progress=row[6],
            created_at=row[7],
            updated_at=row[8],
            target_date=row[9],
            success_criteria=json.loads(row[10]) if row[10] else [],
            progress_history=progress_history or [],
            related_skills=json.loads(row[11]) if row[11] else [],
            related_questions=json.loads(row[12]) if row[12] else [],
            source=row[13],
            tags=json.loads(row[14]) if row[14] else [],
            notes=row[15] or ""
        )

    def is_active(self) -> bool:
        return self.status == GoalStatus.ACTIVE.value

    def is_complete(self) -> bool:
        return self.status == GoalStatus.COMPLETED.value


class GoalManager:
    """
    Manages RAEC's persistent goals.

    Goals provide direction and purpose beyond individual tasks.
    They influence curiosity, prioritization, and self-evaluation.
    """

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = Path(__file__).parent.parent / "data" / "goals.db"

        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the goals database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS goals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    goal_type TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    priority INTEGER NOT NULL DEFAULT 2,
                    progress REAL NOT NULL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    target_date TEXT,
                    success_criteria TEXT,
                    related_skills TEXT,
                    related_questions TEXT,
                    source TEXT NOT NULL,
                    tags TEXT,
                    notes TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS goal_progress (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    goal_id INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    description TEXT NOT NULL,
                    progress_delta REAL NOT NULL,
                    context TEXT,
                    FOREIGN KEY (goal_id) REFERENCES goals(id)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_goals_status
                ON goals(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_goals_priority
                ON goals(priority DESC)
            """)
            conn.commit()

    def create_goal(
        self,
        name: str,
        description: str,
        goal_type: GoalType,
        priority: GoalPriority = GoalPriority.MEDIUM,
        success_criteria: List[str] = None,
        target_date: Optional[str] = None,
        source: str = "user",
        tags: List[str] = None
    ) -> Goal:
        """Create a new goal"""
        now = datetime.now().isoformat()

        goal = Goal(
            id=None,
            name=name,
            description=description,
            goal_type=goal_type.value,
            status=GoalStatus.ACTIVE.value,
            priority=priority.value,
            progress=0.0,
            created_at=now,
            updated_at=now,
            target_date=target_date,
            success_criteria=success_criteria or [],
            progress_history=[],
            related_skills=[],
            related_questions=[],
            source=source,
            tags=tags or [],
            notes=""
        )

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO goals
                (name, description, goal_type, status, priority, progress,
                 created_at, updated_at, target_date, success_criteria,
                 related_skills, related_questions, source, tags, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                goal.name,
                goal.description,
                goal.goal_type,
                goal.status,
                goal.priority,
                goal.progress,
                goal.created_at,
                goal.updated_at,
                goal.target_date,
                json.dumps(goal.success_criteria),
                json.dumps(goal.related_skills),
                json.dumps(goal.related_questions),
                goal.source,
                json.dumps(goal.tags),
                goal.notes
            ))
            goal.id = cursor.lastrowid
            conn.commit()

        return goal

    def update_progress(
        self,
        goal_id: int,
        progress_delta: float,
        description: str,
        context: Optional[str] = None
    ) -> Goal:
        """Update progress on a goal"""
        now = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            # Get current progress
            current = conn.execute(
                "SELECT progress FROM goals WHERE id = ?",
                (goal_id,)
            ).fetchone()

            if not current:
                raise ValueError(f"Goal {goal_id} not found")

            new_progress = min(1.0, max(0.0, current[0] + progress_delta))

            # Update goal
            conn.execute("""
                UPDATE goals
                SET progress = ?, updated_at = ?
                WHERE id = ?
            """, (new_progress, now, goal_id))

            # Record progress update
            conn.execute("""
                INSERT INTO goal_progress
                (goal_id, timestamp, description, progress_delta, context)
                VALUES (?, ?, ?, ?, ?)
            """, (goal_id, now, description, progress_delta, context))

            # Auto-complete if progress reaches 1.0
            if new_progress >= 1.0:
                conn.execute("""
                    UPDATE goals SET status = ? WHERE id = ?
                """, (GoalStatus.COMPLETED.value, goal_id))

            conn.commit()

        return self.get_goal(goal_id)

    def set_status(self, goal_id: int, status: GoalStatus) -> Goal:
        """Change goal status"""
        now = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE goals
                SET status = ?, updated_at = ?
                WHERE id = ?
            """, (status.value, now, goal_id))
            conn.commit()

        return self.get_goal(goal_id)

    def get_goal(self, goal_id: int) -> Optional[Goal]:
        """Get a specific goal"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM goals WHERE id = ?",
                (goal_id,)
            ).fetchone()

            if not row:
                return None

            # Get progress history
            history = conn.execute("""
                SELECT timestamp, description, progress_delta, context
                FROM goal_progress
                WHERE goal_id = ?
                ORDER BY timestamp DESC
            """, (goal_id,)).fetchall()

            progress_history = [
                {
                    "timestamp": h[0],
                    "description": h[1],
                    "progress_delta": h[2],
                    "context": h[3]
                }
                for h in history
            ]

            return Goal.from_row(row, progress_history)

    def get_active_goals(self) -> List[Goal]:
        """Get all active goals, ordered by priority"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT * FROM goals
                WHERE status = 'active'
                ORDER BY priority DESC, created_at ASC
            """).fetchall()

            return [Goal.from_row(row) for row in rows]

    def get_goals_by_type(self, goal_type: GoalType) -> List[Goal]:
        """Get goals of a specific type"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT * FROM goals
                WHERE goal_type = ? AND status = 'active'
                ORDER BY priority DESC
            """, (goal_type.value,)).fetchall()

            return [Goal.from_row(row) for row in rows]

    def get_relevant_goals(self, context: str, limit: int = 3) -> List[Goal]:
        """Get goals relevant to current context (simple keyword match)"""
        active_goals = self.get_active_goals()

        # Simple relevance scoring
        scored = []
        context_lower = context.lower()

        for goal in active_goals:
            score = 0
            # Check name and description
            if any(word in context_lower for word in goal.name.lower().split()):
                score += 2
            if any(word in context_lower for word in goal.description.lower().split()):
                score += 1
            # Check tags
            for tag in goal.tags:
                if tag.lower() in context_lower:
                    score += 1

            if score > 0:
                scored.append((goal, score))

        # Sort by score, then priority
        scored.sort(key=lambda x: (x[1], x[0].priority), reverse=True)

        return [g for g, s in scored[:limit]]

    def infer_goal(
        self,
        observation: str,
        evidence: str,
        llm_interface: Any = None
    ) -> Optional[Goal]:
        """
        Infer a goal from observed user behavior.

        This is called when RAEC notices patterns that suggest
        the user has an unstated objective.
        """
        if not llm_interface:
            return None

        prompt = f"""Based on observed user behavior, infer if there's an implicit goal.

Observation: {observation}
Evidence: {evidence}

If there's a clear implicit goal, respond with:
GOAL: [goal name]
DESCRIPTION: [what they're trying to achieve]
CRITERIA: [how we'd know it's achieved]

If no clear goal can be inferred, respond with:
NO_GOAL

Be conservative - only infer goals with strong evidence."""

        response = llm_interface.generate(prompt, temperature=0.3, max_tokens=200)

        if "NO_GOAL" in response:
            return None

        # Parse response
        try:
            lines = response.strip().split('\n')
            name = ""
            description = ""
            criteria = []

            for line in lines:
                if line.startswith("GOAL:"):
                    name = line.replace("GOAL:", "").strip()
                elif line.startswith("DESCRIPTION:"):
                    description = line.replace("DESCRIPTION:", "").strip()
                elif line.startswith("CRITERIA:"):
                    criteria = [c.strip() for c in line.replace("CRITERIA:", "").split(",")]

            if name and description:
                return self.create_goal(
                    name=name,
                    description=description,
                    goal_type=GoalType.INFERRED,
                    priority=GoalPriority.MEDIUM,
                    success_criteria=criteria,
                    source=f"inferred: {observation[:50]}"
                )
        except Exception:
            pass

        return None

    def link_skill(self, goal_id: int, skill_id: str):
        """Link a skill to a goal"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT related_skills FROM goals WHERE id = ?",
                (goal_id,)
            ).fetchone()

            if row:
                skills = json.loads(row[0]) if row[0] else []
                if skill_id not in skills:
                    skills.append(skill_id)
                    conn.execute(
                        "UPDATE goals SET related_skills = ? WHERE id = ?",
                        (json.dumps(skills), goal_id)
                    )
                    conn.commit()

    def link_question(self, goal_id: int, question_id: int):
        """Link a curiosity question to a goal"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT related_questions FROM goals WHERE id = ?",
                (goal_id,)
            ).fetchone()

            if row:
                questions = json.loads(row[0]) if row[0] else []
                if question_id not in questions:
                    questions.append(question_id)
                    conn.execute(
                        "UPDATE goals SET related_questions = ? WHERE id = ?",
                        (json.dumps(questions), goal_id)
                    )
                    conn.commit()

    def get_stats(self) -> dict:
        """Get goal statistics"""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM goals").fetchone()[0]
            active = conn.execute(
                "SELECT COUNT(*) FROM goals WHERE status = 'active'"
            ).fetchone()[0]
            completed = conn.execute(
                "SELECT COUNT(*) FROM goals WHERE status = 'completed'"
            ).fetchone()[0]

            # By type
            by_type = {}
            for gt in GoalType:
                count = conn.execute(
                    "SELECT COUNT(*) FROM goals WHERE goal_type = ? AND status = 'active'",
                    (gt.value,)
                ).fetchone()[0]
                if count > 0:
                    by_type[gt.value] = count

            # Average progress of active goals
            avg_progress = conn.execute(
                "SELECT AVG(progress) FROM goals WHERE status = 'active'"
            ).fetchone()[0] or 0.0

            return {
                "total_goals": total,
                "active": active,
                "completed": completed,
                "by_type": by_type,
                "avg_progress": avg_progress
            }

    def format_active_goals(self) -> str:
        """Format active goals for display"""
        goals = self.get_active_goals()
        if not goals:
            return "No active goals."

        lines = [f"Active Goals ({len(goals)}):", "=" * 40]

        for g in goals:
            priority_str = "!" * g.priority
            progress_bar = "█" * int(g.progress * 10) + "░" * (10 - int(g.progress * 10))

            lines.append(f"\n[{priority_str}] {g.name}")
            lines.append(f"    {g.description[:60]}...")
            lines.append(f"    Progress: [{progress_bar}] {g.progress:.0%}")
            lines.append(f"    Type: {g.goal_type}")

        return "\n".join(lines)
