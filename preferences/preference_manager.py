"""
Preference Manager - Learning and applying user/context preferences

Preferences are learned from:
- Explicit user statements ("I prefer concise answers")
- Observed patterns (user always edits verbose responses)
- Feedback (positive/negative reactions to outputs)
- Context patterns (coding tasks need different style than chat)
"""

import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from enum import Enum


class PreferenceType(Enum):
    """Types of preferences"""
    COMMUNICATION = "communication"  # How to talk (verbose, concise, formal, casual)
    FORMATTING = "formatting"        # Output format (markdown, plain, code style)
    WORKFLOW = "workflow"            # How to work (ask first, just do it, step by step)
    DOMAIN = "domain"                # Domain-specific (coding conventions, terminology)
    TIMING = "timing"                # When/how often (proactive updates, batch responses)
    CONTENT = "content"              # What to include/exclude (explanations, examples)


class PreferenceStrength(Enum):
    """How strongly established is this preference"""
    WEAK = 1        # Observed once or twice
    MODERATE = 2    # Consistent pattern
    STRONG = 3      # Explicitly stated or very consistent
    ABSOLUTE = 4    # Hard rule from user


@dataclass
class Preference:
    """A learned preference"""
    id: Optional[int]
    name: str
    description: str
    preference_type: str
    strength: int

    # The preference itself
    value: str  # What the preference is
    anti_value: str  # What to avoid (opposite)

    # Evidence
    evidence_count: int  # How many times observed
    last_evidence: str   # Most recent supporting evidence
    created_at: str
    updated_at: str

    # Context
    applies_to: List[str]  # Contexts where this applies (e.g., ["coding", "chat"])
    source: str  # How it was learned (explicit, observed, feedback)

    # Tracking
    applied_count: int  # How many times we've used this preference
    success_rate: float  # How often it led to good outcomes

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_row(cls, row: tuple) -> 'Preference':
        return cls(
            id=row[0],
            name=row[1],
            description=row[2],
            preference_type=row[3],
            strength=row[4],
            value=row[5],
            anti_value=row[6],
            evidence_count=row[7],
            last_evidence=row[8],
            created_at=row[9],
            updated_at=row[10],
            applies_to=json.loads(row[11]) if row[11] else [],
            source=row[12],
            applied_count=row[13],
            success_rate=row[14]
        )


class PreferenceManager:
    """
    Manages learned preferences for RAEC.

    Preferences influence how RAEC:
    - Communicates (verbosity, formality, structure)
    - Formats output (markdown, code style, examples)
    - Approaches tasks (autonomous vs collaborative)
    - Handles uncertainty (ask vs guess)
    """

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = Path(__file__).parent.parent / "data" / "preferences.db"

        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the preferences database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT NOT NULL,
                    preference_type TEXT NOT NULL,
                    strength INTEGER NOT NULL DEFAULT 1,
                    value TEXT NOT NULL,
                    anti_value TEXT,
                    evidence_count INTEGER NOT NULL DEFAULT 1,
                    last_evidence TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    applies_to TEXT,
                    source TEXT NOT NULL,
                    applied_count INTEGER NOT NULL DEFAULT 0,
                    success_rate REAL NOT NULL DEFAULT 0.5
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_preferences_type
                ON preferences(preference_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_preferences_strength
                ON preferences(strength DESC)
            """)
            conn.commit()

    def learn_preference(
        self,
        name: str,
        value: str,
        preference_type: PreferenceType,
        description: str = "",
        anti_value: str = "",
        evidence: str = "",
        applies_to: List[str] = None,
        source: str = "observed",
        strength: PreferenceStrength = PreferenceStrength.WEAK
    ) -> Preference:
        """
        Learn or reinforce a preference.

        If the preference already exists, reinforce it.
        If new, create it.
        """
        now = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            # Check if preference exists
            existing = conn.execute(
                "SELECT id, evidence_count, strength FROM preferences WHERE name = ?",
                (name,)
            ).fetchone()

            if existing:
                # Reinforce existing preference
                new_count = existing[1] + 1
                # Strength increases with evidence (up to a point)
                new_strength = min(4, existing[2] + (1 if new_count % 3 == 0 else 0))

                conn.execute("""
                    UPDATE preferences
                    SET evidence_count = ?, strength = ?, last_evidence = ?, updated_at = ?
                    WHERE id = ?
                """, (new_count, new_strength, evidence, now, existing[0]))
                conn.commit()

                return self.get_preference(existing[0])
            else:
                # Create new preference
                pref = Preference(
                    id=None,
                    name=name,
                    description=description or f"Preference: {value}",
                    preference_type=preference_type.value,
                    strength=strength.value,
                    value=value,
                    anti_value=anti_value,
                    evidence_count=1,
                    last_evidence=evidence,
                    created_at=now,
                    updated_at=now,
                    applies_to=applies_to or [],
                    source=source,
                    applied_count=0,
                    success_rate=0.5
                )

                cursor = conn.execute("""
                    INSERT INTO preferences
                    (name, description, preference_type, strength, value, anti_value,
                     evidence_count, last_evidence, created_at, updated_at, applies_to,
                     source, applied_count, success_rate)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pref.name, pref.description, pref.preference_type, pref.strength,
                    pref.value, pref.anti_value, pref.evidence_count, pref.last_evidence,
                    pref.created_at, pref.updated_at, json.dumps(pref.applies_to),
                    pref.source, pref.applied_count, pref.success_rate
                ))
                pref.id = cursor.lastrowid
                conn.commit()

                return pref

    def learn_from_explicit(self, statement: str, llm_interface: Any = None) -> Optional[Preference]:
        """
        Learn preference from explicit user statement.

        Examples:
        - "I prefer short answers"
        - "Always use markdown"
        - "Don't explain obvious things"
        """
        if not llm_interface:
            return None

        prompt = f"""Extract a preference from this user statement:

"{statement}"

Respond in this format:
NAME: [short identifier, like "concise_responses"]
TYPE: [communication, formatting, workflow, domain, timing, or content]
VALUE: [what to do]
ANTI_VALUE: [what to avoid]
APPLIES_TO: [comma-separated contexts, or "all"]

If this isn't a clear preference statement, respond with:
NOT_A_PREFERENCE"""

        response = llm_interface.generate(prompt, temperature=0.2, max_tokens=150)

        if "NOT_A_PREFERENCE" in response:
            return None

        try:
            lines = response.strip().split('\n')
            parsed = {}

            for line in lines:
                if ':' in line:
                    key, val = line.split(':', 1)
                    parsed[key.strip().upper()] = val.strip()

            if 'NAME' in parsed and 'VALUE' in parsed:
                ptype = PreferenceType.COMMUNICATION
                for pt in PreferenceType:
                    if pt.value == parsed.get('TYPE', '').lower():
                        ptype = pt
                        break

                applies = []
                if parsed.get('APPLIES_TO', 'all').lower() != 'all':
                    applies = [a.strip() for a in parsed['APPLIES_TO'].split(',')]

                return self.learn_preference(
                    name=parsed['NAME'].lower().replace(' ', '_'),
                    value=parsed['VALUE'],
                    preference_type=ptype,
                    anti_value=parsed.get('ANTI_VALUE', ''),
                    evidence=statement,
                    applies_to=applies,
                    source="explicit",
                    strength=PreferenceStrength.STRONG
                )
        except Exception:
            pass

        return None

    def learn_from_feedback(
        self,
        original_output: str,
        feedback: str,
        was_positive: bool,
        context: str = ""
    ) -> Optional[Preference]:
        """
        Learn preference from user feedback on output.

        Positive feedback reinforces patterns.
        Negative feedback creates anti-patterns.
        """
        # This would ideally use LLM to extract what was good/bad
        # For now, simple heuristics

        if was_positive:
            # Reinforce whatever we did
            if len(original_output) < 200:
                return self.learn_preference(
                    name="concise_responses",
                    value="Keep responses brief and to the point",
                    preference_type=PreferenceType.COMMUNICATION,
                    anti_value="Long, verbose explanations",
                    evidence=f"Positive feedback on short response: {feedback[:50]}",
                    source="feedback"
                )
        else:
            # Learn what not to do
            if "too long" in feedback.lower() or "verbose" in feedback.lower():
                return self.learn_preference(
                    name="concise_responses",
                    value="Keep responses brief",
                    preference_type=PreferenceType.COMMUNICATION,
                    anti_value="Long, verbose explanations",
                    evidence=f"Negative feedback: {feedback[:50]}",
                    source="feedback"
                )

        return None

    def get_preference(self, pref_id: int) -> Optional[Preference]:
        """Get a specific preference"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM preferences WHERE id = ?",
                (pref_id,)
            ).fetchone()

            return Preference.from_row(row) if row else None

    def get_preferences_for_context(self, context: str) -> List[Preference]:
        """Get preferences applicable to a context"""
        with sqlite3.connect(self.db_path) as conn:
            # Get all preferences, then filter
            rows = conn.execute("""
                SELECT * FROM preferences
                ORDER BY strength DESC, evidence_count DESC
            """).fetchall()

            preferences = [Preference.from_row(row) for row in rows]

            # Filter by context
            applicable = []
            context_lower = context.lower()

            for pref in preferences:
                if not pref.applies_to:  # Applies to all
                    applicable.append(pref)
                elif any(ctx.lower() in context_lower for ctx in pref.applies_to):
                    applicable.append(pref)

            return applicable

    def get_all_preferences(self) -> List[Preference]:
        """Get all preferences"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT * FROM preferences
                ORDER BY strength DESC, evidence_count DESC
            """).fetchall()

            return [Preference.from_row(row) for row in rows]

    def record_application(self, pref_id: int, was_successful: bool):
        """Record that a preference was applied and whether it worked"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT applied_count, success_rate FROM preferences WHERE id = ?",
                (pref_id,)
            ).fetchone()

            if row:
                count = row[0] + 1
                # Running average of success rate
                new_rate = (row[1] * row[0] + (1.0 if was_successful else 0.0)) / count

                conn.execute("""
                    UPDATE preferences
                    SET applied_count = ?, success_rate = ?, updated_at = ?
                    WHERE id = ?
                """, (count, new_rate, datetime.now().isoformat(), pref_id))
                conn.commit()

    def build_preference_prompt(self, context: str) -> str:
        """
        Build a prompt section describing applicable preferences.

        This gets injected into prompts to guide RAEC's behavior.
        """
        preferences = self.get_preferences_for_context(context)

        if not preferences:
            return ""

        lines = ["User preferences to follow:"]

        for pref in preferences[:5]:  # Limit to top 5
            strength_marker = "!" * pref.strength
            lines.append(f"- [{strength_marker}] {pref.value}")
            if pref.anti_value:
                lines.append(f"  (Avoid: {pref.anti_value})")

        return "\n".join(lines)

    def get_stats(self) -> dict:
        """Get preference statistics"""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM preferences").fetchone()[0]
            total_evidence = conn.execute(
                "SELECT SUM(evidence_count) FROM preferences"
            ).fetchone()[0] or 0
            total_applications = conn.execute(
                "SELECT SUM(applied_count) FROM preferences"
            ).fetchone()[0] or 0

            # By type
            by_type = {}
            for pt in PreferenceType:
                count = conn.execute(
                    "SELECT COUNT(*) FROM preferences WHERE preference_type = ?",
                    (pt.value,)
                ).fetchone()[0]
                if count > 0:
                    by_type[pt.value] = count

            # Average success rate
            avg_success = conn.execute(
                "SELECT AVG(success_rate) FROM preferences WHERE applied_count > 0"
            ).fetchone()[0] or 0.5

            return {
                "total_preferences": total,
                "total_evidence": total_evidence,
                "total_applications": total_applications,
                "by_type": by_type,
                "avg_success_rate": avg_success
            }

    def format_preferences(self) -> str:
        """Format all preferences for display"""
        prefs = self.get_all_preferences()
        if not prefs:
            return "No learned preferences yet."

        lines = [f"Learned Preferences ({len(prefs)}):", "=" * 40]

        for p in prefs:
            strength = "!" * p.strength
            lines.append(f"\n[{strength}] {p.name}")
            lines.append(f"    Value: {p.value}")
            if p.anti_value:
                lines.append(f"    Avoid: {p.anti_value}")
            lines.append(f"    Type: {p.preference_type} | Evidence: {p.evidence_count} | Applied: {p.applied_count}")

        return "\n".join(lines)
