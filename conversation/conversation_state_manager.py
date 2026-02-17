"""Session-level conversation state manager for continuity."""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


REFERENCE_PATTERN = re.compile(r"\b(that|it|this|those|these|earlier|before|above|previous)\b", re.IGNORECASE)
TASK_SIGNAL_PATTERN = re.compile(
    r"\b(create|build|write|fix|update|implement|run|execute|deploy|analyze|summarize)\b",
    re.IGNORECASE,
)
COMMITMENT_PATTERN = re.compile(r"\b(i will|i'll|next i|i can|i should)\b", re.IGNORECASE)


@dataclass
class Turn:
    """A raw conversation turn pair."""

    user: str
    assistant: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user": self.user,
            "assistant": self.assistant,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Turn":
        return cls(
            user=payload.get("user", ""),
            assistant=payload.get("assistant", ""),
            timestamp=payload.get("timestamp", time.time()),
        )


@dataclass
class ConversationState:
    """Session-level continuity state for prompt assembly."""

    session_id: str
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    active_thread_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    mode: str = "chat"

    active_task: Optional[str] = None
    unresolved_references: List[str] = field(default_factory=list)
    last_commitments: List[str] = field(default_factory=list)

    rolling_summary: str = ""
    recent_turns: List[Turn] = field(default_factory=list)

    def update_from_turn(self, user_input: str, assistant_output: str, mode: str) -> None:
        """Update continuity state from a completed turn."""
        normalized_user_input = user_input.strip()

        self.last_updated = time.time()
        self.mode = mode
        self.recent_turns.append(Turn(user=normalized_user_input, assistant=assistant_output))

        # Task continuity: explicit task mode always refreshes active task.
        # Non-task turns only refresh task when action intent is clearly present.
        if mode == "task":
            self.active_task = normalized_user_input
        elif TASK_SIGNAL_PATTERN.search(normalized_user_input) and not REFERENCE_PATTERN.search(normalized_user_input):
            self.active_task = normalized_user_input

        references = REFERENCE_PATTERN.findall(normalized_user_input)
        if references:
            if self.active_task:
                # References can be resolved against active task context.
                self.unresolved_references = []
            else:
                self.unresolved_references = sorted({r.lower() for r in references})
        elif self.unresolved_references:
            # Clear stale unresolved markers once a non-reference turn appears.
            self.unresolved_references = []

        if COMMITMENT_PATTERN.search(assistant_output):
            self.last_commitments.append(assistant_output.strip()[:200])
            self.last_commitments = self.last_commitments[-5:]

    def compress_history(self, max_recent_turns: int = 6) -> bool:
        """Compress older turns into a deterministic rolling summary."""
        if len(self.recent_turns) <= max_recent_turns:
            return False

        overflow = self.recent_turns[:-max_recent_turns]
        self.recent_turns = self.recent_turns[-max_recent_turns:]

        compressed_lines = [
            f"- U: {turn.user[:100]} | A: {turn.assistant[:100]}"
            for turn in overflow
        ]
        prefix = f"{self.rolling_summary}\n" if self.rolling_summary else ""
        self.rolling_summary = (prefix + "\n".join(compressed_lines)).strip()
        return True

    def generate_prompt_context(self) -> str:
        """Build deterministic prompt context payload from current state."""
        recent = "\n".join(
            [f"- U: {turn.user[:120]}\n  A: {turn.assistant[:120]}" for turn in self.recent_turns[-3:]]
        )
        unresolved = ", ".join(self.unresolved_references) if self.unresolved_references else "none"
        commitments = "\n".join([f"- {c}" for c in self.last_commitments[-3:]]) or "- none"

        return (
            "ConversationState:\n"
            f"- session_id: {self.session_id}\n"
            f"- active_thread_id: {self.active_thread_id}\n"
            f"- mode: {self.mode}\n"
            f"- active_task: {self.active_task or 'none'}\n"
            f"- unresolved_references: {unresolved}\n"
            "- last_commitments:\n"
            f"{commitments}\n"
            f"- rolling_summary: {self.rolling_summary or 'none'}\n"
            "- recent_turns:\n"
            f"{recent or '- none'}"
        )

    def reset(self, keep_rolling_summary: bool = False) -> None:
        """Reset active thread state, optionally retaining rolling summary."""
        self.last_updated = time.time()
        self.active_thread_id = str(uuid.uuid4())[:8]
        self.mode = "chat"
        self.active_task = None
        self.unresolved_references = []
        self.last_commitments = []
        self.recent_turns = []
        if not keep_rolling_summary:
            self.rolling_summary = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "active_thread_id": self.active_thread_id,
            "mode": self.mode,
            "active_task": self.active_task,
            "unresolved_references": self.unresolved_references,
            "last_commitments": self.last_commitments,
            "rolling_summary": self.rolling_summary,
            "recent_turns": [turn.to_dict() for turn in self.recent_turns],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ConversationState":
        return cls(
            session_id=payload.get("session_id", str(uuid.uuid4())[:8]),
            created_at=payload.get("created_at", time.time()),
            last_updated=payload.get("last_updated", time.time()),
            active_thread_id=payload.get("active_thread_id", str(uuid.uuid4())[:8]),
            mode=payload.get("mode", "chat"),
            active_task=payload.get("active_task"),
            unresolved_references=payload.get("unresolved_references", []),
            last_commitments=payload.get("last_commitments", []),
            rolling_summary=payload.get("rolling_summary", ""),
            recent_turns=[Turn.from_dict(t) for t in payload.get("recent_turns", [])],
        )


class ConversationStateManager:
    """Persistence wrapper for ConversationState."""

    def __init__(self, state_path: str, session_id: str):
        self.state_path = state_path
        self.state = self._load_or_create(session_id=session_id)

    def _load_or_create(self, session_id: str) -> ConversationState:
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                state = ConversationState.from_dict(payload)
                if state.session_id != session_id:
                    state.session_id = session_id
                return state
            except Exception:
                pass
        return ConversationState(session_id=session_id)

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.state.to_dict(), f, indent=2)
