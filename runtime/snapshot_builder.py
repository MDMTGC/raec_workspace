"""Session snapshot payload construction helpers."""

from __future__ import annotations

import time
from typing import Any, Dict, List


class SessionSnapshotBuilder:
    """Build deterministic session snapshot payloads."""

    def build(
        self,
        *,
        session_id: str,
        message_count: int,
        active_topics: List[str],
        outcomes: List[str],
        memory_context: List[Dict[str, Any]],
        recent_turns: List[Dict[str, Any]],
        conversation_state: Dict[str, Any],
        last_persistence_status: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "session_id": session_id,
            "message_count": message_count,
            "active_topics": active_topics,
            "outcomes": outcomes,
            "memory_context": memory_context,
            "recent_turns": recent_turns,
            "conversation_state": conversation_state,
            "last_persistence_status": last_persistence_status,
            "captured_at": time.time(),
        }
