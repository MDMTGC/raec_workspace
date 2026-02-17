from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import Raec


def test_persist_runtime_state_tracks_last_status() -> None:
    raec = Raec.__new__(Raec)
    raec.conversation = object()
    raec.conversation_state = object()
    raec.identity = object()
    raec._last_persistence_status = {}

    class _Persistence:
        def persist(self, **_kwargs):
            return SimpleNamespace(
                ok=True,
                order=["conversation", "conversation_state", "identity"],
                failures=[],
                persisted_at=123.45,
            )

    raec.persistence = _Persistence()

    raec._persist_runtime_state(reason="turn_complete")

    assert raec._last_persistence_status == {
        "reason": "turn_complete",
        "ok": True,
        "order": ["conversation", "conversation_state", "identity"],
        "failures": [],
        "persisted_at": 123.45,
    }


def test_snapshot_includes_last_persistence_status(tmp_path: Path) -> None:
    raec = Raec.__new__(Raec)
    raec.snapshot_dir = str(tmp_path)
    raec._last_memory_context = [{"memory": "x"}]
    raec._last_persistence_status = {"reason": "reset", "ok": True}

    class _Session:
        session_id = "sess-1"
        key_topics = ["continuity"]
        outcomes = ["ok"]

    class _Conversation:
        current_session = _Session()

        def message_count(self) -> int:
            return 2

        def get_context_messages(self, limit: int = 10):
            return [{"role": "user", "content": "hello"}][:limit]

    class _ConversationState:
        def to_dict(self):
            return {"active_thread_id": "th-1"}

    class _ConversationStateManager:
        state = _ConversationState()

    raec.conversation = _Conversation()
    raec.conversation_state = _ConversationStateManager()

    output_path = raec._create_session_snapshot(limit=5)
    payload = json.loads(Path(output_path).read_text(encoding="utf-8"))

    assert payload["last_persistence_status"] == {"reason": "reset", "ok": True}
    assert payload["conversation_state"] == {"active_thread_id": "th-1"}
