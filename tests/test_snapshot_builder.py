from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.snapshot_builder import SessionSnapshotBuilder


def test_snapshot_builder_includes_core_sections() -> None:
    builder = SessionSnapshotBuilder()
    payload = builder.build(
        session_id="sess-1",
        message_count=4,
        active_topics=["continuity"],
        outcomes=["ok"],
        memory_context=[{"content": "x"}],
        recent_turns=[{"role": "user", "content": "hello"}],
        conversation_state={"active_thread_id": "th-1"},
        last_persistence_status={"reason": "turn_complete", "ok": True},
    )

    assert payload["session_id"] == "sess-1"
    assert payload["message_count"] == 4
    assert payload["active_topics"] == ["continuity"]
    assert payload["conversation_state"] == {"active_thread_id": "th-1"}
    assert payload["last_persistence_status"] == {"reason": "turn_complete", "ok": True}
    assert "captured_at" in payload
