from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conversation.conversation_manager import ConversationManager
from conversation.conversation_state_manager import ConversationStateManager
from identity.self_model import SelfModel
from runtime.persistence import SessionPersistenceCoordinator
from runtime.snapshot_builder import SessionSnapshotBuilder


def test_end_to_end_persistence_roundtrip(tmp_path: Path) -> None:
    conversation_path = tmp_path / "conversation" / "conversation_state.json"
    state_manager_path = tmp_path / "conversation" / "conversation_state_manager.json"
    identity_path = tmp_path / "identity" / "identity.json"

    conversation = ConversationManager(state_path=str(conversation_path))
    state_manager = ConversationStateManager(
        state_path=str(state_manager_path),
        session_id=conversation.current_session.session_id,
    )
    identity = SelfModel(identity_path=str(identity_path))

    user_input = "Create a deployment checklist"
    assistant_output = "I will draft a concise checklist with rollback steps."

    conversation.add_user_message(user_input)
    conversation.add_assistant_message(assistant_output)
    identity.record_interaction()

    state_manager.state.update_from_turn(
        user_input=user_input,
        assistant_output=assistant_output,
        mode="task",
    )
    state_manager.state.compress_history(max_recent_turns=1)

    coordinator = SessionPersistenceCoordinator()
    result = coordinator.persist(
        conversation=conversation,
        conversation_state=state_manager,
        identity=identity,
    )

    assert result.ok is True
    assert result.order == ["conversation", "conversation_state", "identity"]

    reloaded_conversation = ConversationManager(state_path=str(conversation_path))
    reloaded_state_manager = ConversationStateManager(
        state_path=str(state_manager_path),
        session_id=reloaded_conversation.current_session.session_id,
    )
    reloaded_identity = SelfModel(identity_path=str(identity_path))

    assert reloaded_conversation.message_count() >= 2
    assert reloaded_state_manager.state.active_task == user_input
    assert reloaded_state_manager.state.mode == "task"
    assert reloaded_identity.identity.interactions_count >= 1

    snapshot = SessionSnapshotBuilder().build(
        session_id=reloaded_conversation.current_session.session_id,
        message_count=reloaded_conversation.message_count(),
        active_topics=list(reloaded_conversation.current_session.key_topics),
        outcomes=list(reloaded_conversation.current_session.outcomes),
        memory_context=[],
        recent_turns=reloaded_conversation.get_context_messages(limit=2),
        conversation_state=reloaded_state_manager.state.to_dict(),
        last_persistence_status={
            "reason": "roundtrip_test",
            "ok": result.ok,
            "order": result.order,
            "failures": result.failures,
        },
    )

    assert snapshot["session_id"] == reloaded_conversation.current_session.session_id
    assert snapshot["conversation_state"]["active_task"] == user_input
    assert snapshot["last_persistence_status"]["ok"] is True
