from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conversation.conversation_state_manager import ConversationStateManager


def test_state_tracks_task_and_commitments(tmp_path: Path):
    state_path = tmp_path / "conversation_state.json"
    manager = ConversationStateManager(state_path=str(state_path), session_id="sess-1")

    manager.state.update_from_turn(
        user_input="Please create a test script",
        assistant_output="I will write the script next.",
        mode="task",
    )

    assert manager.state.active_task == "Please create a test script"
    assert manager.state.mode == "task"
    assert len(manager.state.last_commitments) == 1


def test_state_compresses_history_and_persists(tmp_path: Path):
    state_path = tmp_path / "conversation_state.json"
    manager = ConversationStateManager(state_path=str(state_path), session_id="sess-2")

    for idx in range(8):
        manager.state.update_from_turn(
            user_input=f"turn {idx}",
            assistant_output=f"reply {idx}",
            mode="chat",
        )

    compressed = manager.state.compress_history(max_recent_turns=3)
    assert compressed is True
    manager.save()

    reloaded = ConversationStateManager(state_path=str(state_path), session_id="sess-2")
    assert len(reloaded.state.recent_turns) == 3
    assert "turn 0" in reloaded.state.rolling_summary
    prompt_context = reloaded.state.generate_prompt_context()
    assert "ConversationState:" in prompt_context
    assert "active_task" in prompt_context


def test_compress_noop_when_under_limit(tmp_path: Path):
    state_path = tmp_path / "conversation_state.json"
    manager = ConversationStateManager(state_path=str(state_path), session_id="sess-3")

    manager.state.update_from_turn(user_input="one", assistant_output="two", mode="chat")
    compressed = manager.state.compress_history(max_recent_turns=5)

    assert compressed is False
    assert manager.state.rolling_summary == ""


def test_reference_resolution_with_active_task(tmp_path: Path):
    state_path = tmp_path / "conversation_state.json"
    manager = ConversationStateManager(state_path=str(state_path), session_id="sess-4")

    manager.state.update_from_turn(
        user_input="Create a deployment checklist for RAEC",
        assistant_output="I will draft it.",
        mode="task",
    )
    manager.state.update_from_turn(
        user_input="Can you refine that with rollback steps?",
        assistant_output="Done.",
        mode="chat",
    )

    assert manager.state.active_task == "Create a deployment checklist for RAEC"
    assert manager.state.unresolved_references == []


def test_unresolved_reference_when_no_active_task(tmp_path: Path):
    state_path = tmp_path / "conversation_state.json"
    manager = ConversationStateManager(state_path=str(state_path), session_id="sess-5")

    manager.state.update_from_turn(
        user_input="Can you expand on that?",
        assistant_output="Please clarify.",
        mode="chat",
    )

    assert "that" in manager.state.unresolved_references
