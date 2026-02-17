from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conversation.intent_classifier import Intent
from main import Raec
from runtime.turn_router import TurnRoute


@dataclass
class DummyUrgency:
    value: int = 0
    name: str = "LOW"


@dataclass
class DummyUserState:
    urgency: DummyUrgency


@dataclass
class DummyContextResult:
    user_state: DummyUserState


@dataclass
class DummyClassification:
    intent: Intent
    confidence: float = 1.0


@dataclass
class DummyConfidence:
    score: float = 0.9


class DummyConversation:
    def __init__(self) -> None:
        self.user_messages: list[str] = []
        self.assistant_messages: list[str] = []
        self.current_session = type("Session", (), {"session_id": "sess-test"})()

    def add_user_message(self, content: str) -> None:
        self.user_messages.append(content)

    def add_assistant_message(self, content: str) -> None:
        self.assistant_messages.append(content)


class DummyState:
    def __init__(self, compress_result: bool) -> None:
        self.compress_result = compress_result
        self.updates: list[tuple[str, str, str]] = []

    def generate_prompt_context(self) -> str:
        return "ConversationState: test"

    def update_from_turn(self, user_input: str, assistant_output: str, mode: str) -> None:
        self.updates.append((user_input, assistant_output, mode))

    def compress_history(self, max_recent_turns: int = 6) -> bool:
        return self.compress_result


class DummyConversationStateManager:
    def __init__(self, compress_result: bool) -> None:
        self.state = DummyState(compress_result=compress_result)


class DummyContextManager:
    def update(self, _user_input: str) -> DummyContextResult:
        return DummyContextResult(user_state=DummyUserState(urgency=DummyUrgency()))


class DummyIdentity:
    def __init__(self) -> None:
        self.interactions = 0

    def record_interaction(self) -> None:
        self.interactions += 1


class DummyPreferences:
    def learn_from_explicit(self, _user_input: str, _llm: Any) -> None:
        return None


class DummyIntentClassifier:
    def __init__(self, intent: Intent) -> None:
        self.intent = intent

    def classify(self, _user_input: str) -> DummyClassification:
        return DummyClassification(intent=self.intent)


class DummyTurnRouter:
    def __init__(self, route: TurnRoute) -> None:
        self.route_result = route

    def route(self, intent: Intent, requested_mode: str) -> TurnRoute:
        assert requested_mode in {"auto", "standard", "collaborative", "incremental"}
        assert intent in {Intent.CHAT, Intent.QUERY, Intent.META, Intent.TASK}
        return self.route_result


class DummyConfidenceTracker:
    def assess_confidence(self, _response: str, task_type: str) -> DummyConfidence:
        assert task_type in {"chat", "query", "meta", "task"}
        return DummyConfidence()

    def should_express_uncertainty(self, _confidence: DummyConfidence) -> bool:
        return False


class DummyCuriosity:
    def analyze_response(self, response: str, user_input: str, session_id: str) -> list[Any]:
        assert response
        assert user_input
        assert session_id == "sess-test"
        return []


def _build_lightweight_raec(intent: Intent, route: TurnRoute, compress_result: bool) -> Raec:
    raec = Raec.__new__(Raec)
    raec.context = DummyContextManager()
    raec.conversation = DummyConversation()
    raec.identity = DummyIdentity()
    raec.preferences = DummyPreferences()
    raec.llm = object()
    raec.intent_classifier = DummyIntentClassifier(intent=intent)
    raec.conversation_state = DummyConversationStateManager(compress_result=compress_result)
    raec.turn_router = DummyTurnRouter(route=route)
    raec.confidence = DummyConfidenceTracker()
    raec.curiosity = DummyCuriosity()
    raec.idle_loop = type("IdleLoop", (), {"record_user_activity": lambda self: None})()

    raec._persist_calls = []
    raec._summary_calls = 0

    def _persist_runtime_state(reason: str) -> None:
        raec._persist_calls.append(reason)

    def _store_conversation_summary() -> None:
        raec._summary_calls += 1

    raec._persist_runtime_state = _persist_runtime_state
    raec._store_conversation_summary = _store_conversation_summary
    raec._handle_chat = lambda user_input, state_context="": f"chat:{user_input}:{state_context}"
    raec._handle_query = lambda user_input, state_context="": f"query:{user_input}:{state_context}"
    raec._handle_meta = lambda user_input, state_context="": f"meta:{user_input}:{state_context}"
    raec._handle_task = lambda user_input, selected_mode, state_context="": (
        f"task:{selected_mode}:{user_input}:{state_context}"
    )
    return raec


def test_process_input_routes_task_updates_state_and_persists() -> None:
    route = TurnRoute(handler_name="task", turn_mode="task", selected_mode="standard")
    raec = _build_lightweight_raec(intent=Intent.TASK, route=route, compress_result=False)

    result = raec.process_input("Build a parser", mode="auto")

    assert result.startswith("task:standard:Build a parser")
    assert raec.conversation.user_messages == ["Build a parser"]
    assert len(raec.conversation.assistant_messages) == 1
    assert raec.conversation_state.state.updates == [
        ("Build a parser", result, "task"),
    ]
    assert raec._summary_calls == 0
    assert raec._persist_calls == ["turn_complete"]


def test_process_input_stores_summary_when_compression_triggers() -> None:
    route = TurnRoute(handler_name="chat", turn_mode="chat")
    raec = _build_lightweight_raec(intent=Intent.CHAT, route=route, compress_result=True)

    _ = raec.process_input("hello", mode="auto")

    assert raec._summary_calls == 1
    assert raec._persist_calls == ["turn_complete"]


def test_process_input_routes_meta_with_meta_mode() -> None:
    route = TurnRoute(handler_name="meta", turn_mode="meta")
    raec = _build_lightweight_raec(intent=Intent.META, route=route, compress_result=False)

    result = raec.process_input("/status", mode="auto")

    assert result.startswith("meta:/status")
    assert raec.conversation_state.state.updates == [("/status", result, "meta")]
    assert raec._persist_calls == ["turn_complete"]


def test_reset_conversation_state_clears_buffers_and_persists() -> None:
    raec = Raec.__new__(Raec)

    class _Conversation:
        def __init__(self) -> None:
            self.cleared = False

        def clear_history(self) -> None:
            self.cleared = True

    class _State:
        def __init__(self) -> None:
            self.reset_called = False
            self.keep_rolling_summary = True

        def reset(self, keep_rolling_summary: bool) -> None:
            self.reset_called = True
            self.keep_rolling_summary = keep_rolling_summary

    raec.conversation = _Conversation()
    raec.conversation_state = type("StateMgr", (), {"state": _State()})()
    raec._last_memory_context = [{"foo": "bar"}]
    raec._last_stored_rolling_summary = "summary"
    persisted: list[str] = []
    raec._persist_runtime_state = lambda reason: persisted.append(reason)

    message = raec._reset_conversation_state()

    assert message == "Conversation context reset for this session."
    assert raec.conversation.cleared is True
    assert raec.conversation_state.state.reset_called is True
    assert raec.conversation_state.state.keep_rolling_summary is False
    assert raec._last_memory_context == []
    assert raec._last_stored_rolling_summary == ""
    assert persisted == ["reset"]


def test_route_turn_uses_router_and_returns_turn_mode() -> None:
    raec = Raec.__new__(Raec)
    raec.turn_router = DummyTurnRouter(route=TurnRoute(handler_name="query", turn_mode="analysis"))
    raec._handle_query = lambda user_input, state_context="": f"query:{user_input}:{state_context}"

    response, turn_mode = raec._route_turn(
        intent=Intent.QUERY,
        requested_mode="auto",
        user_input="What is RAEC?",
        state_context="ConversationState: test",
    )

    assert response.startswith("query:What is RAEC?:ConversationState: test")
    assert turn_mode == "analysis"


def test_finalize_turn_runs_post_processing_and_persists() -> None:
    route = TurnRoute(handler_name="chat", turn_mode="chat")
    raec = _build_lightweight_raec(intent=Intent.CHAT, route=route, compress_result=False)

    raec._finalize_turn(
        user_input="hello",
        response="chat:hello:ConversationState: test",
        turn_mode="chat",
        task_type="chat",
    )

    assert raec.conversation_state.state.updates == [
        ("hello", "chat:hello:ConversationState: test", "chat"),
    ]
    assert raec.conversation.assistant_messages == ["chat:hello:ConversationState: test"]
    assert raec._persist_calls == ["turn_complete"]
