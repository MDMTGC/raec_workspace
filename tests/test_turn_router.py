from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conversation.intent_classifier import Intent
from runtime.turn_router import TurnRouter


def test_turn_router_routes_non_task_intents() -> None:
    router = TurnRouter()

    chat_route = router.route(Intent.CHAT, requested_mode="auto")
    query_route = router.route(Intent.QUERY, requested_mode="auto")
    meta_route = router.route(Intent.META, requested_mode="auto")

    assert chat_route.handler_name == "chat"
    assert chat_route.turn_mode == "chat"
    assert query_route.handler_name == "query"
    assert query_route.turn_mode == "analysis"
    assert meta_route.handler_name == "meta"
    assert meta_route.turn_mode == "meta"


def test_turn_router_resolves_task_mode_from_auto() -> None:
    router = TurnRouter()
    route = router.route(Intent.TASK, requested_mode="auto")

    assert route.handler_name == "task"
    assert route.turn_mode == "task"
    assert route.selected_mode == "standard"


def test_turn_router_resolves_task_mode_from_explicit_mode() -> None:
    router = TurnRouter()
    route = router.route(Intent.TASK, requested_mode="collaborative")

    assert route.handler_name == "task"
    assert route.selected_mode == "collaborative"
