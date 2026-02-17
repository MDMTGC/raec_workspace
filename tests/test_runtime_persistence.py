from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.persistence import SessionPersistenceCoordinator


class DummyStore:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def save(self) -> None:
        self.calls += 1
        if self.fail:
            raise RuntimeError("save failed")


def test_persistence_coordinator_orders_saves() -> None:
    conversation = DummyStore()
    conversation_state = DummyStore()
    identity = DummyStore()

    coordinator = SessionPersistenceCoordinator()
    result = coordinator.persist(
        conversation=conversation,
        conversation_state=conversation_state,
        identity=identity,
    )

    assert result.ok is True
    assert result.order == ["conversation", "conversation_state", "identity"]
    assert conversation.calls == 1
    assert conversation_state.calls == 1
    assert identity.calls == 1


def test_persistence_coordinator_collects_failures() -> None:
    coordinator = SessionPersistenceCoordinator()
    result = coordinator.persist(
        conversation=DummyStore(),
        conversation_state=DummyStore(fail=True),
        identity=DummyStore(),
    )

    assert result.ok is False
    assert ("conversation_state", "save failed") in result.failures
