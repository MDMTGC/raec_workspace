"""Deterministic persistence ordering for session-bound state."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, List, Tuple


@dataclass
class PersistenceResult:
    """Outcome of a persistence pass across session stores."""

    ok: bool
    order: List[str]
    failures: List[Tuple[str, str]] = field(default_factory=list)
    persisted_at: float = field(default_factory=time.time)


class SessionPersistenceCoordinator:
    """Persist stateful session components in deterministic order."""

    def __init__(self, save_order: Tuple[str, ...] = ("conversation", "conversation_state", "identity")) -> None:
        self.save_order = save_order

    def persist(self, *, conversation: Any, conversation_state: Any, identity: Any) -> PersistenceResult:
        """Persist all stores according to configured order.

        Errors are captured and returned so callers can decide whether to continue.
        """
        targets = {
            "conversation": conversation,
            "conversation_state": conversation_state,
            "identity": identity,
        }

        failures: List[Tuple[str, str]] = []
        successful_order: List[str] = []

        for name in self.save_order:
            target = targets.get(name)
            if target is None:
                failures.append((name, "missing_target"))
                continue

            try:
                target.save()
                successful_order.append(name)
            except Exception as exc:  # deliberate boundary capture for persistence contract
                failures.append((name, str(exc)))

        return PersistenceResult(
            ok=not failures,
            order=successful_order,
            failures=failures,
        )
