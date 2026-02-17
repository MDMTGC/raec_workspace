"""Intent-to-handler routing helpers for RAEC turn processing."""

from __future__ import annotations

from dataclasses import dataclass

from conversation.intent_classifier import Intent


@dataclass(frozen=True)
class TurnRoute:
    """Resolved route for a single user turn."""

    handler_name: str
    turn_mode: str
    selected_mode: str | None = None


class TurnRouter:
    """Deterministic mapping from intent + requested mode to handler route."""

    def route(self, intent: Intent, requested_mode: str) -> TurnRoute:
        if intent == Intent.CHAT:
            return TurnRoute(handler_name="chat", turn_mode="chat")
        if intent == Intent.QUERY:
            return TurnRoute(handler_name="query", turn_mode="analysis")
        if intent == Intent.META:
            return TurnRoute(handler_name="meta", turn_mode="meta")

        selected_mode = requested_mode if requested_mode != "auto" else "standard"
        return TurnRoute(handler_name="task", turn_mode="task", selected_mode=selected_mode)
