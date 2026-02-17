# RAEC Session Persistence Contract

## Purpose

Define deterministic ownership and save ordering for session-bound state to prevent drift and partial-write ambiguity.

## Owned Stores

1. `ConversationManager`
   - Owns message history and session summary state (`conversation/conversation_state.json`).
2. `ConversationStateManager`
   - Owns continuity thread/task/reference state (`conversation/conversation_state_manager.json`).
3. `SelfModel`
   - Owns identity and reflection state (`identity/identity.json`).

## Save Order

`conversation -> conversation_state -> identity`

This order is enforced by `SessionPersistenceCoordinator`.

## Call Sites

`Raec._persist_runtime_state(reason)` is the only intended runtime path for multi-store saves and is currently used on:
- turn completion,
- reset flow,
- shutdown flow.

## Failure Semantics

- Coordinator attempts all targets in order.
- Failures are collected in `PersistenceResult.failures` and surfaced as warnings.
- Partial success is represented by `PersistenceResult.ok == False` with successful store names retained in `PersistenceResult.order`.

## Testing

- `tests/test_runtime_persistence.py` verifies ordering and structured failure collection.
