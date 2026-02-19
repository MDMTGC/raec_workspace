# RAEC Roadmap Status Update

## Purpose

Record current verified wiring status, identify active bugs/gaps, and define the next execution sequence.

## Verification Snapshot

- Continuity state manager is initialized and fed through turn processing.
- Deterministic turn routing is active via `TurnRouter`.
- Deterministic multi-store persistence is active via `SessionPersistenceCoordinator`.
- Snapshot generation is wired through `SessionSnapshotBuilder` and includes persistence status.
- Prompt debug logging is wired and can export assembled prompts.
- Test suite currently validates continuity, routing, persistence, snapshot observability, curiosity normalization, and process-input lifecycle boundaries.

## Bugs / Gaps Identified

1. **Reset lifecycle repopulation bug (fixed in this milestone)**
   - Previously, `/reset` would clear conversation state and then immediately re-add the reset turn through normal finalization, undermining the reset semantics.
   - Mitigation: short-circuit turn finalization after `/reset` meta execution.

2. **Error visibility gap on state-manager load failures (fixed)**
   - `ConversationStateManager._load_or_create(...)` now records `last_load_error` and emits warning output on malformed/unreadable state files while falling back safely.

3. **Cross-subsystem integration depth still shallow (partially addressed)**
   - Existing integration tests use lightweight stubs around `Raec.__new__` for orchestration boundaries, and a real persisted roundtrip integration test has been added for `ConversationManager` + `ConversationStateManager` + `SelfModel` with `SessionPersistenceCoordinator`.
   - Remaining gap: add full `Raec` runtime end-to-end flow coverage once model/tool external dependencies are isolated for deterministic tests.

## Updated Next Steps (Priority Order)

1. **P1: Expand integration realism**
   - Add a bounded end-to-end test that exercises real `ConversationManager` + `ConversationStateManager` save/reload sequence around turn processing.

2. **P1: Add snapshot contract checks**
   - Validate snapshot schema stability (required keys/types) to avoid tooling regressions.

3. **P2: Context-budget instrumentation**
   - Add prompt-component size accounting into observability output (identity/memory/state/history breakdown).

4. **P2: Curiosity refinement continuation**
   - Add investigation success/failure taxonomy metrics and expose in curiosity status output.

## Completion Criteria for Next Milestone

- State load failures are diagnosable from logs/snapshots.
- At least one real persisted end-to-end turn lifecycle test passes. ✅
- Snapshot schema has explicit regression checks.
