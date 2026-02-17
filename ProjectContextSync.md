# Project Context Sync (Codex)

This note captures the current shared understanding from:
- `AGENTS.md`
- `CodexTaskBacklog.md`
- `ConversationContinuitySchema.md`
- supporting architecture docs (`RAEC.md`, `SCAFFOLDING_ANALYSIS.md`)

## Core framing

RAEC is a **local-first cognitive system** (not a chatbot persona) with strong emphasis on deterministic scaffolding, modularity, and inspectable prompt assembly.

The highest-priority architecture issue is **conversation continuity**, framed as an interface-layer state management gap rather than a model-quality issue.

## Immediate priority track

1. **Observability first**
   - trace prompt assembly end-to-end (`process_input()` → LLM call)
   - make identity/memory/history injection explicit
   - add debug output and session snapshots

2. **Conversation State Layer**
   - implement `ConversationState` with session metadata + thread/task/reference state
   - integrate into per-session `process_input()` lifecycle
   - inject generated prompt context deterministically

3. **History compression**
   - keep recent raw turns
   - compress older turns into rolling summaries
   - store summaries as memory type `SUMMARY`

4. **Continuity validation**
   - add tests for multi-turn coherence, reference resolution, and active-task carryover

## Proposed implementation order (practical)

1. Add prompt assembly tracing hooks (read-only diagnostics).
2. Introduce `ConversationState` data model + serializer.
3. Wire `ConversationState` through conversation manager lifecycle.
4. Add rolling-summary compressor and trigger policy.
5. Add continuity tests and regression fixtures.

## Definition of done for continuity milestone

- Session retains active thread/task across turns.
- Deictic references ("that", "earlier", "it") resolve using tracked state.
- Prompt context contains deterministic, inspectable state payload.
- Context window stays bounded via rolling summary compaction.
- All new behavior covered by pytest tests.

## Notes

This file is a working coordination artifact for coding sessions and can be updated as the backlog is executed.
