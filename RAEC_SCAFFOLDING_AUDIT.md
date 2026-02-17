# RAEC Scaffolding Audit

## Purpose

Provide a full, current-state audit of RAEC scaffolding with concrete findings, risk ranking, and a forward execution roadmap.

## Audit Scope

- Core runtime orchestration and lifecycle (`main.py`)
- Planning + execution scaffolding (`planner/`, `tools/`, `agents/`, `evaluators/`)
- Memory and persistence (`memory/`, `conversation/`, `identity/`)
- Interface-layer continuity and observability (`conversation/conversation_state_manager.py`, `observability/prompt_debug.py`)
- Adjacent agency subsystems (goals, preferences, self-eval, context, uncertainty, curiosity)
- Documentation alignment (`RAEC.md`, `SCAFFOLDING_ANALYSIS.md`, backlog/schema docs)

## Method

1. Static architecture review of key modules and wiring points.
2. Regression check with full local tests (`pytest -q`).
3. Consistency pass on scaffolding interfaces and persistence boundaries.

## Current State Snapshot

### What is solid

- **Continuity layer is implemented and wired** into `process_input(...)` with per-turn updates, context generation, compression, reset, and persistence.
- **Prompt observability is implemented** via deterministic JSON prompt logs and snapshot export paths.
- **Scaffolding breadth is strong**: planner, tool executor, orchestrator, evaluator, memory tiers, goals/preferences/evaluation/context/uncertainty/curiosity all initialize in one runtime.
- **Baseline tests pass** for continuity, prompt debug, and web modules.

### Major structural strengths

- Local-first architecture and modular package layout are preserved.
- Deterministic state serialization is present for both conversation manager and conversation state manager.
- Explicit intent routing (`chat/query/task/meta`) is integrated and readable.

## Findings (Risk-ranked)

### Critical

1. **No critical runtime blockers found in this audit pass.**

### High

1. **`main.py` remains a high-concentration integration surface (~1500 lines)**.
   - Impact: difficult change isolation, higher regression probability, and slower feature velocity.
   - Recommendation: split runtime into composable services (`runtime/bootstrap.py`, `runtime/router.py`, `runtime/tasks.py`, `runtime/meta.py`) with thin entrypoint orchestration.

2. **Scaffolding maturity is uneven across subsystems**.
   - Continuity/observability paths now have targeted tests, but several adjacent agency subsystems are lightly validated in integration terms.
   - Recommendation: add high-value integration tests that exercise cross-subsystem turn flows (intent → planner → evaluator → memory/summary persistence).

### Medium

1. **Conversation state model previously contained misplaced behavior in `Turn`**.
   - A thread-reset method existed on `Turn` (entity-level type) even though reset semantics belong to `ConversationState`.
   - This audit removed that misplaced method to tighten data-model boundaries.

2. **Documentation drift risk remains** between implementation and high-level docs.
   - `SCAFFOLDING_ANALYSIS.md` reports completion phases that should be periodically verified against executable behavior via explicit “evidence links” (tests/commands).
   - Recommendation: introduce a lightweight `docs/verification_matrix.md` mapping each claim to a test command.

### Low

1. **Dual persistence concepts for conversation** (`ConversationManager` and `ConversationStateManager`) are valid but require explicit ownership docs.
   - Recommendation: add a short contract section clarifying responsibilities and write-order guarantees.

## Projected Roadmap (From This Point)

### Phase A — Stabilize Integration Surface (next 1–2 milestones)

1. **Refactor monolith boundaries in `main.py`**
   - Extract request routing and handlers into dedicated modules.
   - Keep `Raec` as composition root only.

2. **Add integration tests for full turn lifecycle**
   - Include continuity update + summary persistence + prompt debug artifact assertions.

3. **Define persistence contracts**
   - Document and test ordering/idempotence for `conversation.save()`, `conversation_state.save()`, `identity.save()`.

### Phase B — Interface Optimization (backlog-aligned)

4. **ResponseSynthesizer layer**
   - Separate internal reasoning outputs from conversational rendering.

5. **Mode-awareness hardening**
   - Formal switching rules for chat/task/analysis/meta and deterministic mode transitions.

6. **Context budget policy**
   - Token-aware priority ordering for identity/memory/state/history slices.

### Phase C — Reliability + Learning Depth

7. **Failure-trace diff logging**
   - On continuity misses or failed plans, store compact prompt/context deltas.

8. **Belief-level failure learning loop**
   - Convert recurring failures into structured BELIEF records for planner retrieval.

9. **Cross-subsystem health report command**
   - Add an operator-facing diagnostics command that reports readiness, recent failures, and persistence freshness.

## Immediate Next Action

Start with **Phase A.1** (runtime decomposition of `main.py`) and pair it with **Phase A.2** tests in the same milestone to avoid architecture-only drift.
