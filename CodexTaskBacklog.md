# RAEC Codex Task Backlog

## Phase 1 — Observability

1. Map Prompt Assembly
   - Trace full flow from process_input() to llm_interface
   - Document identity injection
   - Document memory injection
   - Document conversation history reconstruction

2. Add Debug Mode
   - Log fully assembled prompt before model call
   - Add optional JSON export of prompt components

3. Add Session Snapshot Tool
   - Dump active session state
   - Dump memory retrieval context
   - Dump last N conversation turns

---

## Phase 2 — Conversation State Layer

4. Design ConversationState class
   - Fields:
     - active_thread_id
     - active_task
     - unresolved_references
     - rolling_summary
     - mode (chat/task/meta)
     - last_response_commitments
   - Methods:
     - update_from_turn()
     - generate_prompt_context()
     - compress_history()

5. Integrate ConversationState into process_input()
   - Initialize per session
   - Update after each turn
   - Inject into prompt assembly

6. Add Rolling Summary Compression
   - After N turns, compress older turns into summary
   - Store in memory as SUMMARY type
   - Preserve recent raw turns

---

## Phase 3 — Interface Optimization

7. Separate Chat Rendering from Core Reasoning
   - Introduce ResponseSynthesizer layer
   - Convert planner output into conversational output
   - Remove identity dominance from each turn

8. Introduce Mode Awareness
   - Chat mode
   - Task mode
   - Analytical mode
   - Explicit switching rules

9. Add Continuity Tests
   - Multi-turn coherence
   - Reference resolution (“that”, “earlier”)
   - Active task carryover

---

## Phase 4 — Performance & Stability

10. Optimize Context Size
    - Token budget accounting
    - Priority ordering of context injection

11. Add Conversation Reset Rules
    - Time-based reset
    - Explicit user reset command
    - Context overflow reset

12. Add Failure Trace Logging
    - When continuity fails, log prompt diff

