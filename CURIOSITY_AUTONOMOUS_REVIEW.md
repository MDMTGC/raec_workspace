# Curiosity Engine / Autonomous Scaffolding Review

## Purpose

Review the current curiosity and idle-loop scaffolding, identify expansion/refinement opportunities, and record immediate improvements applied in this milestone.

## Current Strengths

- Persistent curiosity queue with priority-aware retrieval (`QuestionQueue`).
- Automatic uncertainty/knowledge-gap extraction from assistant outputs (`CuriosityEngine.analyze_response`).
- Autonomous idle investigation loop with safety controls: idle threshold, interval, and per-session cap (`IdleLoop`).
- Full transparency hooks via callbacks and web-activity logging integration in runtime.

## Gaps / Refinement Opportunities

1. **Question normalization drift**
   - LLM-extracted questions may include list prefixes, quotes, multiline output, or non-question phrasing.
   - Impact: duplicate/noisy queue entries and weaker search quality.

2. **Autonomous quality scoring**
   - Current queue selection is priority+age only; source reliability/confidence is not factored.

3. **Adaptive investigation budgeting**
   - Session-level max investigations is static; should eventually adapt to user mode (task-heavy vs exploratory).

4. **Post-investigation utility scoring**
   - Findings are stored, but utility-to-user (actionable vs informational) is not currently tracked.

## Refinements Applied Now

- Added deterministic question normalization in `CuriosityEngine`:
  - trims whitespace and quote wrappers,
  - normalizes multiline outputs to single-line questions,
  - enforces question-mark termination,
  - converts non-interrogative fragments into a searchable question form,
  - caps length for queue consistency.
- Routed uncertainty and knowledge-gap extraction through the normalization path.
- Normalized generated ambient questions before enqueue/usage.

## Expansion Roadmap (Recommended)

### Near-term

1. Add source-quality metadata to question records.
2. Add confidence score on investigation findings.
3. Add operator command for curiosity diagnostics (queue quality + investigate success ratio).

### Mid-term

4. Adaptive curiosity budget policy tied to active mode/task urgency.
5. Topic-level dedupe with semantic similarity (not exact-string only).
6. Failure taxonomy for investigation errors (network, parse, synthesis, source-quality).

### Longer-term

7. Tie curiosity outcomes into belief formation and planner retrieval weighting.
8. Learn user-specific curiosity preferences (domains to prioritize/suppress).
