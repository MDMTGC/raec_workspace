# RAEC Scaffolding Weakness Analysis & Improvement Plan

## Fixes Already Applied (This Session)

### Initial Bug Fixes
1. **`parse_json` parameter mismatch** - Now accepts `**_kwargs` to gracefully ignore unexpected params
2. **Missing HTML extraction tool** - Added `web.extract_text` and `web.extract_links` to WebTools
3. **Verification system blind to plan failures** - Added `_verify_plan_result()` that checks step completion rates, failed/blocked counts, and the plan's own success flag
4. **Planner tool signature awareness** - Enhanced `get_tools_for_llm()` to show REQUIRED markers, default values, and types. Updated planning prompt to warn against inventing parameters and to mandate `web.extract_text` after `web.http_get`
5. **No failure recovery** - Added LLM fallback when tool steps fail or dependencies are unmet, with `recovery_attempts` tracking

### Phase 1: Core Reliability (COMPLETED)
6. **W2 - Plan validation pass** - `_validate_params()` now inspects actual tool signatures via `inspect.signature()`, strips invalid params (e.g., `pretty` from `parse_json`), and warns about missing required params before execution begins. Invalid tools fall back to LLM reasoning.
7. **W3/W9 - Composite web tools** - Added `web.fetch_text`, `web.fetch_json`, `web.fetch_links` as single-step tools that handle fetch→parse internally. Planner prompt updated to prefer these over raw `http_get`. This eliminates the #1 failure mode (HTML piped into parse_json).
8. **W10 - Return type annotations** - Updated `get_tools_for_llm()` to show `-> ReturnType` for every tool with a return annotation. Docstrings updated to describe output types. Planner prompt updated with rule: "Ensure the output type of one step matches the expected input type of the next step."

### Phase 2: Adaptive Execution (COMPLETED)
9. **W1 - Adaptive re-planning** - `run()` now wraps execution in a retry loop (max 2 replans). If completion rate < 80%, errors are fed back to the LLM via `_generate_recovery_plan()` which uses `ERROR_RECOVERY` task type and explicitly tells the LLM to avoid the same failing approaches.
10. **W4 - Retry with param correction** - `_retry_with_corrected_params()` asks the LLM to fix tool parameters before falling back to pure reasoning. Only triggers for retryable errors (param issues, HTTP errors). Uses `PARAM_GENERATION` task type for fast 3B routing. Falls through to LLM reasoning fallback if retry also fails.
11. **W6 - Per-step verification** - `_quick_verify_step_output()` runs after every successful tool execution. Catches: None/empty output, error-string-as-value (e.g. `"HTTP GET error: 400"`), unexpected HTML in non-HTML tools, error dicts. Warnings are tracked in results without blocking execution.

---

## Scaffolding Weaknesses Identified

### CRITICAL: The Planner Is the Weakest Link

The planner is the brain of the system, but it has fundamental structural problems:

#### W1. Planner has no feedback loop with tool results
**Problem:** The planner generates a static plan upfront, then execution is a blind sequential walk. When step 1 fails (bad arXiv URL), steps 2-3 are simply blocked. The planner never gets a chance to say "OK, that approach didn't work, let me try a different URL or a different source entirely."

**Current mitigation:** We added LLM fallback for failed steps, but this is reactive patching — the LLM reasoning step doesn't have access to the tool registry and can't generate alternative tool-based plans.

**Proposed fix: Adaptive re-planning.** After N consecutive failures (or when completion rate drops below threshold), pause execution and send the partial results + errors back to the planner LLM to generate a recovery sub-plan. This is standard in modern agent frameworks (ReAct loop pattern).

#### W2. Planner doesn't validate its own output
**Problem:** The planner generates tool calls with parameters like `pretty=True` that don't exist, URLs with wrong API syntax, and `data.parse_json` for HTML content. It has no self-checking step.

**Proposed fix: Plan validation pass.** After generating the plan, run a validation step that:
- Checks every TOOL line against the actual tool registry
- Checks every PARAMS dict against the tool's actual `inspect.signature()`
- Rejects plans with invalid tools/params and regenerates with error feedback
- This is essentially "compile before execute" — cheap and high-value

#### W3. No content-type awareness in web pipeline
**Problem:** `web.http_get` returns raw content. The planner doesn't know if it's JSON, HTML, XML, or plain text, and blindly pipes it into `data.parse_json`. This caused the HuggingFace HTML → parse_json failure.

**Proposed fix:**
- Option A: Make `web.http_get` return a structured result with content-type header info
- Option B: Add a `web.detect_content_type` tool
- Option C (recommended): Add a `web.fetch_and_parse` high-level tool that auto-detects content type and returns parsed data (JSON → dict, HTML → extracted text, XML → dict)

#### W4. No max retry / error budget
**Problem:** When the arXiv API returns 400, RAEC doesn't retry with a corrected URL. The recovery fallback we added uses LLM reasoning, but that's not tool-backed and produces low-quality results for data retrieval tasks.

**Proposed fix:** Add a configurable retry strategy per tool category:
- `web.*` tools: retry with exponential backoff (network errors)
- Tool param errors: retry once with corrected params (via LLM)
- Hard failures: fall back to LLM reasoning

---

### HIGH: Verification System Gaps

#### W5. Verification checks syntax/patterns but not task completion
**Problem (now partially fixed):** The old verification ran regex patterns on the output string. A plan dict with `success: False` would still pass because the string `{'success': False, 'steps': [...]}` didn't match error patterns like `error: ` or `traceback`.

**Status:** Fixed for plan results. But verification still doesn't check whether the **task objective** was actually achieved — only whether the output looks error-free.

**Proposed fix: Task-objective verification.** Use the LLM to check: "Given the original task '{task}' and the execution results, was the task objective actually achieved?" This is the SEMANTIC verification level — currently it exists but is only used when explicitly requested, and it's not included in standard mode.

#### W6. Verification runs too late
**Problem:** Verification only runs after the entire plan completes. If step 1 of 7 fails catastrophically, 6 more steps still execute (or block), wasting time and LLM tokens.

**Proposed fix: Per-step verification option.** In incremental mode, each step is already verified. Bring lightweight per-step checks into standard mode: after each tool execution, run a quick check (did the tool succeed? is the output non-empty? is it the right type for the next step's input?).

---

### MEDIUM: Orchestration & Agent Architecture

#### W7. Multi-agent mode doesn't use the ToolEnabledPlanner
**Problem:** Collaborative mode uses `MultiAgentOrchestrator._execute_standard_workflow()`, which has the planner agent generate a plan via LLM text, then the executor agent tries to execute it — but the executor only has rudimentary single-tool-call parsing. It doesn't use `ToolEnabledPlanner` with its multi-step plan execution, parameter resolution, and dependency tracking.

**Result:** Collaborative mode is significantly less capable than standard mode for actual tool-based tasks. It's good for reasoning-only tasks but weak for execution.

**Proposed fix:** Have the executor agent in collaborative mode delegate to `ToolEnabledPlanner.run()` for tool-based execution, so it inherits all the plan parsing, dependency resolution, and recovery logic.

#### W8. ActionExecutor bypasses the full pipeline
**Problem:** `action_executor.py` is a fast-path that pattern-matches tasks and calls tools directly, bypassing planning, memory, skills, and verification. It's good for speed but means frequent actions (like file creation) don't get verified, don't build skills, and don't store experiences.

**Proposed fix:** After ActionExecutor completes, optionally run lightweight post-execution hooks: store experience in memory, check if the result matches a skill pattern, and run basic verification. This preserves the speed advantage while still learning.

---

### MEDIUM: Tool System Gaps

#### W9. No tool composition / pipelines
**Problem:** Tools are atomic — each does one thing. Common multi-step patterns (fetch → extract → search) must be manually chained by the planner. The planner frequently gets this wrong.

**Proposed fix: Composite tools / tool pipelines.** Define common tool chains as first-class composite tools:
- `web.fetch_text(url)` = `http_get` → `extract_text`
- `web.fetch_json(url)` = `http_get` → `parse_json` (with error handling)
- `file.read_and_search(filepath, pattern)` = `read_file` → `search_text`

These reduce planner complexity and eliminate the most common failure mode (wrong intermediate piping).

#### W10. Tools don't report expected output types
**Problem:** When the planner chains tools, it doesn't know that `web.http_get` returns a string (HTML), `data.parse_json` expects a JSON string and returns a dict, `data.filter_list` expects a list. Type mismatches cause silent failures.

**Proposed fix:** Add return type annotations to tools and expose them in `get_tools_for_llm()`. The planner prompt should include input→output type info so the LLM can reason about compatibility.

---

### LOW: Memory & Learning Gaps

#### W11. Failed executions don't contribute to learning
**Problem:** When a plan fails, the failure is stored as an experience, but the **reason** for failure isn't analyzed or stored as a structured lesson. Next time a similar task comes in, RAEC will retrieve the past failure experience but won't know how to avoid the same mistake.

**Proposed fix: Failure analysis and belief formation.** On plan failure:
1. Analyze the error chain (which step failed, why, what was the cascade)
2. Store as a BELIEF: "arXiv API requires bracket date range syntax" or "HuggingFace blog returns HTML, not JSON"
3. Future plans can query these beliefs to avoid known pitfalls

#### W12. Skill extraction threshold is too coarse
**Problem:** Skills are only extracted from fully successful plan completions. Partial successes (6/7 steps worked) don't produce skills, even though the working steps may contain valuable patterns.

**Proposed fix:** Extract skills from individual successful steps, not just complete plans. A step that successfully fetches and parses web content is a reusable pattern regardless of whether the overall plan succeeded.

---

## Improvement Priority & Sequencing

### Phase 1: Core Reliability — COMPLETED
1. **W2 - Plan validation pass** — Catches bad tool calls before execution
2. **W3/W9 - Composite web tools** — `web.fetch_text()`, `web.fetch_json()` eliminate the #1 failure mode
3. **W10 - Return type annotations** — Helps planner chain tools correctly

### Phase 2: Adaptive Execution — COMPLETED
4. **W1 - Adaptive re-planning** — ReAct-style loop: plan → execute → observe → replan
5. **W4 - Retry strategy** — Per-tool-category retry with param correction
6. **W6 - Per-step verification** — Catch failures early in standard mode

### Phase 3: Learning & Integration — COMPLETED
7. **W11 - Failure analysis** — Turn failures into beliefs for future avoidance
8. **W5 - Task-objective verification** — Enable semantic check in standard mode
9. **W7 - Collaborative mode tools** — Wire ToolEnabledPlanner into multi-agent executor
10. **W8 - ActionExecutor hooks** — Post-execution learning for fast-path actions
11. **W12 - Step-level skill extraction** — Learn from partial successes
