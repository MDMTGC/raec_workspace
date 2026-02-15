# RAEC - Reflective Agentic Ecosystem Composer

## What RAEC Is

A local LLM swarm + autonomous agent system (~17K lines Python) that plans, executes, learns, and explores on its own. Runs on Ollama with DeepSeek R1 32B as the orchestrator model, on an AMD Radeon RX 7900 XT (20GB VRAM, ROCm).

**Owner:** MDMTGC | **Platform:** Windows 11 | **Workspace:** `C:\Users\MDMTGC\Desktop\raec_workspace`

## Quick Start

```bash
# GUI (recommended)
python raec_gui.py

# Headless
python main.py

# As module
from main import Raec
raec = Raec()
result = raec.process_input("your request here")
raec.close()
```

**Prerequisites:** Ollama running (`ollama serve`), `raec:latest` model pulled, `pip install -r requirements.txt`

## Architecture

```
raec_workspace/
├── main.py                 # Entry point, Raec class (1349 lines)
├── raec_gui.py             # GUI with breathing orb, conversation view
├── config.yaml             # Main configuration (device: rocm)
│
├── raec_core/
│   ├── llm_interface.py    # Ollama API wrapper
│   ├── model_swarm.py      # Task-type routing to specialized models
│   └── core_rules.py       # Immutable safety constraints
│
├── planner/
│   └── planner_tools.py    # ToolEnabledPlanner: plan → execute → verify
│
├── tools/
│   ├── core_tools.py       # 50+ tools across 7 categories
│   ├── executor.py         # ToolExecutor with stats tracking
│   └── tool_interface.py   # ToolResult, ToolInterface base
│
├── memory/
│   └── memory_db.py        # FAISS + SQLite hierarchical memory
│
├── skills/
│   └── skill_graph.py      # Audited skill extraction and reuse
│
├── agents/
│   └── orchestrator.py     # Multi-agent Planner→Executor→Critic
│
├── evaluators/
│   └── logic_checker.py    # 5-level verification system
│
├── identity/
│   └── self_model.py       # Persistent identity, traits, reflections
│
├── conversation/
│   ├── conversation_manager.py  # Session tracking, message history
│   └── intent_classifier.py     # CHAT/QUERY/TASK/META routing
│
├── curiosity/              # Autonomous exploration
│   ├── engine.py           # Detects gaps, investigates via web
│   ├── questions.py        # SQLite question queue with priority
│   └── idle_loop.py        # Background thread, fires after 60s idle
│
├── web/                    # Transparent internet access
│   ├── fetcher.py          # URL content retrieval
│   ├── search.py           # DuckDuckGo search
│   └── activity_log.py     # Full activity transparency
│
├── goals/                  # Persistent goal tracking
│   └── goal_manager.py     # SQLite-backed goals with progress
│
├── preferences/            # Learned user preferences
│   └── preference_manager.py
│
├── evaluation/             # Self-evaluation
│   └── self_evaluator.py   # Success rate, calibration, trends
│
├── toolsmith/              # Dynamic tool creation
│   └── tool_forge.py       # LLM generates → tests → deploys tools
│
├── proactive/              # Proactive notifications
│   └── notifier.py         # Session greetings, pending alerts
│
├── context/                # Context awareness
│   └── context_manager.py  # Urgency, mood, time-of-day detection
│
├── uncertainty/            # Confidence tracking
│   └── confidence.py       # Per-response confidence scoring
│
└── config/
    └── swarm_config.json   # Model swarm tier routing
```

## Model Swarm

| Tier | Model | VRAM | Role |
|------|-------|------|------|
| Orchestrator | `deepseek-r1:32b` | 19GB | Reasoning, planning, synthesis, error recovery |
| Coder | `qwen2.5-coder:7b` | ~5GB | Code generation, refactoring |
| Ganglia | `qwen3:4b` / `phi4-mini` | ~3GB | Fast classification, param generation |
| Curator/SSM | `jamba-reasoning:3b` | ~2GB | Memory digestion, summarization |
| Fallback | `qwen3:14b` | ~9GB | When orchestrator under VRAM pressure |

Serial loading on 20GB card. Ollama auto-unloads between model switches.

## Execution Modes

| Mode | Flow | Use When |
|------|------|----------|
| `standard` | Skill check → Plan → Execute → Verify → Learn | Default for most tasks |
| `collaborative` | Planner agent → Executor agent → Critic agent → Revise | Complex multi-step tasks |
| `incremental` | Generate steps → Verify each → Halt on failure | When you need a reasoning trace |

## Autonomy Systems

### Curiosity Engine
- Background thread starts on boot
- After 60s of no user input, investigates a question from the queue
- Every response is scanned for uncertainty patterns → generates questions
- Max 5 investigations per session, 2 min apart
- Findings stored as EXPERIENCE memories and surfaced in GUI chat
- GUI orb turns purple with wandering animation when exploring

### Agency Modules
- **Goals** — Persistent objectives (user-given, inferred, self-improvement)
- **Preferences** — Learns from explicit statements ("I prefer...", "always...", "never...")
- **Self-Evaluation** — Tracks success rate, trend, overconfidence
- **Toolsmith** — Generates, tests, and deploys new tools at runtime
- **Proactive** — Notifications, session greetings, pending alerts
- **Context** — Urgency detection, mood awareness, time-of-day
- **Uncertainty** — Per-response confidence scoring, calibration

## Commands

```
/help         Show all commands
/status       System status (knowledge, agency, performance)
/skills       Learned skills summary
/curiosity    Curiosity engine status
/learned      What RAEC learned autonomously
/questions    Pending investigation questions
/web          Recent web activity log
/goals        Active goals
/preferences  Learned preferences
/performance  Self-evaluation report
/tools        Generated tools
/notifications Pending notifications
/confidence   Calibration report
```

## Scaffolding (Hardened)

All 12 weaknesses from SCAFFOLDING_ANALYSIS.md addressed across 3 phases:

**Phase 1 — Core Reliability:**
- Plan validation (inspect.signature checks before execution)
- Composite web tools (fetch_text/fetch_json/fetch_links — single-step fetch→parse)
- Return type annotations in tool signatures

**Phase 2 — Adaptive Execution:**
- Adaptive re-planning (retry loop, max 2 replans if <80% completion)
- Param correction retry (LLM fixes params before falling back to reasoning)
- Per-step verification (catches error-as-value, empty output, HTML in wrong tools)
- Error content escalation (error strings in output → hard failure, not soft warning)

**Phase 3 — Learning & Integration:**
- Failure→belief formation (extracts lessons from failed plans as BELIEF memories)
- Task-objective semantic verification (LLM checks "did this accomplish the task?")
- ToolEnabledPlanner wired into multi-agent executor
- ActionExecutor post-execution learning hooks
- Step-level skill extraction from partial successes

See `SCAFFOLDING_ANALYSIS.md` for the full technical analysis.

## Memory System

Four types, FAISS + SQLite backed:

| Type | Purpose | Example |
|------|---------|---------|
| FACT | Verified atomic knowledge | "Python is dynamically typed" |
| EXPERIENCE | Task outcomes, interactions | "Successfully fetched Wikipedia article" |
| BELIEF | Lessons learned, hypotheses | "fetch_text fails on sites without User-Agent" |
| SUMMARY | Compacted old experiences | "10 web tasks: 7 succeeded, 3 failed on auth" |

Beliefs evolve with evidence. Experiences compact into summaries via SSM model after 24h.

## Key API

```python
from main import Raec

raec = Raec()

# Conversation (routes via intent classifier)
response = raec.process_input("your message", mode="auto")

# Direct task execution
result = raec.execute_task("task description", mode="standard")

# Web access
raec.search_web("query", reason="why")
raec.web_fetch("https://...", reason="why")

# Curiosity
raec.add_question("What is X?")
raec.get_curiosity_stats()
raec.what_did_you_learn()

# Goals
raec.add_goal("name", "description", goal_type="user_given")
raec.get_active_goals()

# Performance
raec.analyze_performance()

# Shutdown
raec.close()
```

## GUI

- **Breathing Orb** — Visual state indicator
  - Cyan (idle) → Amber pulse (thinking) → Green (success) → Red (error) → Purple wandering (curious)
- **Conversation View** — Message bubbles with timestamps
- **Input Bar** — Mode selector (auto/standard/collab/step), text entry
- **Status Bar** — Intent, session info, system state

## Configuration

`config.yaml`:
```yaml
model:
  name: raec:latest
  device: rocm          # AMD GPU
memory:
  db_path: data/embeddings/raec_memory.db
tools:
  python_timeout: 60
planner:
  max_steps: 10
```

`config/swarm_config.json` — Model routing tiers, VRAM budgets, fallback chains.

## Development Notes

- Git repo with worktree workflow (no push auth — HTTPS only, no token/SSH)
- Worktree at `.claude/worktrees/vigorous-fermi`, merge to main from desktop dir
- `raec:latest` in Ollama = DeepSeek R1 32B Q4_K_M with custom system prompt
- All agency module databases in `data/` (SQLite): curiosity.db, goals.db, preferences.db, etc.
- `requirements.txt`: torch, transformers, sentence-transformers, faiss-cpu, requests, beautifulsoup4, selenium, pyyaml

## Session History

- **2026-02-05:** Test suite fixes, GUI rewrite, codebase cleanup
- **2026-02-13:** Web research bug fixes, scaffolding analysis (12 weaknesses identified), Phase 1 complete
- **2026-02-13:** Phase 2 complete (adaptive re-planning, param retry, per-step verification)
- **2026-02-14:** Phase 3 complete (failure→belief, semantic verification, multi-agent planner, learning hooks, step-level skills)
- **2026-02-15:** DeepSeek 32B orchestrator upgrade, curiosity engine + 8 agency modules integrated, GUI wired with curious orb state
- **2026-02-15:** Audit fixes (callback chaining, dead code), User-Agent fix for web tools, scaffolding fix (error-as-value escalation to hard failure)
