\# AGENTS.md — RAEC Project Context



\## Project Name

RAEC — Reflective Agentic Ecosystem Composer



\## Mission



RAEC is a local-first, multi-model agentic AI system designed to replace reliance on frontier hosted models by using rich scaffolding and layered architecture on limited hardware (Ryzen 5600X, 32GB RAM, RX 7900 XT 20GB VRAM).



RAEC is not intended to simulate a human.  

It is intended to function as a high-capability cognitive engine with an intuitive conversational interface.



\## Design Philosophy



\- Execution reliability > persona theatrics

\- Structured cognition > chat illusion

\- Explicit state > implicit magic

\- Modular architecture > monolith

\- Scaffolding compensates for model scale limits

\- Conversation layer is an interface shell, not the intelligence core



\## Core Architectural Layers



1\. Cognitive Core

&nbsp;  - Planner

&nbsp;  - Executor

&nbsp;  - Tool system

&nbsp;  - Memory (FACT / EXPERIENCE / BELIEF / SUMMARY)

&nbsp;  - Skill graph

&nbsp;  - Goal manager

&nbsp;  - Curiosity engine

&nbsp;  - Self-evaluation

&nbsp;  - Confidence tracking



2\. Model Swarm

&nbsp;  - DeepSeek R1 32B (orchestrator reasoning)

&nbsp;  - Qwen / smaller models for classification and code

&nbsp;  - Fallback routing under VRAM pressure



3\. Interface Layer

&nbsp;  - Conversation manager

&nbsp;  - Intent classifier

&nbsp;  - Prompt assembly

&nbsp;  - GUI / CLI interaction



\## Important Constraints



\- Must run locally

\- Must respect 20GB VRAM ceiling

\- Serial model loading allowed

\- No dependence on cloud inference

\- Prompt assembly must be deterministic and inspectable



\## Current Known Weakness



Conversation continuity feels disjointed.

The system reconstructs context each turn rather than maintaining a persistent session state abstraction.



This is NOT a model quality issue.

It is an architectural interface issue.



\## Future Direction



Add a "Conversation State Manager" subsystem that:



\- Tracks active conversational thread

\- Maintains session-level state

\- Compresses history into rolling summaries

\- Separates cognitive core from conversational shell

\- Does not anthropomorphize the system



\## Engineering Standards



\- Python 3.13

\- Use type hints

\- Modular design

\- Avoid hidden global state

\- Add pytest tests for new subsystems

\- Prefer clarity over cleverness

\- Keep components independently testable



\## Commit Guidelines



Each change must include:



\- Purpose

\- Files modified

\- Example usage if relevant

\- Tests if behavior changes





