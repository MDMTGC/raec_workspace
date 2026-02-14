"""
Model Swarm - Hierarchical Multi-Model Router for RAEC

Routes tasks to specialized models based on task type:
- Orchestrator (large): High-level reasoning, synthesis, complex decisions
- Coder (medium): Code generation, analysis, bug detection
- Tool Agent (small): Tool selection, param generation, routing
- Data Agent (small): JSON parsing, data transformation

This allows RAEC to approximate frontier model capabilities
using a coordinated swarm of local specialized models.
"""

import requests
import time
import json
import os
from typing import Optional, Dict, Any, Generator
from enum import Enum


class TaskType(Enum):
    """Categories of tasks that map to model specializations"""
    # Orchestrator tasks (large model - 32B+)
    REASONING = "reasoning"
    SYNTHESIS = "synthesis"
    PLANNING = "planning"
    ERROR_RECOVERY = "error_recovery"

    # Coder tasks (code-specialized model - 4-7B)
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    BUG_DETECTION = "bug_detection"
    CODE_REVIEW = "code_review"
    SYNTAX_REPAIR = "syntax_repair"

    # Router/Tool agent tasks (3B "ganglia" class - <50ms, ~2GB VRAM)
    TOOL_SELECTION = "tool_selection"
    PARAM_GENERATION = "param_generation"
    ROUTING = "routing"
    INTENT_CLASSIFICATION = "intent_classification"

    # Data agent tasks (3B class)
    JSON_PARSING = "json_parsing"
    DATA_TRANSFORM = "data_transform"
    EXTRACTION = "extraction"

    # Curator tasks (SSM/Mamba - linear scaling for long context)
    # Critical: Use SSMs not transformers for memory operations
    MEMORY_DIGEST = "memory_digest"
    LOG_COMPRESSION = "log_compression"
    HISTORY_SUMMARY = "history_summary"
    CONTEXT_CURATION = "context_curation"

    # Default
    DEFAULT = "default"


# Default model assignments - can be overridden by config
# Architecture based on 2026 best practices:
# - 32B orchestrator for reasoning
# - 4-7B coder for code tasks
# - 3B "ganglia" for routing (<50ms)
# - SSM/Mamba for memory curation (linear scaling)
DEFAULT_MODEL_MAP = {
    # Orchestrator - DeepSeek R1 32B for strong reasoning
    TaskType.REASONING: "deepseek-r1:32b",
    TaskType.SYNTHESIS: "deepseek-r1:32b",
    TaskType.PLANNING: "deepseek-r1:32b",
    TaskType.ERROR_RECOVERY: "deepseek-r1:32b",

    # Coder - code-specialized models (4-7B sweet spot)
    TaskType.CODE_GENERATION: "qwen2.5-coder:7b",
    TaskType.CODE_ANALYSIS: "qwen2.5-coder:7b",
    TaskType.BUG_DETECTION: "qwen2.5-coder:7b",
    TaskType.CODE_REVIEW: "qwen2.5-coder:7b",
    TaskType.SYNTAX_REPAIR: "qwen3:4b",

    # Router/Tool agent - 3B "ganglia" class (<50ms, ~2GB VRAM)
    TaskType.TOOL_SELECTION: "qwen3:4b",
    TaskType.PARAM_GENERATION: "qwen3:4b",
    TaskType.ROUTING: "phi4-mini:latest",
    TaskType.INTENT_CLASSIFICATION: "phi4-mini:latest",

    # Data agent - 3-4B class
    TaskType.JSON_PARSING: "qwen3:4b",
    TaskType.DATA_TRANSFORM: "qwen3:4b",
    TaskType.EXTRACTION: "qwen3:4b",

    # Curator - SSM/Mamba for memory operations (linear scaling)
    # CRITICAL: Don't use transformers for these - O(n²) will explode on long logs
    TaskType.MEMORY_DIGEST: "jamba-reasoning:3b",
    TaskType.LOG_COMPRESSION: "jamba-reasoning:3b",
    TaskType.HISTORY_SUMMARY: "jamba-reasoning:3b",
    TaskType.CONTEXT_CURATION: "jamba-reasoning:3b",

    # Fallback
    TaskType.DEFAULT: "raec:latest",
}

# Recommended models for each role (to be pulled via ollama)
# ============================================================
# VALIDATED 2026 LANDSCAPE (Research: Feb 5, 2026)
# ============================================================
#
# Key findings from 2026 research:
# 1. Data quality > model size (Phi-3 proved 3.8B can match 12B+ with good data)
# 2. 4B model in 2026 routinely outperforms 13B from 2023
# 3. SSM/Mamba achieves 220K context in 24GB VRAM vs ~32K for transformers
# 4. Falcon-H1 hybrid (attention + Mamba-2) beats Qwen3-32B at half the size
#
RECOMMENDED_MODELS = {
    # Large orchestrator for complex reasoning (32B+)
    # DeepSeek-R1 remains strong; Falcon-H1-34B is hybrid alternative
    "orchestrator": [
        "deepseek-r1:32b",      # Current: strong reasoning
        "qwen2.5:32b",          # Strong general purpose
        "falcon-h1:34b",        # NEW: Hybrid attention+Mamba, beats Qwen3-32B
    ],

    # Code specialist (7B sweet spot - validated)
    # Qwen 2.5 Coder 7B confirmed as strongest sub-7B for code
    "coder": [
        "qwen2.5-coder:7b",     # BEST: Purpose-built, competitive with GPT-4o on code
        "deepseek-r1-distill-qwen:7b",  # Good for multi-step logical analysis
        "codellama:7b",         # Fallback
    ],

    # Fast 3B "ganglia" class for routing/simple tasks (<50ms, ~2GB VRAM)
    # 2026 3B outperforms 2023 7B - confirmed by benchmarks
    "router": [
        "smollm3:3b",           # NEW: Hugging Face, outperforms Llama-3.2-3B, tool-calling support
        "qwen3:4b",             # Qwen3-4B-Instruct-2507: best fine-tuned, optimized non-thinking
        "phi-4-mini",           # Microsoft: best reasoning-to-size ratio sub-7B
        "llama3.2:3b",          # Meta: solid baseline, 40-60 tok/s on laptop GPU
    ],
    "tool_agent": [
        "smollm3:3b",           # BEST for agentic: native tool-calling, 64K context
        "qwen3:4b",             # Excellent tool usage benchmarks
        "phi-4-mini",           # Good at structured output
    ],
    "data_agent": [
        "qwen3:4b",             # Strong JSON/structured tasks
        "smollm3:3b",           # Good extraction
        "phi-4-mini",           # Reliable parsing
    ],

    # SSM/Mamba models for memory curation (linear O(n) scaling)
    # ============================================================
    # CRITICAL: Transformers have O(n²) attention - EXPLODES on execution logs
    # SSMs achieve 220K context in same VRAM where transformers cap at ~32K
    # ============================================================
    "curator": [
        "falcon-mamba:7b",      # AVAILABLE ON OLLAMA: Hudson/falcon-mamba-instruct
        "falcon-h1:3b",         # NEW: Hybrid attention+Mamba, parallel architecture
        "jamba:3b",             # AVAILABLE ON OLLAMA: sam860/jamba-reasoning:3b
        # Pure Mamba achieves 4-5x inference throughput vs same-size transformer
    ],
}


class ModelSwarm:
    """
    Hierarchical model router that dispatches tasks to specialized models.

    Architecture:
        ┌─────────────────────────────────────────┐
        │         ORCHESTRATOR (32B+)             │
        │   Planning, Reasoning, Synthesis        │
        └────────────────┬────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
    ┌───────────┐  ┌───────────┐  ┌───────────┐
    │  CODER    │  │   TOOL    │  │   DATA    │
    │  (7B)     │  │  AGENT    │  │  AGENT    │
    │           │  │  (3B)     │  │  (3B)     │
    └───────────┘  └───────────┘  └───────────┘
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        endpoint: str = "http://localhost:11434/api/generate",
        timeout: int = 300,
        fallback_model: str = "raec:latest"
    ):
        self.endpoint = endpoint
        self.timeout = timeout
        self.fallback_model = fallback_model

        # Load model map from config or use defaults
        self.model_map = self._load_config(config_path)

        # Track available models
        self.available_models = self._check_available_models()

        # Statistics
        self.stats = {
            "calls_by_task": {},
            "calls_by_model": {},
            "latency_by_model": {},
        }

    def _load_config(self, config_path: Optional[str]) -> Dict[TaskType, str]:
        """Load model assignments from config file or use defaults"""
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)

                model_map = {}
                for task_name, model in config.get("model_map", {}).items():
                    try:
                        task_type = TaskType(task_name)
                        model_map[task_type] = model
                    except ValueError:
                        print(f"[!] Unknown task type in config: {task_name}")

                # Fill in missing with defaults
                for task_type in TaskType:
                    if task_type not in model_map:
                        model_map[task_type] = DEFAULT_MODEL_MAP[task_type]

                print(f"[OK] Loaded model swarm config from {config_path}")
                return model_map

            except Exception as e:
                print(f"[!] Failed to load config: {e}, using defaults")

        return DEFAULT_MODEL_MAP.copy()

    def _check_available_models(self) -> set:
        """Query Ollama for available models"""
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=5)
            r.raise_for_status()
            models = {m["name"] for m in r.json().get("models", [])}
            return models
        except Exception as e:
            print(f"[!] Could not query Ollama models: {e}")
            return set()

    def get_model_for_task(self, task_type: TaskType) -> str:
        """Get the appropriate model for a task type, with fallback"""
        model = self.model_map.get(task_type, self.fallback_model)

        # Check if model is available
        if model not in self.available_models:
            # Try to find an available alternative
            if self.fallback_model in self.available_models:
                return self.fallback_model
            # Last resort: return first available model
            if self.available_models:
                return next(iter(self.available_models))
            return model  # Return anyway, let Ollama error

        return model

    def generate(
        self,
        prompt: str,
        task_type: TaskType = TaskType.DEFAULT,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: Optional[list] = None,
        model_override: Optional[str] = None
    ) -> str:
        """
        Generate a response using the appropriate model for the task type.

        Args:
            prompt: The prompt to send
            task_type: Type of task (determines which model to use)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            model_override: Force a specific model (bypasses routing)

        Returns:
            Generated text response
        """
        # Select model
        model = model_override or self.get_model_for_task(task_type)

        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "num_predict": max_tokens,
            "stream": False
        }

        if stop:
            payload["stop"] = stop

        start_time = time.perf_counter()

        for attempt in range(3):
            try:
                r = requests.post(
                    self.endpoint,
                    json=payload,
                    timeout=self.timeout
                )
                r.raise_for_status()

                elapsed = time.perf_counter() - start_time
                self._record_stats(task_type, model, elapsed)

                return r.json()["response"].strip()

            except Exception as e:
                if attempt == 2:
                    raise RuntimeError(f"LLM call failed after 3 attempts: {e}")
                time.sleep(1.5)

    def stream(
        self,
        prompt: str,
        task_type: TaskType = TaskType.DEFAULT,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: Optional[list] = None,
        model_override: Optional[str] = None
    ) -> Generator[str, None, None]:
        """Streaming version of generate"""
        model = model_override or self.get_model_for_task(task_type)

        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "num_predict": max_tokens,
            "stream": True
        }

        if stop:
            payload["stop"] = stop

        with requests.post(
            self.endpoint,
            json=payload,
            stream=True,
            timeout=self.timeout
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if line:
                    yield line

    def _record_stats(self, task_type: TaskType, model: str, elapsed: float):
        """Record statistics for monitoring"""
        task_name = task_type.value

        # Calls by task
        if task_name not in self.stats["calls_by_task"]:
            self.stats["calls_by_task"][task_name] = 0
        self.stats["calls_by_task"][task_name] += 1

        # Calls by model
        if model not in self.stats["calls_by_model"]:
            self.stats["calls_by_model"][model] = 0
        self.stats["calls_by_model"][model] += 1

        # Latency by model
        if model not in self.stats["latency_by_model"]:
            self.stats["latency_by_model"][model] = []
        self.stats["latency_by_model"][model].append(elapsed)

    def get_stats(self) -> Dict[str, Any]:
        """Get swarm statistics"""
        stats = {
            "calls_by_task": self.stats["calls_by_task"],
            "calls_by_model": self.stats["calls_by_model"],
            "avg_latency_by_model": {},
            "available_models": list(self.available_models),
        }

        for model, latencies in self.stats["latency_by_model"].items():
            if latencies:
                stats["avg_latency_by_model"][model] = sum(latencies) / len(latencies)

        return stats

    def save_config(self, path: str):
        """Save current model map to config file"""
        config = {
            "model_map": {
                task_type.value: model
                for task_type, model in self.model_map.items()
            },
            "fallback_model": self.fallback_model,
            "recommended_models": RECOMMENDED_MODELS,
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"[OK] Saved swarm config to {path}")

    def set_model(self, task_type: TaskType, model: str):
        """Update model assignment for a task type"""
        self.model_map[task_type] = model
        print(f"[OK] {task_type.value} -> {model}")

    def refresh_available_models(self):
        """Re-check which models are available"""
        self.available_models = self._check_available_models()
        return self.available_models


# Convenience function to infer task type from context
def infer_task_type(prompt: str, context: Optional[Dict] = None) -> TaskType:
    """
    Attempt to infer the task type from the prompt content.
    This allows automatic routing without explicit task_type specification.

    Priority order matters - more specific patterns checked first.
    """
    prompt_lower = prompt.lower()

    # Memory/curation keywords -> SSM/Mamba (CRITICAL: linear scaling needed)
    # Check these FIRST before synthesis catches "summarize"
    if any(kw in prompt_lower for kw in ["summarize history", "digest log", "compress memory",
                                          "curate context", "memory summary", "execution log",
                                          "execution history", "session history", "compress log"]):
        return TaskType.MEMORY_DIGEST

    # Syntax repair (3B is often enough)
    if any(kw in prompt_lower for kw in ["syntax error", "missing comma", "fix syntax",
                                          "parse error", "invalid json"]):
        return TaskType.SYNTAX_REPAIR

    # Intent classification / routing (3B ganglia, <50ms)
    if any(kw in prompt_lower for kw in ["classify", "what type", "categorize", "route to"]):
        return TaskType.INTENT_CLASSIFICATION

    # Code-related keywords (4-7B coder)
    if any(kw in prompt_lower for kw in ["def ", "class ", "function", "bug", "error in code", "fix the"]):
        if "bug" in prompt_lower or "error" in prompt_lower or "fix" in prompt_lower:
            return TaskType.BUG_DETECTION
        return TaskType.CODE_ANALYSIS

    # Tool/param keywords (3B ganglia)
    if any(kw in prompt_lower for kw in ["tool:", "params:", "select tool", "which tool"]):
        return TaskType.TOOL_SELECTION

    # Planning keywords (32B orchestrator)
    if any(kw in prompt_lower for kw in ["plan", "steps", "break down", "how to"]):
        return TaskType.PLANNING

    # JSON/data keywords (3B data agent)
    if any(kw in prompt_lower for kw in ["json", "parse", "extract", "transform"]):
        return TaskType.JSON_PARSING

    # Synthesis keywords (32B orchestrator)
    if any(kw in prompt_lower for kw in ["summarize", "explain", "conclude", "result"]):
        return TaskType.SYNTHESIS

    return TaskType.DEFAULT


class LLMInterface:
    """
    Drop-in replacement for the original LLMInterface that uses ModelSwarm.
    Maintains backward compatibility while enabling swarm routing.
    """

    def __init__(
        self,
        model: str = "raec:latest",
        endpoint: str = "http://localhost:11434/api/generate",
        timeout: int = 300,
        config_path: Optional[str] = None,
        use_swarm: bool = True
    ):
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout
        self.use_swarm = use_swarm

        if use_swarm:
            self.swarm = ModelSwarm(
                config_path=config_path,
                endpoint=endpoint,
                timeout=timeout,
                fallback_model=model
            )
        else:
            self.swarm = None

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: Optional[list] = None,
        task_type: Optional[TaskType] = None
    ) -> str:
        """
        Generate response, optionally routing through swarm.

        If task_type is not specified, attempts to infer it from the prompt.
        """
        if self.use_swarm and self.swarm:
            # Auto-infer task type if not specified
            if task_type is None:
                task_type = infer_task_type(prompt)

            return self.swarm.generate(
                prompt=prompt,
                task_type=task_type,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop
            )
        else:
            # Legacy single-model behavior
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "num_predict": max_tokens,
                "stream": False
            }

            if stop:
                payload["stop"] = stop

            for attempt in range(3):
                try:
                    r = requests.post(
                        self.endpoint,
                        json=payload,
                        timeout=self.timeout
                    )
                    r.raise_for_status()
                    return r.json()["response"].strip()
                except Exception as e:
                    if attempt == 2:
                        raise RuntimeError(f"LLM call failed: {e}")
                    time.sleep(1.5)

    def stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: Optional[list] = None
    ) -> Generator[str, None, None]:
        """Streaming generation"""
        if self.use_swarm and self.swarm:
            task_type = infer_task_type(prompt)
            yield from self.swarm.stream(
                prompt=prompt,
                task_type=task_type,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop
            )
        else:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "num_predict": max_tokens,
                "stream": True
            }

            if stop:
                payload["stop"] = stop

            with requests.post(
                self.endpoint,
                json=payload,
                stream=True,
                timeout=self.timeout
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if line:
                        yield line

    def get_swarm_stats(self) -> Optional[Dict]:
        """Get swarm statistics if swarm is enabled"""
        if self.swarm:
            return self.swarm.get_stats()
        return None
