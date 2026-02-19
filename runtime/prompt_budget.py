"""Prompt budget instrumentation helpers."""

from __future__ import annotations

from typing import Any, Dict


class PromptBudgetAnalyzer:
    """Estimate prompt/component budget for observability and guardrails."""

    def __init__(self, warn_tokens: int = 6000) -> None:
        self.warn_tokens = warn_tokens

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        # Lightweight deterministic heuristic for local instrumentation.
        return max(1, len(text) // 4)

    def analyze(self, prompt: str, components: Dict[str, Any]) -> Dict[str, Any]:
        component_char_lengths = {
            key: len(str(value))
            for key, value in components.items()
        }
        prompt_chars = len(prompt)
        prompt_tokens_est = self._estimate_tokens(prompt)

        return {
            "prompt_chars": prompt_chars,
            "prompt_tokens_est": prompt_tokens_est,
            "component_chars": component_char_lengths,
            "total_component_chars": sum(component_char_lengths.values()),
            "warn_threshold_tokens": self.warn_tokens,
            "over_budget": prompt_tokens_est > self.warn_tokens,
        }
