"""Prompt observability utilities."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class PromptDebugRecord:
    """Serializable prompt debug payload."""

    timestamp: float
    source: str
    prompt: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "source": self.source,
            "prompt": self.prompt,
            "metadata": self.metadata,
        }


class PromptDebugLogger:
    """Stores assembled prompts for deterministic inspection."""

    def __init__(self, enabled: bool = False, export_dir: str = "logs/prompt_debug", print_prompt: bool = False):
        self.enabled = enabled
        self.export_dir = export_dir
        self.print_prompt = print_prompt

    def log(self, source: str, prompt: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Persist a prompt record and return output path if written."""
        if not self.enabled:
            return None

        record = PromptDebugRecord(
            timestamp=time.time(),
            source=source,
            prompt=prompt,
            metadata=metadata or {},
        )

        os.makedirs(self.export_dir, exist_ok=True)
        filename = f"{int(record.timestamp * 1000)}_{source.replace(' ', '_')}.json"
        output_path = os.path.join(self.export_dir, filename)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2)

        if self.print_prompt:
            print(f"\n[PROMPT DEBUG] source={source} -> {output_path}")
            print(prompt)
            print("[END PROMPT DEBUG]\n")

        return output_path
