# raec_core/tool_interface.py

from dataclasses import dataclass
from typing import Any, Dict, Callable, Optional
import traceback
import time


@dataclass
class ToolResult:
    success: bool
    output: Any = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ToolInterface:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}

    def register(self, name: str, fn: Callable):
        if not callable(fn):
            raise ValueError(f"Tool '{name}' is not callable")
        self.tools[name] = fn

    def list_tools(self) -> Dict[str, str]:
        return {
            name: (fn.__doc__ or "").strip()
            for name, fn in self.tools.items()
        }

    def execute(self, name: str, **kwargs) -> ToolResult:
        start_time = time.time()

        if name not in self.tools:
            return ToolResult(
                success=False,
                error=f"Tool '{name}' not found",
                metadata={
                    "tool": name,
                    "duration": 0.0
                }
            )

        try:
            result = self.tools[name](**kwargs)
            return ToolResult(
                success=True,
                output=result,
                metadata={
                    "tool": name,
                    "duration": round(time.time() - start_time, 4),
                    "args": kwargs
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                metadata={
                    "tool": name,
                    "args": kwargs,
                    "duration": round(time.time() - start_time, 4),
                    "traceback": traceback.format_exc()
                }
            )
