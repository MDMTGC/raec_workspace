# raec_core/executor.py

from typing import Dict, Any
from raec_core.tool_interface import ToolInterface


class ToolExecutor:
    def __init__(self, tool_interface: ToolInterface):
        self.tool_interface = tool_interface

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        tool = self.tool_interface.get(tool_name)

        if not tool:
            raise ValueError(f"Tool '{tool_name}' not registered")

        try:
            return tool.func(arguments)
        except Exception as e:
            return {
                "error": str(e),
                "tool": tool_name,
                "arguments": arguments,
            }
