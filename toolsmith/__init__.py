"""
RAEC Toolsmith - Dynamic tool creation and capability expansion

When RAEC needs a capability it doesn't have, it builds a tool.
Tools are:
- Generated from natural language specifications
- Tested before deployment
- Versioned and tracked
- Subject to verification
"""

from .tool_forge import ToolForge, GeneratedTool, ToolSpec

__all__ = [
    'ToolForge',
    'GeneratedTool',
    'ToolSpec',
]
