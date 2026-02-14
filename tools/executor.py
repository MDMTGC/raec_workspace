"""
Tool Executor - Bridges the tool interface with core tools.

Registers all core tools, tracks execution stats, and provides
the primary tool execution API for the rest of the system.
"""

import time
from typing import Dict, Any, List, Optional
from enum import Enum

from tools.tool_interface import ToolInterface, ToolResult
from tools.core_tools import (
    FileTools, WebTools, DataTools, CodeTools,
    TextTools, MathTools, SystemTools, TOOL_REGISTRY
)


class ToolType(Enum):
    """Categories of tools available in the system"""
    FILE = "file"
    WEB = "web"
    DATA = "data"
    CODE = "code"
    TEXT = "text"
    MATH = "math"
    SYSTEM = "system"


class ToolExecutor:
    """
    Main tool execution engine.

    Registers all core tools on init, routes execution requests,
    and tracks per-tool statistics (call count, success rate, timing).
    """

    def __init__(self, python_timeout: int = 60):
        self.interface = ToolInterface()
        self.python_timeout = python_timeout

        # Execution statistics: tool_key -> {calls, successes, total_ms}
        self._stats: Dict[str, Dict[str, Any]] = {}

        # Register every method from every tool class in core_tools
        self._register_core_tools()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def _register_core_tools(self):
        """Walk TOOL_REGISTRY and register each static method."""
        for category, cls in TOOL_REGISTRY.items():
            for attr_name in dir(cls):
                if attr_name.startswith("_"):
                    continue
                fn = getattr(cls, attr_name)
                if callable(fn):
                    key = f"{category}.{attr_name}"
                    self.interface.register(key, fn)
                    self._stats[key] = {
                        "calls": 0,
                        "successes": 0,
                        "total_ms": 0.0,
                    }

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, tool_key: str, **kwargs) -> ToolResult:
        """
        Execute a tool by its dotted key (e.g. 'file.read_file').

        Also accepts split arguments:
            execute('file', 'read_file', filepath='x.txt')
        """
        start = time.perf_counter()
        result = self.interface.execute(tool_key, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if tool_key in self._stats:
            self._stats[tool_key]["calls"] += 1
            self._stats[tool_key]["total_ms"] += elapsed_ms
            if result.success:
                self._stats[tool_key]["successes"] += 1

        return result

    def execute_by_category(
        self, category: str, tool_name: str, **kwargs
    ) -> ToolResult:
        """Execute a tool using separate category and name."""
        return self.execute(f"{category}.{tool_name}", **kwargs)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_tools(self) -> Dict[str, str]:
        """Return {tool_key: docstring} for every registered tool."""
        return self.interface.list_tools()

    def get_tools_for_llm(self) -> str:
        """
        Return a plain-text description of all available tools,
        suitable for inclusion in an LLM prompt.
        Includes parameter signatures with types, defaults,
        required/optional markers, and return types for accurate tool usage.
        """
        import inspect

        lines = ["**Available Tools:**", ""]
        current_category = None

        for key in sorted(self.interface.tools.keys()):
            cat = key.split(".")[0] if "." in key else "other"
            if cat != current_category:
                current_category = cat
                lines.append(f"  [{cat.upper()}]")

            fn = self.interface.tools[key]
            doc = (fn.__doc__ or "").strip().split('\n')[0]

            # Build detailed parameter info
            try:
                sig = inspect.signature(fn)
                param_parts = []
                for pname, param in sig.parameters.items():
                    if pname.startswith('_'):
                        continue
                    if param.kind == inspect.Parameter.VAR_KEYWORD:
                        continue
                    if param.kind == inspect.Parameter.VAR_POSITIONAL:
                        continue
                    ptype = ""
                    if param.annotation != inspect.Parameter.empty:
                        ptype = param.annotation.__name__ if hasattr(param.annotation, '__name__') else str(param.annotation)

                    if param.default != inspect.Parameter.empty:
                        default_repr = repr(param.default)
                        param_parts.append(f"{pname}: {ptype}={default_repr}" if ptype else f"{pname}={default_repr}")
                    else:
                        marker = " [REQUIRED]" if ptype else ""
                        param_parts.append(f"{pname}: {ptype}{marker}" if ptype else f"{pname}{marker}")

                param_str = f"({', '.join(param_parts)})" if param_parts else "()"

                # Get return type
                ret = sig.return_annotation
                if ret != inspect.Parameter.empty and ret is not None:
                    ret_name = ret.__name__ if hasattr(ret, '__name__') else str(ret)
                    ret_str = f" -> {ret_name}"
                else:
                    ret_str = ""
            except (ValueError, TypeError):
                param_str = "()"
                ret_str = ""

            lines.append(f"    {key}{param_str}{ret_str}: {doc or '(no description)'}")

        lines.append("")
        return "\n".join(lines)

    def get_tool_stats(self) -> Dict[str, Any]:
        """Aggregate statistics across all tools."""
        total_tools = len(self._stats)
        total_calls = sum(s["calls"] for s in self._stats.values())
        total_successes = sum(s["successes"] for s in self._stats.values())
        active = sum(1 for s in self._stats.values() if s["calls"] > 0)

        return {
            "total_tools": total_tools,
            "verified": total_tools,  # all core tools are verified
            "active": active,
            "total_executions": total_calls,
            "avg_success_rate": (
                total_successes / total_calls if total_calls else 1.0
            ),
        }

    def detect_bottlenecks(self, threshold_ms: float = 500) -> List[Dict]:
        """Return tools whose average execution time exceeds *threshold_ms*."""
        bottlenecks = []
        for key, s in self._stats.items():
            if s["calls"] == 0:
                continue
            avg_ms = s["total_ms"] / s["calls"]
            if avg_ms > threshold_ms:
                bottlenecks.append({
                    "tool": key,
                    "avg_ms": round(avg_ms, 2),
                    "calls": s["calls"],
                })
        if bottlenecks:
            print(f"   Found {len(bottlenecks)} bottleneck(s):")
            for b in bottlenecks:
                print(f"     {b['tool']}: avg {b['avg_ms']}ms over {b['calls']} calls")
        else:
            print("   No bottlenecks detected")
        return bottlenecks

    # ------------------------------------------------------------------
    # TOOL EVOLUTION
    # ------------------------------------------------------------------
    # Allow the tool registry to grow from successful executions.
    # New tools can be registered dynamically from learned patterns
    # or user-provided implementations.
    # ------------------------------------------------------------------

    def register_evolved_tool(
        self,
        tool_key: str,
        fn: callable,
        description: str = "",
        source: str = "evolved",
        test_cases: Optional[List[Dict]] = None
    ) -> bool:
        """
        Register a new tool that evolved from successful patterns.

        Args:
            tool_key: Dotted key (e.g., 'custom.my_tool')
            fn: The callable to register
            description: Human-readable description
            source: Where this tool came from ('evolved', 'user', 'extracted')
            test_cases: Optional test cases for verification

        Returns:
            True if registration succeeded
        """
        # Prevent overwriting core tools
        if tool_key in self._stats and self._stats[tool_key].get('core', False):
            print(f"[!] Cannot overwrite core tool: {tool_key}")
            return False

        # Verify the tool if test cases provided
        if test_cases:
            passed = 0
            for tc in test_cases:
                try:
                    result = fn(**tc.get('input', {}))
                    expected = tc.get('expected')
                    if expected is None or result == expected:
                        passed += 1
                except Exception as e:
                    print(f"[!] Test case failed: {e}")

            if passed < len(test_cases) * 0.8:  # 80% threshold
                print(f"[!] Tool {tool_key} failed verification ({passed}/{len(test_cases)} tests)")
                return False

        # Register the tool
        self.interface.register(tool_key, fn)
        self._stats[tool_key] = {
            "calls": 0,
            "successes": 0,
            "total_ms": 0.0,
            "core": False,
            "source": source,
            "description": description,
            "evolved_at": time.time(),
            "source_code": None,  # Will be set if registered via register_python_tool
        }

        print(f"[+] Registered evolved tool: {tool_key}")
        return True

    def register_python_tool(
        self,
        tool_key: str,
        code: str,
        description: str = "",
        test_cases: Optional[List[Dict]] = None
    ) -> bool:
        """
        Register a new tool from Python code.

        The code should define a function matching the tool name.
        Example:
            code = '''
            def my_tool(x, y):
                return x + y
            '''
            executor.register_python_tool('custom.my_tool', code)

        Args:
            tool_key: Dotted key where last part is function name
            code: Python code defining the function
            description: Human-readable description
            test_cases: Optional test cases for verification

        Returns:
            True if registration succeeded
        """
        # Extract function name from key
        func_name = tool_key.split('.')[-1]

        # Create isolated namespace
        namespace = {}

        try:
            # Execute code in isolated namespace
            exec(code, namespace)

            if func_name not in namespace:
                print(f"[!] Code does not define function: {func_name}")
                return False

            fn = namespace[func_name]

            if not callable(fn):
                print(f"[!] {func_name} is not callable")
                return False

            # Register via standard method
            success = self.register_evolved_tool(
                tool_key=tool_key,
                fn=fn,
                description=description or f"Python tool: {func_name}",
                source="python_code",
                test_cases=test_cases
            )

            # Store original code for persistence
            if success:
                self._stats[tool_key]["source_code"] = code

            return success

        except Exception as e:
            print(f"[!] Failed to compile tool code: {e}")
            return False

    def extract_tool_from_execution(
        self,
        task_description: str,
        execution_result: Dict,
        solution_code: str
    ) -> Optional[str]:
        """
        Extract a reusable tool from a successful execution.

        When a task is completed successfully with custom code,
        this method can extract that code as a new tool.

        Args:
            task_description: What the task was
            execution_result: The result of execution
            solution_code: The code that solved it

        Returns:
            tool_key if extraction succeeded, None otherwise
        """
        if not execution_result.get('success', False):
            return None

        # Generate tool key from task
        import re
        import hashlib

        # Clean task to create key
        words = re.findall(r'\w+', task_description.lower())[:3]
        base_key = '_'.join(words) if words else 'unnamed'

        # Add hash for uniqueness
        hash_suffix = hashlib.md5(solution_code.encode()).hexdigest()[:6]
        tool_key = f"evolved.{base_key}_{hash_suffix}"

        # Try to register
        success = self.register_python_tool(
            tool_key=tool_key,
            code=solution_code,
            description=f"Extracted from: {task_description[:50]}..."
        )

        return tool_key if success else None

    def get_evolved_tools(self) -> List[Dict]:
        """Return list of evolved (non-core) tools."""
        evolved = []
        for key, stats in self._stats.items():
            if not stats.get('core', True):
                evolved.append({
                    'tool_key': key,
                    'source': stats.get('source', 'unknown'),
                    'description': stats.get('description', ''),
                    'calls': stats['calls'],
                    'successes': stats['successes'],
                    'evolved_at': stats.get('evolved_at'),
                })
        return evolved

    def deprecate_tool(self, tool_key: str) -> bool:
        """
        Deprecate an evolved tool that is no longer useful.

        Core tools cannot be deprecated.
        """
        if tool_key not in self._stats:
            return False

        if self._stats[tool_key].get('core', True):
            print(f"[!] Cannot deprecate core tool: {tool_key}")
            return False

        # Remove from interface
        if hasattr(self.interface, 'unregister'):
            self.interface.unregister(tool_key)

        # Mark as deprecated in stats
        self._stats[tool_key]['deprecated'] = True
        self._stats[tool_key]['deprecated_at'] = time.time()

        print(f"[-] Deprecated tool: {tool_key}")
        return True

    def save_evolved_tools(self, filepath: str = "tools/evolved_tools.json"):
        """Persist evolved tools to disk."""
        import json
        import os
        import inspect

        evolved = []
        for key, stats in self._stats.items():
            if stats.get('core', True):
                continue
            if stats.get('deprecated', False):
                continue

            # Get stored source code
            source_code = stats.get('source_code')

            if not source_code:
                # Try to extract from function
                fn = self.interface.tools.get(key)
                try:
                    source_code = inspect.getsource(fn) if fn else None
                except (TypeError, OSError):
                    source_code = None

            if source_code:  # Only save if we have code
                evolved.append({
                    'tool_key': key,
                    'source': stats.get('source'),
                    'description': stats.get('description', ''),
                    'evolved_at': stats.get('evolved_at'),
                    'source_code': source_code,
                })

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(evolved, f, indent=2)

        print(f"[OK] Saved {len(evolved)} evolved tools to {filepath}")

    def load_evolved_tools(self, filepath: str = "tools/evolved_tools.json"):
        """Load evolved tools from disk."""
        import json
        import os

        if not os.path.exists(filepath):
            return

        with open(filepath, 'r') as f:
            tools = json.load(f)

        loaded = 0
        for tool_data in tools:
            if tool_data.get('source_code'):
                success = self.register_python_tool(
                    tool_key=tool_data['tool_key'],
                    code=tool_data['source_code'],
                    description=tool_data.get('description', '')
                )
                if success:
                    loaded += 1

        print(f"[OK] Loaded {loaded} evolved tools from {filepath}")

    # ------------------------------------------------------------------
    # LEGACY API (for test compatibility)
    # ------------------------------------------------------------------

    @property
    def execution_history(self) -> List[Dict]:
        """Return execution history for test compatibility."""
        if not hasattr(self, '_execution_history'):
            self._execution_history = []
        return self._execution_history

    def _record_execution(self, record: Dict):
        """Record an execution for test compatibility."""
        if not hasattr(self, '_execution_history'):
            self._execution_history = []
        self._execution_history.append(record)

    def run_python(self, code: str, description: str = "") -> str:
        """
        Execute Python code and return output.

        This is a convenience method for test compatibility.
        Wraps the code.run_python tool.
        """
        start = time.perf_counter()
        result = self.execute("code.run_python", code=code)
        elapsed = time.perf_counter() - start

        # Record execution
        self._record_execution({
            'type': 'python',
            'description': description,
            'success': result.success,
            'runtime': elapsed,
            'output': result.output if result.success else result.error
        })

        if result.success:
            return str(result.output) if result.output else ""
        else:
            return f"Error: {result.error}"
