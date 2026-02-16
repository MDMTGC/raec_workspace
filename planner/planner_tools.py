"""
Tool-Enabled Planner Module
Plans and executes tasks using actual tools
"""
import json
import time
import re
import inspect
from typing import List, Dict, Optional, Any
from enum import Enum


class PlanStatus(Enum):
    """Status of plan execution"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class PlanStep:
    """Individual step in a plan with tool execution"""
    def __init__(
        self,
        step_id: int,
        description: str,
        tool_category: Optional[str] = None,
        tool_name: Optional[str] = None,
        tool_params: Optional[Dict] = None,
        dependencies: Optional[List[int]] = None,
        status: PlanStatus = PlanStatus.PENDING,
        result: Optional[str] = None,
        error: Optional[str] = None
    ):
        self.step_id = step_id
        self.description = description
        self.tool_category = tool_category
        self.tool_name = tool_name
        self.tool_params = tool_params or {}
        self.dependencies = dependencies or []
        self.status = status
        self.result = result
        self.error = error
        self.start_time = None
        self.end_time = None


class ToolEnabledPlanner:
    """
    Planner that can actually execute tasks using tools
    
    Features:
    - Multi-step reasoning
    - Memory-augmented planning  
    - Actual tool execution
    - Error correction loops
    - Skill reuse from memory
    """
    
    def __init__(self, llm, memory, skills, tools=None, max_steps=10):
        self.llm = llm
        self.memory = memory
        self.skills = skills
        self.max_steps = max_steps
        self.current_plan = None
        self.execution_history = []

        # Tool executor — expect a ToolExecutor instance from main.py
        if tools is None:
            from tools.executor import ToolExecutor
            self.tools = ToolExecutor()
        else:
            self.tools = tools
    
    def run(self, task: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute a task with planning and tool usage.
        Includes adaptive re-planning: if execution fails badly,
        the planner generates a recovery plan using error context.
        """
        print(f"\n{'='*70}")
        print(f"[o] TASK: {task}")
        print(f"{'='*70}\n")

        max_replan_attempts = self.max_steps // 5 or 1  # Scale with plan complexity
        max_replan_attempts = min(max_replan_attempts, 2)  # Cap at 2 replans

        # Check for similar past tasks in memory
        similar_experiences = self._retrieve_similar_tasks(task)

        all_errors = []

        for attempt in range(1 + max_replan_attempts):
            if attempt == 0:
                # First attempt: normal planning
                plan_steps = self._generate_plan_with_tools(task, similar_experiences, context)
            else:
                # Re-plan attempt: feed errors back to the planner
                print(f"\n{'='*70}")
                print(f"[~] ADAPTIVE RE-PLAN (attempt {attempt}/{max_replan_attempts})")
                print(f"{'='*70}\n")
                plan_steps = self._generate_recovery_plan(
                    task, all_errors, similar_experiences, context
                )

            # Store plan in memory
            plan_memory_id = self._store_plan(task, plan_steps)

            # Execute plan using tools
            results = self._execute_plan_with_tools(plan_steps, task, context)

            # Check if execution succeeded or is good enough
            steps = results.get('steps', [])
            total = len(steps)
            completed = sum(1 for s in steps if s.get('status') == 'completed')
            completion_rate = completed / total if total else 0.0

            if results.get('success') or completion_rate >= 0.8:
                # Good enough — stop re-planning
                break

            # Collect errors for the next re-plan attempt
            all_errors.extend(results.get('errors', []))

            if attempt < max_replan_attempts:
                print(f"\n   [!] Completion rate: {completion_rate:.0%} — triggering re-plan")

        # Store execution results
        self._store_execution_results(task, results, plan_memory_id)

        return {
            'task': task,
            'plan_id': plan_memory_id,
            'steps': [self._step_to_dict(s) for s in plan_steps],
            'results': results,
            'success': results.get('success', False),
            'replan_attempts': attempt
        }
    
    def _retrieve_similar_tasks(self, task: str) -> List[Dict]:
        """Query memory for similar past tasks and relevant beliefs."""
        try:
            from memory.memory_db import MemoryType

            print("[?] Searching memory for similar past tasks...")
            similar = self.memory.query(
                query_text=task,
                memory_types=[MemoryType.EXPERIENCE, MemoryType.SUMMARY],
                k=3,
                min_confidence=0.7
            )

            if similar:
                print(f"   [OK] Found {len(similar)} relevant past experiences")
            else:
                print("   • No similar past experiences found")

            # W11: Also retrieve relevant beliefs (lessons from past failures)
            beliefs = self.memory.query(
                query_text=task,
                memory_types=[MemoryType.BELIEF],
                k=3,
                min_confidence=0.4
            )
            if beliefs:
                print(f"   [OK] Found {len(beliefs)} relevant beliefs/lessons")
                similar.extend(beliefs)

            print()
            return similar
        except Exception as e:
            print(f"   [!] Memory query failed: {e}\n")
            return []
    
    def _generate_plan_with_tools(
        self,
        task: str,
        similar_experiences: List[Dict],
        context: Optional[Dict]
    ) -> List[PlanStep]:
        """Generate a plan with tool assignments"""
        print("[=] Generating execution plan with tool assignments...\n")
        
        # Get available tools for LLM
        tools_doc = self.tools.get_tools_for_llm()
        
        # Build prompt
        prompt = self._build_planning_prompt_with_tools(
            task, similar_experiences, context, tools_doc
        )
        
        try:
            # Generate plan from LLM using PLANNING task type for model routing
            from raec_core.model_swarm import TaskType
            plan_text = self.llm.generate(prompt, temperature=0.5, max_tokens=2048, task_type=TaskType.PLANNING)
            
            # Parse plan into steps with tool assignments
            steps = self._parse_plan_with_tools(plan_text)
            
            print(f"[OK] Generated plan with {len(steps)} steps:\n")
            for step in steps:
                tool_info = f" [{step.tool_category}.{step.tool_name}]" if step.tool_category else ""
                deps = f" (depends on: {step.dependencies})" if step.dependencies else ""
                print(f"   {step.step_id}. {step.description}{tool_info}{deps}")
            print()
            
            return steps
            
        except Exception as e:
            print(f"[!] Plan generation failed: {e}")
            # Fallback to simple single-step plan
            return [PlanStep(1, f"Complete task: {task}")]
    
    def _generate_recovery_plan(
        self,
        task: str,
        errors: List[Dict],
        similar_experiences: List[Dict],
        context: Optional[Dict]
    ) -> List[PlanStep]:
        """
        Generate a recovery plan after a failed execution attempt.
        Feeds the error summary back to the LLM so it can avoid the same mistakes.
        """
        print("[=] Generating recovery plan from error context...\n")

        tools_doc = self.tools.get_tools_for_llm()

        # Build error summary
        error_lines = []
        for err in errors[-5:]:  # Last 5 errors max
            error_lines.append(
                f"- Step {err.get('step_id', '?')}: {str(err.get('error', 'unknown'))[:200]}"
            )
        error_summary = "\n".join(error_lines) if error_lines else "Unknown failures"

        # Build full recovery prompt using the standard prompt builder + error preamble
        base_prompt = self._build_planning_prompt_with_tools(
            task, similar_experiences, context, tools_doc
        )

        recovery_preamble = f"""**RECOVERY MODE:** The previous plan for this task FAILED.

**Errors from previous attempt:**
{error_summary}

**You MUST use a different approach.** Do NOT repeat the same failing tool calls.
If a URL returned an error, try a different URL or data source.
If a tool failed with bad parameters, use different parameters or a different tool.
Prefer composite tools (web.fetch_text, web.fetch_json) over raw http_get.

"""
        prompt = recovery_preamble + base_prompt

        try:
            from raec_core.model_swarm import TaskType
            plan_text = self.llm.generate(
                prompt, temperature=0.6, max_tokens=2048,
                task_type=TaskType.ERROR_RECOVERY
            )

            steps = self._parse_plan_with_tools(plan_text)

            print(f"[OK] Recovery plan with {len(steps)} steps:\n")
            for step in steps:
                tool_info = f" [{step.tool_category}.{step.tool_name}]" if step.tool_category else ""
                deps = f" (depends on: {step.dependencies})" if step.dependencies else ""
                print(f"   {step.step_id}. {step.description}{tool_info}{deps}")
            print()

            return steps

        except Exception as e:
            print(f"[!] Recovery plan generation failed: {e}")
            return [PlanStep(1, f"Complete task using reasoning: {task}")]

    def _build_planning_prompt_with_tools(
        self,
        task: str,
        similar_experiences: List[Dict],
        context: Optional[Dict],
        tools_doc: str
    ) -> str:
        """Build planning prompt with tool information"""
        prompt_parts = [
            "You are a task planner with access to tools.",
            f"\n**USER TASK:**\n{task}",
        ]

        if context:
            prompt_parts.append(f"\n**Context:**\n{json.dumps(context, indent=2)}")

        if similar_experiences:
            prompt_parts.append("\n**Relevant Past Experiences:**")
            for exp in similar_experiences[:3]:
                prompt_parts.append(f"- {exp['content'][:100]}")

        prompt_parts.append(f"\n{tools_doc}")

        prompt_parts.append('''
\n**CRITICAL INSTRUCTIONS:**
1. ONLY use tools from the list above. Do NOT invent tools or parameters that don't exist.
2. Every step with a tool MUST have PARAMS with ALL required arguments.
3. ONLY use parameter names shown in the tool signatures. Do NOT add extra parameters.
4. Use $stepN to reference output from step N (e.g., $step1, $step2).
5. Pay attention to return types (shown as -> type). Ensure the output type of one step matches the expected input type of the next step. For example, web.fetch_text returns str, so pass it to tools expecting str, not list.
6. data.parse_json only works on JSON strings, NOT on HTML. Use web.fetch_json for JSON APIs.
7. For web pages, ALWAYS use web.fetch_text (returns clean text) instead of web.http_get (returns raw HTML).

**Required Format - Follow EXACTLY:**
1. [Description of step]
   TOOL: category.tool_name
   PARAMS: {"param1": "value1", "param2": "value2"}

2. [Description] [DEPENDS: 1]
   TOOL: category.tool_name
   PARAMS: {"param": "$step1"}

**Common Patterns:**
- Get desktop path: TOOL: system.get_desktop_path, PARAMS: {}
- Get home directory: TOOL: system.get_home_dir, PARAMS: {}
- Join paths: TOOL: system.join_path, PARAMS: {"base": "$step1", "parts": "filename.txt"}
- List files: TOOL: file.list_directory, PARAMS: {"dirpath": "."}
- Read file: TOOL: file.read_file, PARAMS: {"filepath": "filename.py"}
- Write file: TOOL: file.write_file, PARAMS: {"filepath": "$step2", "content": "content here"}
- Analyze code: TOOL: code.validate_python, PARAMS: {"code": "$step2"}
- Run code: TOOL: code.run_python, PARAMS: {"code": "print('hello')"}
- Run shell/git: TOOL: code.run_shell, PARAMS: {"command": "git add file.txt", "cwd": "/path/to/repo"}
- Change directory: TOOL: system.change_dir, PARAMS: {"path": "/path/to/directory"}
- Count items: TOOL: data.count, PARAMS: {"data": "$step1"}
- Filter list: TOOL: data.filter_list, PARAMS: {"data": "$step1", "condition": ".py"}

**For web content retrieval (PREFERRED - single-step composite tools):**
- Fetch readable text from a web page: TOOL: web.fetch_text, PARAMS: {"url": "https://example.com"}
- Fetch JSON from an API endpoint: TOOL: web.fetch_json, PARAMS: {"url": "https://api.example.com/data"}
- Fetch all links from a page: TOOL: web.fetch_links, PARAMS: {"url": "https://example.com"}

These composite tools handle fetching + parsing in one step. ALWAYS prefer these over raw http_get.

Example for web research:
1. Fetch readable text from web page
   TOOL: web.fetch_text
   PARAMS: {"url": "https://example.com/article"}
2. Search for relevant content [DEPENDS: 1]
   TOOL: text.search_text
   PARAMS: {"text": "$step1", "pattern": "keyword"}

**Only use web.http_get if you need raw HTML** (e.g., to extract_links or parse specific HTML elements).

**For desktop/directory file operations:**
1. Use system.get_desktop_path to get the directory path
2. Use system.join_path to combine directory + filename
3. Use file.write_file with the full path from step 2

**IMPORTANT:** If you cannot accomplish a step with available tools, describe it WITHOUT a TOOL line and the system will use LLM reasoning instead.

Generate the plan now:
''')

        return "\n".join(prompt_parts)
    
    def _parse_plan_with_tools(self, plan_text: str) -> List[PlanStep]:
        """Parse LLM output into steps with tool assignments"""
        steps = []
        lines = plan_text.strip().split('\n')

        current_step_id = None
        current_desc = None
        current_tool_cat = None
        current_tool_name = None
        current_params = {}
        current_deps = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Match numbered steps
            step_match = re.match(r'^(\d+)[.)]\s*(.+)$', line)
            if step_match:
                # Save previous step if exists
                if current_step_id is not None:
                    steps.append(PlanStep(
                        step_id=current_step_id,
                        description=current_desc,
                        tool_category=current_tool_cat,
                        tool_name=current_tool_name,
                        tool_params=current_params,
                        dependencies=current_deps
                    ))

                # Start new step
                current_step_id = int(step_match.group(1))
                current_desc = step_match.group(2).strip()
                current_tool_cat = None
                current_tool_name = None
                current_params = {}
                current_deps = []

                # Extract dependencies from description
                dep_match = re.search(r'\[DEPENDS:\s*([\d,\s]+)\]', current_desc, re.IGNORECASE)
                if dep_match:
                    dep_str = dep_match.group(1)
                    current_deps = [int(x.strip()) for x in dep_str.split(',') if x.strip().isdigit()]
                    current_desc = re.sub(r'\[DEPENDS:[^\]]+\]', '', current_desc, flags=re.IGNORECASE).strip()

            # Match TOOL line
            tool_match = re.match(r'TOOL:\s*(\w+)\.(\w+)', line, re.IGNORECASE)
            if tool_match:
                current_tool_cat = tool_match.group(1)
                current_tool_name = tool_match.group(2)

            # Match PARAMS line
            params_match = re.match(r'PARAMS:\s*(.+)', line, re.IGNORECASE)
            if params_match:
                params_str = params_match.group(1)
                parsed_params = None
                try:
                    # Try to parse as dict - prefer json.loads for safety
                    import ast
                    try:
                        parsed_params = json.loads(params_str)
                    except json.JSONDecodeError:
                        # Fallback to ast.literal_eval for Python dict syntax
                        parsed_params = ast.literal_eval(params_str)
                except (ValueError, SyntaxError) as e:
                    print(f"   [!] Failed to parse PARAMS: {params_str[:50]}... ({e})")
                    parsed_params = {}  # Use empty dict as fallback

                if parsed_params is not None:
                    current_params = parsed_params

        # Save last step
        if current_step_id is not None:
            steps.append(PlanStep(
                step_id=current_step_id,
                description=current_desc,
                tool_category=current_tool_cat,
                tool_name=current_tool_name,
                tool_params=current_params,
                dependencies=current_deps
            ))

        # If parsing failed, create generic steps
        if not steps:
            for i, line in enumerate(lines, 1):
                if re.match(r'^\d+[.)]', line):
                    steps.append(PlanStep(i, line.strip()))

        # Validate and repair steps
        steps = self._validate_and_repair_steps(steps)

        return steps

    def _validate_and_repair_steps(self, steps: List[PlanStep]) -> List[PlanStep]:
        """
        Validate tools exist, check parameter names against actual signatures,
        strip invalid params, and repair missing required params where possible.
        """
        available_tools = set(self.tools.list_tools().keys())

        for step in steps:
            if not (step.tool_category and step.tool_name):
                continue

            tool_key = f"{step.tool_category}.{step.tool_name}"

            # Check if tool exists
            if tool_key not in available_tools:
                print(f"   [!] Tool '{tool_key}' not found - will use LLM reasoning")
                step.tool_category = None
                step.tool_name = None
                step.tool_params = {}
                continue

            # Validate parameters against actual signature
            fn = self.tools.interface.tools.get(tool_key)
            if fn:
                step.tool_params = self._validate_params(step, tool_key, fn)

            # Repair missing params based on tool and description
            if not step.tool_params:
                step.tool_params = self._infer_params(step, tool_key)

        return steps

    def _validate_params(
        self, step: PlanStep, tool_key: str, fn: callable
    ) -> Dict[str, Any]:
        """
        Validate step params against the tool's actual signature.
        Strips invalid params, warns about missing required ones.
        """
        if not step.tool_params:
            return step.tool_params

        try:
            sig = inspect.signature(fn)
        except (ValueError, TypeError):
            return step.tool_params

        valid_names = set()
        required_names = set()

        for pname, param in sig.parameters.items():
            if param.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
                continue
            if pname.startswith('_'):
                continue
            valid_names.add(pname)
            if param.default is inspect.Parameter.empty:
                required_names.add(pname)

        # Strip invalid params
        cleaned = {}
        for key, value in step.tool_params.items():
            if key in valid_names:
                cleaned[key] = value
            else:
                print(f"   [!] Step {step.step_id}: Stripped invalid param '{key}' from {tool_key}")

        # Warn about missing required params (that aren't $step references yet)
        for req in required_names:
            if req not in cleaned:
                print(f"   [!] Step {step.step_id}: Missing required param '{req}' for {tool_key}")

        return cleaned

    def _infer_params(self, step: PlanStep, tool_key: str) -> Dict[str, Any]:
        """
        Try to infer parameters from the step description and tool signature.
        """
        desc_lower = step.description.lower()
        params = {}

        # file.list_directory
        if tool_key == "file.list_directory":
            params["dirpath"] = "."

        # file.read_file
        elif tool_key == "file.read_file":
            # Try to extract filename from description
            file_match = re.search(r'[`"\']?(\w+\.\w+)[`"\']?', step.description)
            if file_match:
                params["filepath"] = file_match.group(1)

        # code.validate_python or code.run_python
        elif tool_key in ("code.validate_python", "code.run_python"):
            # These need code from previous step - use placeholder
            params["code"] = "$step_prev"

        # data.count
        elif tool_key == "data.count":
            params["data"] = "$step_prev"

        # data.filter_list
        elif tool_key == "data.filter_list":
            params["data"] = "$step_prev"
            # Try to extract condition
            if ".py" in desc_lower or "python" in desc_lower:
                params["condition"] = ".py"
            elif "filter" in desc_lower:
                params["condition"] = ""

        return params
    
    def _execute_plan_with_tools(
        self,
        steps: List[PlanStep],
        task: str,
        context: Optional[Dict]
    ) -> Dict[str, Any]:
        """Execute plan steps using actual tools, with failure recovery."""
        print("[*]  EXECUTING PLAN WITH TOOLS\n")
        print(f"{'='*70}\n")

        completed_steps = set()
        failed_steps = set()
        step_results = {}

        results = {
            'success': True,
            'steps': [],
            'errors': [],
            'recovery_attempts': 0
        }

        for step in steps:
            # Check dependencies — if any dependency failed, try LLM fallback
            unmet_deps = [dep for dep in step.dependencies if dep not in completed_steps]
            failed_deps = [dep for dep in unmet_deps if dep in failed_steps]

            if unmet_deps and not failed_deps:
                # Dependencies just haven't run yet (shouldn't happen with sequential exec)
                step.status = PlanStatus.BLOCKED
                results['steps'].append(self._step_to_dict(step))
                print(f"[||]  Step {step.step_id}: BLOCKED (dependencies not met)")
                continue

            if failed_deps:
                # Dependencies failed — try to recover via LLM reasoning
                print(f"[~]  Step {step.step_id}: Dependencies {failed_deps} failed — attempting LLM fallback")
                results['recovery_attempts'] += 1
                try:
                    step.start_time = time.time()
                    fallback_result = self._execute_step_with_llm(step, task, step_results, context)
                    step.result = fallback_result
                    step.status = PlanStatus.COMPLETED
                    step.end_time = time.time()
                    completed_steps.add(step.step_id)
                    step_results[step.step_id] = step.result
                    results['steps'].append(self._step_to_dict(step))
                    print(f"   [OK] Recovered via LLM reasoning")
                    print(f"   Duration: {step.end_time - step.start_time:.2f}s\n")
                    continue
                except Exception as e:
                    step.status = PlanStatus.BLOCKED
                    step.error = f"Recovery failed: {e}"
                    step.end_time = time.time()
                    results['steps'].append(self._step_to_dict(step))
                    print(f"   [X] Recovery failed: {e}\n")
                    continue

            # Execute step
            print(f"[>]  Step {step.step_id}: {step.description}")
            step.status = PlanStatus.IN_PROGRESS
            step.start_time = time.time()

            try:
                # Execute using tools if specified
                if step.tool_category and step.tool_name:
                    tool_key = f"{step.tool_category}.{step.tool_name}"

                    # Resolve params - inject previous step results where needed
                    resolved_params = self._resolve_params(
                        step.tool_params, step_results, step.step_id
                    )

                    print(f"   Using tool: {tool_key}")
                    print(f"   Params: {resolved_params}")

                    step_result = self.tools.execute(
                        tool_key,
                        **resolved_params
                    )

                    # Check ToolResult.success flag, then verify output content
                    warning = None
                    if step_result.success:
                        warning = self._quick_verify_step_output(
                            tool_key, step_result.output, step
                        )
                        if warning and ("contains error" in warning.lower() or "error dict" in warning.lower()):
                            # Hard failure: output looks like an error message
                            # Downgrade success to failure so recovery kicks in
                            print(f"   [!] Step verify FAILED: {warning}")
                            step_result = type(step_result)(
                                success=False, output=step_result.output,
                                error=warning, metadata=step_result.metadata
                            )
                        elif warning:
                            # Soft warning (empty output, unexpected HTML, etc.)
                            print(f"   [!] Step verify warning: {warning}")
                            step.result = step_result.output
                            step.status = PlanStatus.COMPLETED
                            step.end_time = time.time()
                            completed_steps.add(step.step_id)
                            step_results[step.step_id] = step_result.output
                            results['steps'].append(self._step_to_dict(step))
                            results.setdefault('warnings', []).append({
                                'step_id': step.step_id,
                                'warning': warning
                            })
                            print(f"   [OK] Result (with warning): {str(step_result.output)[:100]}...")
                            print(f"   Duration: {step.end_time - step.start_time:.2f}s\n")

                    # After verification: handle clean success, or fall through to failure
                    if step_result.success and not warning:
                        step.result = step_result.output
                        step.status = PlanStatus.COMPLETED
                        step.end_time = time.time()
                        completed_steps.add(step.step_id)
                        step_results[step.step_id] = step_result.output
                        results['steps'].append(self._step_to_dict(step))
                        print(f"   [OK] Result: {str(step_result.output)[:100]}...")
                        print(f"   Duration: {step.end_time - step.start_time:.2f}s\n")
                    elif not step_result.success:
                        # Tool failed — try param correction retry, then LLM fallback
                        error_msg = step_result.error or str(step_result.output)
                        print(f"   [!] Tool failed: {error_msg}")
                        results['recovery_attempts'] += 1

                        # Stage 1: Try correcting params via LLM and retrying the tool
                        retry_result = self._retry_with_corrected_params(
                            step, tool_key, resolved_params, error_msg, task
                        )
                        if retry_result is not None:
                            step.result = retry_result
                            step.status = PlanStatus.COMPLETED
                            step.end_time = time.time()
                            completed_steps.add(step.step_id)
                            step_results[step.step_id] = step.result
                            results['steps'].append(self._step_to_dict(step))
                            print(f"   [OK] Recovered via param correction retry\n")
                        else:
                            # Stage 2: Fall back to LLM reasoning
                            print(f"   [~] Attempting LLM fallback...")
                            try:
                                fallback_result = self._execute_step_with_llm(
                                    step, task, step_results, context,
                                    error_context=f"Tool {tool_key} failed: {error_msg}"
                                )
                                step.result = fallback_result
                                step.status = PlanStatus.COMPLETED
                                step.end_time = time.time()
                                completed_steps.add(step.step_id)
                                step_results[step.step_id] = step.result
                                results['steps'].append(self._step_to_dict(step))
                                print(f"   [OK] Recovered via LLM reasoning\n")
                            except Exception:
                                step.status = PlanStatus.FAILED
                                step.error = error_msg
                                step.result = step_result.output
                                step.end_time = time.time()
                                failed_steps.add(step.step_id)
                                results['success'] = False
                                results['errors'].append({
                                    'step_id': step.step_id,
                                    'error': error_msg
                                })
                                results['steps'].append(self._step_to_dict(step))
                                print(f"   [X] All recovery failed\n")
                else:
                    # No tool specified - use LLM to execute
                    print(f"   [R] Using LLM reasoning")
                    step_result = self._execute_step_with_llm(step, task, step_results, context)
                    step.result = step_result
                    step.status = PlanStatus.COMPLETED
                    step.end_time = time.time()
                    completed_steps.add(step.step_id)
                    step_results[step.step_id] = step.result
                    results['steps'].append(self._step_to_dict(step))
                    print(f"   [OK] Completed")
                    print(f"   Duration: {step.end_time - step.start_time:.2f}s\n")

            except Exception as e:
                step.status = PlanStatus.FAILED
                step.error = str(e)
                step.end_time = time.time()

                failed_steps.add(step.step_id)
                results['success'] = False
                results['errors'].append({
                    'step_id': step.step_id,
                    'error': str(e)
                })

                results['steps'].append(self._step_to_dict(step))
                print(f"   [X] Failed: {e}\n")

        # Calculate blocked steps
        blocked_steps = [s for s in steps if s.status == PlanStatus.BLOCKED]

        # Mark as failed if any steps failed or were blocked
        if failed_steps or blocked_steps:
            results['success'] = False

        # Summary
        print(f"{'='*70}")
        print(f"[#] EXECUTION SUMMARY")
        print(f"{'='*70}")
        print(f"   Total steps: {len(steps)}")
        print(f"   Completed: {len(completed_steps)}")
        print(f"   Failed: {len(failed_steps)}")
        print(f"   Blocked: {len(blocked_steps)}")
        warnings = results.get('warnings', [])
        print(f"   Recovery attempts: {results['recovery_attempts']}")
        if warnings:
            print(f"   Warnings: {len(warnings)}")
        print(f"   Success: {results['success']}")
        print()

        return results

    def _resolve_params(
        self,
        params: Dict[str, Any],
        step_results: Dict[int, str],
        current_step_id: int = None
    ) -> Dict[str, Any]:
        """
        Resolve parameter values, injecting previous step results where needed.

        Strategy:
        - $stepN references get resolved to step N's output
        - $step_prev references get resolved to the previous step's output
        - Keys that suggest data input get last result if value is a placeholder
        """
        if not params:
            return params

        resolved = {}
        last_result = list(step_results.values())[-1] if step_results else None
        prev_step_id = current_step_id - 1 if current_step_id and current_step_id > 1 else None

        # Keys that typically need data from previous steps
        data_keys = {'data', 'input', 'list', 'items', 'content', 'text', 'numbers', 'code', 'source'}

        for key, value in params.items():
            if isinstance(value, str):
                # Handle $step_prev
                if '$step_prev' in value.lower():
                    if last_result:
                        resolved[key] = self._extract_result_data(last_result)
                        continue
                    else:
                        # No previous step available - skip this param with warning
                        print(f"   [!] No previous step result for '{key}' - param skipped")
                        continue

                # Handle $stepN references (e.g., $step1, $step2)
                step_ref = re.search(r'\$step(\d+)', value, re.IGNORECASE)
                if step_ref:
                    step_num = int(step_ref.group(1))
                    if step_num in step_results:
                        resolved[key] = self._extract_result_data(step_results[step_num])
                        continue
                    # If referenced step doesn't exist yet, try last result
                    elif last_result:
                        resolved[key] = self._extract_result_data(last_result)
                        continue
                    else:
                        # Referenced step not available - skip with warning
                        print(f"   [!] Step {step_num} result not available for '{key}' - param skipped")
                        continue

                # Check for angle bracket placeholders like <contents_of_file>
                if re.match(r'^<[^>]+>$', value) and last_result:
                    resolved[key] = self._extract_result_data(last_result)
                    continue

                # If key suggests data input and value looks like a placeholder
                if key.lower() in data_keys:
                    # Simple word or placeholder = inject last result
                    if re.match(r'^[\w\s\[\]]+$', value) and last_result:
                        resolved[key] = self._extract_result_data(last_result)
                        continue

                # Placeholder patterns
                if any(p in value.lower() for p in ['[list', '[result', 'previous', 'filtered']):
                    if last_result:
                        resolved[key] = self._extract_result_data(last_result)
                        continue

            # Default: pass through unchanged
            resolved[key] = value

        return resolved

    def _extract_result_data(self, result) -> Any:
        """
        Extract usable data from a step result.

        Since we now store step_result.output directly (not str(step_result)),
        this method is much simpler - it just returns the data as-is in most cases.
        """
        # If None, return None
        if result is None:
            return None

        # If already a useful type (list, dict, int, float), return as-is
        if isinstance(result, (list, dict, int, float, bool)):
            return result

        # For strings, return as-is (this is the common case now)
        return result

    def _execute_step_with_llm(
        self,
        step: PlanStep,
        task: str,
        previous_results: Dict,
        context: Optional[Dict],
        error_context: Optional[str] = None
    ) -> str:
        """Execute step using LLM when no tool is specified or as fallback."""
        # Truncate previous results to avoid prompt bloat
        truncated_results = {}
        for k, v in previous_results.items():
            v_str = str(v)
            truncated_results[k] = v_str[:500] + '...' if len(v_str) > 500 else v_str

        parts = [
            f"Original Task: {task}",
            f"Current Step: {step.description}",
        ]

        if error_context:
            parts.append(f"\nThe tool-based approach failed: {error_context}")
            parts.append("Use your reasoning to accomplish this step instead.")

        if truncated_results:
            parts.append(f"\nPrevious Results: {json.dumps(truncated_results, indent=2)}")

        if context:
            parts.append(f"\nContext: {json.dumps(context, indent=2)}")

        parts.append("\nProvide the result of this step concisely.")

        prompt = "\n".join(parts)
        from raec_core.model_swarm import TaskType
        return self.llm.generate(prompt, temperature=0.6, max_tokens=512, task_type=TaskType.REASONING)
    
    def _quick_verify_step_output(
        self, tool_key: str, output: Any, step: PlanStep
    ) -> Optional[str]:
        """
        Lightweight per-step verification. Returns a warning string if the output
        looks problematic, or None if it looks OK. Does NOT use the LLM.
        """
        if output is None:
            return "Output is None — contains error"

        output_str = str(output)

        # Check for empty output (string, list, dict)
        if not output_str.strip():
            return "Output is empty — contains error"

        # Empty collections are failures, not soft warnings
        if output == [] or output == {} or output == "[]" or output == "{}":
            return "Output is empty collection — contains error"

        # Very short output from web fetches is suspicious (e.g. just "Sign in")
        web_tools = {'web.fetch_text', 'web.fetch_json', 'web.fetch_links'}
        if tool_key in web_tools and isinstance(output, str) and len(output.strip()) < 50:
            return f"Output too short for web fetch ({len(output.strip())} chars) — contains error"

        # Check for error strings in output (tool returned an error as a value)
        error_prefixes = [
            'Error:', 'HTTP GET error:', 'HTTP POST error:',
            'Fetch error:', 'Download error:', 'Execution error:',
            'Command error:', 'JSON parse error:',
        ]
        for prefix in error_prefixes:
            if output_str.startswith(prefix):
                return f"Output contains error: {output_str[:120]}"

        # Check for HTML in output when tool shouldn't return HTML
        non_html_tools = {
            'web.fetch_text', 'web.fetch_json', 'data.parse_json',
            'text.search_text', 'data.filter_list', 'data.count',
            'code.run_python', 'code.run_shell',
        }
        if tool_key in non_html_tools and isinstance(output, str):
            if '<html' in output_str[:200].lower() or '<!doctype' in output_str[:200].lower():
                return "Output contains unexpected HTML"

        # Check for error dicts
        if isinstance(output, dict) and 'error' in output and len(output) <= 3:
            return f"Output is error dict: {output.get('error', '')[:120]}"

        return None

    def _retry_with_corrected_params(
        self,
        step: PlanStep,
        tool_key: str,
        original_params: Dict,
        error_msg: str,
        task: str
    ) -> Optional[Any]:
        """
        Ask the LLM to correct tool parameters and retry the tool call once.
        Returns the tool output on success, or None if retry also fails.
        """
        # Only retry for param-related or HTTP errors, not for systemic issues
        retryable_hints = [
            'argument', 'param', 'type', 'missing', 'invalid', 'expected',
            '400', '404', '403', '500', 'error:', 'timeout',
        ]
        error_lower = error_msg.lower()
        if not any(hint in error_lower for hint in retryable_hints):
            return None

        print(f"   [~] Attempting param correction for {tool_key}...")

        # Get the tool's actual signature for the LLM
        fn = self.tools.interface.tools.get(tool_key)
        if not fn:
            return None

        try:
            sig = inspect.signature(fn)
            sig_str = str(sig)
        except (ValueError, TypeError):
            sig_str = "(unknown)"

        prompt = f"""A tool call failed. Fix the parameters and respond with ONLY the corrected JSON params.

Tool: {tool_key}{sig_str}
Step description: {step.description}
Original task: {task}
Original params: {json.dumps(original_params, default=str)[:500]}
Error: {error_msg[:300]}

Respond with ONLY a JSON object containing the corrected parameters. No explanation.
Example: {{"param1": "value1", "param2": "value2"}}
"""
        try:
            from raec_core.model_swarm import TaskType
            correction = self.llm.generate(
                prompt, temperature=0.3, max_tokens=256,
                task_type=TaskType.PARAM_GENERATION
            )

            # Parse the corrected params
            json_match = re.search(r'\{[^{}]+\}', correction, re.DOTALL)
            if not json_match:
                return None

            corrected_params = json.loads(json_match.group(0))

            # Validate corrected params
            corrected_step = PlanStep(
                step.step_id, step.description,
                step.tool_category, step.tool_name, corrected_params
            )
            validated = self._validate_params(corrected_step, tool_key, fn)

            if not validated:
                return None

            # Retry the tool call
            print(f"   [~] Retrying {tool_key} with corrected params: {validated}")
            retry_result = self.tools.execute(tool_key, **validated)

            if retry_result.success:
                return retry_result.output
            else:
                print(f"   [!] Retry also failed: {retry_result.error}")
                return None

        except Exception as e:
            print(f"   [!] Param correction failed: {e}")
            return None

    def _step_to_dict(self, step: PlanStep) -> Dict:
        """Convert step to dictionary"""
        return {
            'step_id': step.step_id,
            'description': step.description,
            'tool': f"{step.tool_category}.{step.tool_name}" if step.tool_category else None,
            'params': step.tool_params,
            'status': step.status.value,
            'result': step.result,
            'error': step.error,
            'duration': step.end_time - step.start_time if step.end_time and step.start_time else None
        }
    
    def _store_plan(self, task: str, steps: List[PlanStep]) -> int:
        """Store plan in memory"""
        try:
            from memory.memory_db import MemoryType
            
            plan_text = f"Plan for: {task}\n"
            plan_text += "\n".join([
                f"{s.step_id}. {s.description}" + 
                (f" [Tool: {s.tool_category}.{s.tool_name}]" if s.tool_category else "")
                for s in steps
            ])
            
            plan_id = self.memory.store(
                content=plan_text,
                memory_type=MemoryType.EXPERIENCE,
                metadata={
                    'task': task,
                    'num_steps': len(steps),
                    'plan_type': 'generated'
                },
                source='planner'
            )
            
            return plan_id
        except Exception as e:
            print(f"[!] Failed to store plan: {e}")
            return -1
    
    def _store_execution_results(self, task: str, results: Dict, plan_id: int):
        """Store execution results in memory, with failure analysis for failed plans."""
        try:
            from memory.memory_db import MemoryType

            result_text = f"Execution results for: {task}\n"
            result_text += f"Success: {results['success']}\n"
            result_text += f"Completed steps: {len([s for s in results['steps'] if s['status'] == 'completed'])}"

            result_id = self.memory.store(
                content=result_text,
                memory_type=MemoryType.EXPERIENCE,
                metadata={
                    'task': task,
                    'success': results['success'],
                    'num_steps': len(results['steps']),
                    'errors': results.get('errors', [])
                },
                confidence=1.0 if results['success'] else 0.5,
                source='planner_execution',
                linked_to=[plan_id] if plan_id > 0 else None
            )

            # W11: Failure analysis — extract beliefs from failed plans
            if not results['success'] and results.get('errors'):
                self._analyze_failure_and_store_beliefs(
                    task, results, result_id
                )

            # Note: Skill extraction is handled centrally in main.py after verification
            # to avoid duplicate skill creation

        except Exception as e:
            print(f"[!] Failed to store execution results: {e}")

    def _analyze_failure_and_store_beliefs(
        self, task: str, results: Dict, experience_id: int
    ):
        """
        Analyze a failed plan execution and store structured beliefs
        so future plans can avoid the same mistakes.

        Extracts lessons from each error:
        - Which tool failed and why
        - What approach didn't work
        - What should be done differently
        """
        from memory.memory_db import MemoryType

        errors = results.get('errors', [])
        steps = results.get('steps', [])

        if not errors:
            return

        print("\n[~] Analyzing failure for belief formation...")

        # Build structured error chain
        error_chain = []
        for err in errors:
            step_id = err.get('step_id')
            error_msg = str(err.get('error', 'unknown'))[:300]

            # Find the step details
            step_info = next(
                (s for s in steps if s.get('step_id') == step_id), {}
            )
            tool_used = step_info.get('tool', 'unknown')
            step_desc = step_info.get('description', '')[:200]

            error_chain.append({
                'step_id': step_id,
                'tool': tool_used,
                'description': step_desc,
                'error': error_msg,
            })

        # Build a concise prompt for the LLM to extract beliefs
        chain_text = "\n".join(
            f"- Step {e['step_id']} ({e['tool']}): {e['description']}\n  Error: {e['error']}"
            for e in error_chain[:5]
        )

        prompt = f"""A plan failed. Analyze the errors and extract 1-3 short lessons (beliefs) to avoid this failure in the future.

Task: {task[:300]}
Error chain:
{chain_text}

For each lesson, write ONE concise sentence stating what to do or avoid.
Format each as a separate line starting with "BELIEF:".
Example:
BELIEF: arXiv API requires bracket syntax for date ranges, not slash syntax.
BELIEF: HuggingFace blog pages return HTML, use web.fetch_text not web.fetch_json.
"""
        try:
            from raec_core.model_swarm import TaskType
            analysis = self.llm.generate(
                prompt, temperature=0.3, max_tokens=300,
                task_type=TaskType.REASONING
            )

            # Parse beliefs from response
            beliefs = []
            for line in analysis.split('\n'):
                line = line.strip()
                if line.upper().startswith('BELIEF:'):
                    belief_text = line[len('BELIEF:'):].strip()
                    if belief_text and len(belief_text) > 10:
                        beliefs.append(belief_text)

            # Store each belief in memory
            for belief_text in beliefs[:3]:  # Cap at 3 beliefs per failure
                belief_id = self.memory.store(
                    content=belief_text,
                    memory_type=MemoryType.BELIEF,
                    metadata={
                        'source_task': task[:200],
                        'error_chain': error_chain[:3],
                        'auto_extracted': True,
                    },
                    confidence=0.7,  # Start moderate, evolve with evidence
                    source='failure_analysis',
                    linked_to=[experience_id]
                )
                print(f"   [+] Belief stored: {belief_text}")

            if not beliefs:
                # Fallback: store a simple heuristic belief from the first error
                first_err = error_chain[0] if error_chain else {}
                fallback_belief = (
                    f"Tool {first_err.get('tool', '?')} failed for task like "
                    f"'{task[:80]}': {first_err.get('error', 'unknown')[:120]}"
                )
                self.memory.store(
                    content=fallback_belief,
                    memory_type=MemoryType.BELIEF,
                    metadata={
                        'source_task': task[:200],
                        'auto_extracted': True,
                        'fallback': True,
                    },
                    confidence=0.5,
                    source='failure_analysis',
                    linked_to=[experience_id]
                )
                print(f"   [+] Fallback belief stored: {fallback_belief[:100]}...")

        except Exception as e:
            print(f"   [!] Failure analysis failed: {e}")


# Backward compatibility
Planner = ToolEnabledPlanner
