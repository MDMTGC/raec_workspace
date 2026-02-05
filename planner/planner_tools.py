"""
Tool-Enabled Planner Module
Plans and executes tasks using actual tools
"""
import json
import time
import re
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
        Execute a task with planning and tool usage
        """
        print(f"\n{'='*70}")
        print(f"[o] TASK: {task}")
        print(f"{'='*70}\n")
        
        # Check for similar past tasks in memory
        similar_experiences = self._retrieve_similar_tasks(task)
        
        # Generate plan with tool assignment
        plan_steps = self._generate_plan_with_tools(task, similar_experiences, context)
        
        # Store plan in memory
        plan_memory_id = self._store_plan(task, plan_steps)
        
        # Execute plan using tools
        results = self._execute_plan_with_tools(plan_steps, task, context)
        
        # Store execution results
        self._store_execution_results(task, results, plan_memory_id)
        
        return {
            'task': task,
            'plan_id': plan_memory_id,
            'steps': [self._step_to_dict(s) for s in plan_steps],
            'results': results,
            'success': results.get('success', False)
        }
    
    def _retrieve_similar_tasks(self, task: str) -> List[Dict]:
        """Query memory for similar past tasks"""
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
                print(f"   [OK] Found {len(similar)} relevant past experiences\n")
            else:
                print("   • No similar past experiences found\n")
            
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
            # Generate plan from LLM
            plan_text = self.llm.generate(prompt, temperature=0.5, max_tokens=2048)
            
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
\n**Instructions:**
1. Break the task into clear, actionable steps.
2. For each step that requires a tool, specify:
   - TOOL: category.tool_name
   - PARAMS: {"param_name": "value"} (use exact parameter names from tool signatures)
3. Number steps sequentially (1, 2, 3...).
4. If a step depends on another, add [DEPENDS: X].

Example Output Format:
1. List directory contents
   TOOL: file.list_directory
   PARAMS: {"dirpath": "."}

2. Read the input file [DEPENDS: 1]
   TOOL: file.read_file
   PARAMS: {"filepath": "input.txt"}

3. Process the data [DEPENDS: 2]
   TOOL: code.run_python
   PARAMS: {"code": "result = data.upper()"}

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
                try:
                    params_str = params_match.group(1)
                    # Try to parse as dict
                    current_params = eval(params_str)
                except:
                    pass
        
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
        
        return steps
    
    def _execute_plan_with_tools(
        self,
        steps: List[PlanStep],
        task: str,
        context: Optional[Dict]
    ) -> Dict[str, Any]:
        """Execute plan steps using actual tools"""
        print("[*]  EXECUTING PLAN WITH TOOLS\n")
        print(f"{'='*70}\n")
        
        completed_steps = set()
        failed_steps = set()
        step_results = {}
        
        results = {
            'success': True,
            'steps': [],
            'errors': []
        }
        
        for step in steps:
            # Check dependencies
            if not all(dep in completed_steps for dep in step.dependencies):
                step.status = PlanStatus.BLOCKED
                results['steps'].append(self._step_to_dict(step))
                print(f"[||]  Step {step.step_id}: BLOCKED (dependencies not met)")
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
                    resolved_params = self._resolve_params(step.tool_params, step_results)

                    print(f"   Using tool: {tool_key}")
                    print(f"   Params: {resolved_params}")

                    step_result = self.tools.execute(
                        tool_key,
                        **resolved_params
                    )
                    
                    step.result = str(step_result)
                    step.status = PlanStatus.COMPLETED
                    print(f"   [OK] Result: {str(step_result)[:100]}...")
                else:
                    # No tool specified - use LLM to execute
                    print(f"   [R] Using LLM reasoning")
                    step_result = self._execute_step_with_llm(step, task, step_results, context)
                    step.result = step_result
                    step.status = PlanStatus.COMPLETED
                    print(f"   [OK] Completed")
                
                step.end_time = time.time()
                completed_steps.add(step.step_id)
                step_results[step.step_id] = step.result
                
                results['steps'].append(self._step_to_dict(step))
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
        
        # Summary
        print(f"{'='*70}")
        print(f"[#] EXECUTION SUMMARY")
        print(f"{'='*70}")
        print(f"   Total steps: {len(steps)}")
        print(f"   Completed: {len(completed_steps)}")
        print(f"   Failed: {len(failed_steps)}")
        print(f"   Blocked: {len([s for s in steps if s.status == PlanStatus.BLOCKED])}")
        print(f"   Success: {results['success']}")
        print()
        
        return results

    def _resolve_params(
        self,
        params: Dict[str, Any],
        step_results: Dict[int, str]
    ) -> Dict[str, Any]:
        """
        Resolve parameter values, injecting previous step results where needed.

        Strategy:
        - If param key suggests data input (data, input, list, items, content, text)
          and value is a simple word, inject last result
        - Explicit references like $step1 get resolved
        - Concrete values (paths, expressions) pass through
        """
        import re

        if not params or not step_results:
            return params

        resolved = {}
        last_result = list(step_results.values())[-1] if step_results else None

        # Keys that typically need data from previous steps
        data_keys = {'data', 'input', 'list', 'items', 'content', 'text', 'numbers', 'code', 'source'}

        for key, value in params.items():
            # If key suggests data input and value looks like a placeholder
            if key.lower() in data_keys and isinstance(value, str):
                # Check for angle bracket placeholders like <contents_of_file>
                if re.match(r'^<[^>]+>$', value) and last_result:
                    resolved[key] = self._extract_result_data(last_result)
                    continue
                # Simple word or placeholder = inject last result
                if re.match(r'^[\w\s\[\]]+$', value) and last_result:
                    # Try to extract actual data from ToolResult
                    resolved[key] = self._extract_result_data(last_result)
                    continue

            if isinstance(value, str):
                lower_val = value.lower()

                # Explicit step references
                step_ref = re.search(r'\$?step[_\s]*(\d+)', lower_val)
                if step_ref:
                    step_num = int(step_ref.group(1))
                    if step_num in step_results:
                        resolved[key] = self._extract_result_data(step_results[step_num])
                        continue

                # Placeholder patterns
                if any(p in lower_val for p in ['[list', '[result', 'previous', 'filtered']):
                    if last_result:
                        resolved[key] = self._extract_result_data(last_result)
                        continue

            # Default: pass through unchanged
            resolved[key] = value

        return resolved

    def _extract_result_data(self, result_str: str) -> Any:
        """Extract actual data from a ToolResult string or return as-is"""
        import re
        import ast

        result_str = str(result_str)

        # Try to extract output from ToolResult string
        if 'output=' in result_str:
            # Find the output= part and extract the value
            output_start = result_str.find('output=')
            if output_start != -1:
                rest = result_str[output_start + 7:]  # After 'output='

                # Handle None
                if rest.startswith('None'):
                    return None

                # Handle list [...] - match balanced brackets
                if rest.startswith('['):
                    bracket_count = 0
                    end_idx = 0
                    for i, char in enumerate(rest):
                        if char == '[':
                            bracket_count += 1
                        elif char == ']':
                            bracket_count -= 1
                            if bracket_count == 0:
                                end_idx = i + 1
                                break
                    if end_idx > 0:
                        list_str = rest[:end_idx]
                        try:
                            return ast.literal_eval(list_str)
                        except:
                            return list_str

                # Handle string '...' - find matching quote
                if rest.startswith("'"):
                    end_idx = 1
                    while end_idx < len(rest):
                        if rest[end_idx] == "'" and (end_idx == 1 or rest[end_idx-1] != '\\'):
                            break
                        end_idx += 1
                    if end_idx < len(rest):
                        return rest[1:end_idx]

                # Handle dict {...} - match balanced braces
                if rest.startswith('{'):
                    brace_count = 0
                    end_idx = 0
                    for i, char in enumerate(rest):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = i + 1
                                break
                    if end_idx > 0:
                        dict_str = rest[:end_idx]
                        try:
                            return ast.literal_eval(dict_str)
                        except:
                            return dict_str

                # Handle number
                num_match = re.match(r'^(\d+(?:\.\d+)?)', rest)
                if num_match:
                    num_str = num_match.group(1)
                    return float(num_str) if '.' in num_str else int(num_str)

        return result_str

    def _execute_step_with_llm(
        self,
        step: PlanStep,
        task: str,
        previous_results: Dict,
        context: Optional[Dict]
    ) -> str:
        """Execute step using LLM when no tool is specified"""
        prompt = f"""
Execute this step:

Original Task: {task}
Current Step: {step.description}
Previous Results: {json.dumps(previous_results, indent=2) if previous_results else 'None'}
Context: {json.dumps(context, indent=2) if context else 'None'}

Provide the result of this step.
"""
        return self.llm.generate(prompt, temperature=0.6, max_tokens=512)
    
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
        """Store execution results in memory"""
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
            
            if results['success']:
                self._consider_skill_extraction(task, results, result_id)
                
        except Exception as e:
            print(f"[!] Failed to store execution results: {e}")
    
    def _consider_skill_extraction(self, task: str, results: Dict, result_id: int):
        """Consider extracting successful execution as a skill"""
        print(f"[!] Task completed successfully - candidate for skill extraction")


# Backward compatibility
Planner = ToolEnabledPlanner
