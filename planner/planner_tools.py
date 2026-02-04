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
        
        # Tool interface
        if tools is None:
            from tools.tool_interface import ToolInterface
            self.tools = ToolInterface()
        else:
            self.tools = tools
    
    def run(self, task: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute a task with planning and tool usage
        """
        print(f"\n{'='*70}")
        print(f"🎯 TASK: {task}")
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
            
            print("🔍 Searching memory for similar past tasks...")
            similar = self.memory.query(
                query_text=task,
                memory_types=[MemoryType.EXPERIENCE, MemoryType.SUMMARY],
                k=3,
                min_confidence=0.7
            )
            
            if similar:
                print(f"   ✓ Found {len(similar)} relevant past experiences\n")
            else:
                print("   • No similar past experiences found\n")
            
            return similar
        except Exception as e:
            print(f"   ⚠ Memory query failed: {e}\n")
            return []
    
    def _generate_plan_with_tools(
        self,
        task: str,
        similar_experiences: List[Dict],
        context: Optional[Dict]
    ) -> List[PlanStep]:
        """Generate a plan with tool assignments"""
        print("📋 Generating execution plan with tool assignments...\n")
        
        # Get available tools for LLM
        from tools.tool_interface import get_tools_for_llm
        tools_doc = get_tools_for_llm()
        
        # Build prompt
        prompt = self._build_planning_prompt_with_tools(
            task, similar_experiences, context, tools_doc
        )
        
        try:
            # Generate plan from LLM
            plan_text = self.llm.generate(prompt, temperature=0.5, max_tokens=2048)
            
            # Parse plan into steps with tool assignments
            steps = self._parse_plan_with_tools(plan_text)
            
            print(f"✓ Generated plan with {len(steps)} steps:\n")
            for step in steps:
                tool_info = f" [{step.tool_category}.{step.tool_name}]" if step.tool_category else ""
                deps = f" (depends on: {step.dependencies})" if step.dependencies else ""
                print(f"   {step.step_id}. {step.description}{tool_info}{deps}")
            print()
            
            return steps
            
        except Exception as e:
            print(f"⚠ Plan generation failed: {e}")
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
        
        prompt_parts.append("""
\n**Instructions:**
1. Break the task into clear, actionable steps.
2. For each step that requires a tool, specify:
   - TOOL: category.tool_name
   - PARAMS: {param1: value1, param2: value2}
3. Number steps sequentially (1, 2, 3...).
4. If a step depends on another, add [DEPENDS: X].

Example Output Format:
1. Read the input file
   TOOL: file.read_file
   PARAMS: {filepath: "input.txt"}

2. Process the data [DEPENDS: 1]
   TOOL: code.run_python
   PARAMS: {code: "print('processing')"}

3. Write output [DEPENDS: 2]
   TOOL: file.write_file
   PARAMS: {filepath: "output.txt", content: "result"}

Generate the plan now:
""")
        
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
        print("⚙️  EXECUTING PLAN WITH TOOLS\n")
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
                print(f"⏸️  Step {step.step_id}: BLOCKED (dependencies not met)")
                continue
            
            # Execute step
            print(f"▶️  Step {step.step_id}: {step.description}")
            step.status = PlanStatus.IN_PROGRESS
            step.start_time = time.time()
            
            try:
                # Execute using tools if specified
                if step.tool_category and step.tool_name:
                    print(f"   🔧 Using tool: {step.tool_category}.{step.tool_name}")
                    print(f"   📥 Params: {step.tool_params}")
                    
                    step_result = self.tools.execute(
                        step.tool_category,
                        step.tool_name,
                        **step.tool_params
                    )
                    
                    step.result = str(step_result)
                    step.status = PlanStatus.COMPLETED
                    print(f"   ✓ Result: {str(step_result)[:100]}...")
                else:
                    # No tool specified - use LLM to execute
                    print(f"   🤖 Using LLM reasoning")
                    step_result = self._execute_step_with_llm(step, task, step_results, context)
                    step.result = step_result
                    step.status = PlanStatus.COMPLETED
                    print(f"   ✓ Completed")
                
                step.end_time = time.time()
                completed_steps.add(step.step_id)
                step_results[step.step_id] = step.result
                
                results['steps'].append(self._step_to_dict(step))
                print(f"   ⏱️  Duration: {step.end_time - step.start_time:.2f}s\n")
                
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
                print(f"   ✗ Failed: {e}\n")
        
        # Summary
        print(f"{'='*70}")
        print(f"📊 EXECUTION SUMMARY")
        print(f"{'='*70}")
        print(f"   Total steps: {len(steps)}")
        print(f"   Completed: {len(completed_steps)}")
        print(f"   Failed: {len(failed_steps)}")
        print(f"   Blocked: {len([s for s in steps if s.status == PlanStatus.BLOCKED])}")
        print(f"   Success: {results['success']}")
        print()
        
        return results
    
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
            print(f"⚠ Failed to store plan: {e}")
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
            print(f"⚠ Failed to store execution results: {e}")
    
    def _consider_skill_extraction(self, task: str, results: Dict, result_id: int):
        """Consider extracting successful execution as a skill"""
        print(f"💡 Task completed successfully - candidate for skill extraction")


# Backward compatibility
Planner = ToolEnabledPlanner
