"""
Enhanced Planner Module
Multi-step reasoning with memory integration and error correction
"""
import json
import time
from typing import List, Dict, Optional, Any
from enum import Enum
import re


class PlanStatus(Enum):
    """Status of plan execution"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class PlanStep:
    """Individual step in a plan"""
    def __init__(
        self,
        step_id: int,
        description: str,
        dependencies: Optional[List[int]] = None,
        status: PlanStatus = PlanStatus.PENDING,
        result: Optional[str] = None,
        error: Optional[str] = None
    ):
        self.step_id = step_id
        self.description = description
        self.dependencies = dependencies or []
        self.status = status
        self.result = result
        self.error = error
        self.start_time = None
        self.end_time = None


class Planner:
    """
    Advanced planner with:
    - Multi-step reasoning
    - Memory-augmented planning
    - Error correction loops
    - Skill reuse from memory
    """
    
    def __init__(self, llm, memory, skills, max_steps=10):
        self.llm = llm
        self.memory = memory
        self.skills = skills
        self.max_steps = max_steps
        self.current_plan = None
        self.execution_history = []
    
    def run(self, task: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute a task with planning and memory integration
        """
        print(f"\n{'='*70}")
        print(f"🎯 TASK: {task}")
        print(f"{'='*70}\n")
        
        # Check for similar past tasks in memory
        similar_experiences = self._retrieve_similar_tasks(task)
        
        # Generate plan with memory augmentation
        plan_steps = self._generate_plan(task, similar_experiences, context)
        
        # Store plan in memory as EXPERIENCE
        plan_memory_id = self._store_plan(task, plan_steps)
        
        # Execute plan
        results = self._execute_plan(plan_steps, task)
        
        # Store execution results
        self._store_execution_results(task, results, plan_memory_id)
        
        return {
            'task': task,
            'plan_id': plan_memory_id,
            'steps': plan_steps,
            'results': results,
            'success': results.get('success', False)
        }
    
    def _retrieve_similar_tasks(self, task: str) -> List[Dict]:
        """Query memory for similar past tasks"""
        try:
            # Import MemoryType here to avoid circular imports
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
    
    def _generate_plan(
        self,
        task: str,
        similar_experiences: List[Dict],
        context: Optional[Dict]
    ) -> List[PlanStep]:
        """Generate a multi-step plan using LLM"""
        print("📋 Generating execution plan...\n")
        
        # Build prompt with memory augmentation
        prompt = self._build_planning_prompt(task, similar_experiences, context)
        
        try:
            # Generate plan from LLM
            plan_text = self.llm.generate(prompt, temperature=0.5, max_tokens=1024)
            
            # Parse plan into structured steps
            steps = self._parse_plan(plan_text)
            
            print(f"✓ Generated plan with {len(steps)} steps:\n")
            for step in steps:
                deps = f" (depends on: {step.dependencies})" if step.dependencies else ""
                print(f"   {step.step_id}. {step.description}{deps}")
            print()
            
            return steps
            
        except Exception as e:
            print(f"⚠ Plan generation failed: {e}")
            # Fallback to simple single-step plan
            return [PlanStep(1, f"Complete task: {task}")]
    
    def _build_planning_prompt(
        self,
        task: str,
        similar_experiences: List[Dict],
        context: Optional[Dict]
    ) -> str:
        """Build planning prompt with memory and context"""
        # SCRUBBED: Removed "You are Raec..." persona introduction.
        # REPLACED WITH: Pure functional role definition.
        prompt_parts = [
            "ROLE: Technical Planning Algorithm.",
            "OBJECTIVE: Break the user's task into a sequential, logical execution plan.",
            f"\n**USER TASK:**\n{task}",
        ]
        
        if context:
            prompt_parts.append(f"\n**Context:**\n{json.dumps(context, indent=2)}")
        
        if similar_experiences:
            prompt_parts.append("\n**Relevant Past Experiences:**")
            for exp in similar_experiences[:3]:
                prompt_parts.append(f"- {exp['content']}")
        
        prompt_parts.append("""
\n**Instructions:**
1. Break the task into clear, actionable steps.
2. Number them sequentially (1, 2, 3...).
3. If a step depends on a previous one, add [DEPENDS: X].

Example Output Format:
1. Research the problem domain
2. Design the solution architecture [DEPENDS: 1]
3. Implement core components [DEPENDS: 2]
4. Test and validate [DEPENDS: 3]

Generate the plan now:
""")
        
        return "\n".join(prompt_parts)
    
    def _parse_plan(self, plan_text: str) -> List[PlanStep]:
        """Parse LLM output into structured plan steps"""
        steps = []
        lines = plan_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Match numbered steps: "1. Description" or "1) Description"
            match = re.match(r'^(\d+)[.)]\s*(.+)$', line)
            if match:
                step_id = int(match.group(1))
                description = match.group(2).strip()
                
                # Extract dependencies: [DEPENDS: 1,2]
                dependencies = []
                dep_match = re.search(r'\[DEPENDS:\s*([\d,\s]+)\]', description, re.IGNORECASE)
                if dep_match:
                    dep_str = dep_match.group(1)
                    dependencies = [int(x.strip()) for x in dep_str.split(',') if x.strip().isdigit()]
                    # Remove dependency notation from description
                    description = re.sub(r'\[DEPENDS:[^\]]+\]', '', description, flags=re.IGNORECASE).strip()
                
                steps.append(PlanStep(step_id, description, dependencies))
        
        # If parsing failed, create single step
        if not steps:
            steps.append(PlanStep(1, plan_text.strip()[:200]))
        
        return steps
    
    def _execute_plan(self, steps: List[PlanStep], task: str) -> Dict[str, Any]:
        """Execute plan steps with dependency resolution"""
        print("⚙️  EXECUTING PLAN\n")
        print(f"{'='*70}\n")
        
        completed_steps = set()
        failed_steps = set()
        results = {
            'success': True,
            'steps': [],
            'errors': []
        }
        
        for step in steps:
            # Check dependencies
            if not all(dep in completed_steps for dep in step.dependencies):
                step.status = PlanStatus.BLOCKED
                results['steps'].append({
                    'step_id': step.step_id,
                    'description': step.description,
                    'status': 'blocked',
                    'reason': f"Waiting for steps: {[d for d in step.dependencies if d not in completed_steps]}"
                })
                print(f"⏸️  Step {step.step_id}: BLOCKED (dependencies not met)")
                continue
            
            # Execute step
            print(f"▶️  Step {step.step_id}: {step.description}")
            step.status = PlanStatus.IN_PROGRESS
            step.start_time = time.time()
            
            try:
                # Execute step (placeholder - actual execution would use tools)
                step_result = self._execute_step(step, task)
                step.result = step_result
                step.status = PlanStatus.COMPLETED
                step.end_time = time.time()
                
                completed_steps.add(step.step_id)
                results['steps'].append({
                    'step_id': step.step_id,
                    'description': step.description,
                    'status': 'completed',
                    'result': step_result,
                    'duration': step.end_time - step.start_time
                })
                print(f"   ✓ Completed ({step.end_time - step.start_time:.2f}s)")
                
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
                print(f"   ✗ Failed: {e}")
            
            print()
        
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
    
    def _execute_step(self, step: PlanStep, task: str) -> str:
        """
        Execute individual step (placeholder for actual tool execution)
        """
        # For now, simulate execution
        time.sleep(0.1)  # Simulate work
        return f"Executed: {step.description}"
    
    def _store_plan(self, task: str, steps: List[PlanStep]) -> int:
        """Store plan in memory as EXPERIENCE"""
        try:
            from memory.memory_db import MemoryType
            
            plan_text = f"Plan for: {task}\n"
            plan_text += "\n".join([f"{s.step_id}. {s.description}" for s in steps])
            
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
            print(f"⚠ Failed to store plan in memory: {e}")
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
        """Determine if successful execution should be extracted as a skill"""
        # This will be enhanced when we build the skill graph
        print(f"💡 Task completed successfully - candidate for skill extraction")