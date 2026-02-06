"""
Raec - Autonomous Reasoning and Execution Core
Complete integration with all upgraded components

Fixed Integration:
- All imports corrected
- Method calls aligned
- Proper error handling
- Three execution modes working
"""
import sys
import os
import yaml
import time
from typing import Dict, Any, Optional

# Path Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Core imports - use swarm-enabled LLM interface
from raec_core.model_swarm import LLMInterface, TaskType
from planner.planner_tools import ToolEnabledPlanner as Planner
from memory.memory_db import HierarchicalMemoryDB, MemoryType
from skills.skill_graph import SkillGraph, SkillCategory, SkillStatus
from tools.executor import ToolExecutor, ToolType
from agents.orchestrator import MultiAgentOrchestrator, AgentRole
from evaluators.logic_checker import LogicChecker, VerificationLevel
from raec_core.core_rules import CoreRulesEngine


class Raec:
    """
    Main Raec system integrating all components:
    - Hierarchical Memory (Facts, Experiences, Beliefs, Summaries)
    - Audited Skill Graph (ASG-SI with verification)
    - Runtime Tool Evolution (Live-SWE-agent patterns)
    - Multi-Agent Orchestration (AutoGen/CrewAI)
    - Advanced Verification (ToolReflection + incremental)
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        print("\n" + "="*70)
        print("[R] INITIALIZING RAEC SYSTEM")
        print("="*70 + "\n")
        
        # Load configuration
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_config_path = os.path.join(base_dir, config_path)
        
        with open(full_config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize LLM with swarm routing
        print("[*]  Initializing LLM interface...")
        swarm_config_path = os.path.join(base_dir, "config/swarm_config.json")
        swarm_config_exists = os.path.exists(swarm_config_path)
        if not swarm_config_exists:
            print(f"   [!] Swarm config not found at {swarm_config_path}")
            print("       Using default model routing (no task-specific routing)")
        self.llm = LLMInterface(
            model=self.config['model']['name'],
            config_path=swarm_config_path if swarm_config_exists else None,
            use_swarm=True
        )
        print("   [OK] Connected to model:", self.config['model']['name'])
        if self.llm.swarm:
            print("   [OK] Model swarm enabled")
        
        # Initialize subsystems
        db_path = os.path.join(base_dir, self.config['memory']['db_path'])
        skill_path = os.path.join(base_dir, "skills/skill_db.json")
        
        print("\n[*]  Initializing subsystems...")
        
        # Memory
        self.memory = HierarchicalMemoryDB(db_path=db_path)
        print("   [OK] Hierarchical Memory initialized")
        
        # Skills
        self.skills = SkillGraph(storage_path=skill_path)
        print("   [OK] Skill Graph loaded")
        
        # Tools
        self.tools = ToolExecutor(
            python_timeout=self.config['tools'].get('python_timeout', 60)
        )
        print("   [OK] Tool Executor ready")
        
        # Multi-Agent Orchestrator
        self.orchestrator = MultiAgentOrchestrator(self.llm)
        self._initialize_agents()
        print("   [OK] Multi-Agent Orchestrator configured")
        
        # Verification
        self.evaluator = LogicChecker(self.llm)
        print("   [OK] Logic Checker enabled")

        # Core Rules Engine (immutable constraint layer)
        self.rules = CoreRulesEngine()
        print("   [OK] Core Rules Engine active")
        
        # Planner (integrates memory, skills, and tools)
        self.planner = Planner(
            self.llm,
            self.memory,
            self.skills,
            tools=self.tools,
            max_steps=self.config['planner'].get('max_steps', 10)
        )
        print("   [OK] Planner initialized")
        
        # System directive
        self.logic_directive = "You are a technical reasoning engine. Provide objective analysis."

        # Action tracking for rules engine
        self._recent_actions = []
        self._max_action_history = 100

        # Memory curation settings
        self._memory_curation_threshold = 50  # Curate after this many experiences
        self._last_curation_count = 0

        print("\n" + "="*70)
        print("[OK] RAEC SYSTEM ONLINE")
        print("="*70 + "\n")
    
    def _initialize_agents(self):
        """Set up default agents for multi-agent mode"""
        self.orchestrator.create_agent(
            AgentRole.PLANNER,
            capabilities=["task_decomposition", "planning"],
            description="Plans and breaks down complex tasks"
        )
        
        self.orchestrator.create_agent(
            AgentRole.EXECUTOR,
            capabilities=["execution", "implementation"],
            description="Executes plans and implements solutions"
        )
        
        self.orchestrator.create_agent(
            AgentRole.CRITIC,
            capabilities=["review", "quality_assurance"],
            description="Reviews and validates work quality"
        )
    
    def process_input(self, task: str, mode: str = "standard") -> str:
        """
        Main entry point for processing user input
        
        Args:
            task: User's task
            mode: Execution mode (standard, collaborative, incremental)
            
        Returns:
            Result string
        """
        # Execute with specified mode
        full_task = f"{self.logic_directive}\n\nTask: {task}"
        result = self.execute_task(full_task, mode=mode)
        
        if not result.get('success'):
            return f"Error: {result.get('error', 'Execution failed')}"
        
        # Synthesize output
        synthesis_prompt = f"""
Task: {task}
Execution Data: {str(result.get('steps', []))[:500]}

Based on the execution data, provide a direct, concise response.
"""
        
        return self.llm.generate(synthesis_prompt, temperature=0.5, max_tokens=512)
    
    def execute_task(
        self,
        task: str,
        mode: str = "standard",
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Execute a task with specified mode
        
        Modes:
        - standard: Normal planning and execution with memory/skills
        - collaborative: Multi-agent workflow with self-correction
        - incremental: Step-by-step execution with verification
        
        Args:
            task: Task to execute
            mode: Execution mode
            context: Additional context
            
        Returns:
            Execution result dictionary
        """
        print(f"\n{'='*70}")
        print(f"[o] EXECUTING TASK (Mode: {mode})")
        print(f"{'='*70}")
        print(f"Task: {task}\n")

        # Gate through core rules
        allowed, modified_data, messages = self.rules.gate(
            action_type='task_execution',
            action_data={'task': task, 'mode': mode},
            context={'recent_actions': getattr(self, '_recent_actions', [])}
        )

        if not allowed:
            return {
                'success': False,
                'error': 'Blocked by core rules',
                'messages': messages,
                'task': task
            }

        for msg in messages:
            print(f"[RULES] {msg}")

        if mode == "standard":
            result = self._execute_standard(task, context)
        elif mode == "collaborative":
            result = self._execute_collaborative(task, context)
        elif mode == "incremental":
            result = self._execute_incremental(task, context)
        else:
            return {
                'success': False,
                'error': f"Unknown mode: {mode}",
                'task': task
            }

        # Trigger memory curation if threshold met
        self._maybe_curate_memory()

        return result
    
    def _execute_standard(self, task: str, context: Optional[Dict]) -> Dict[str, Any]:
        """
        Standard execution mode:
        1. Check for existing skills
        2. If no skill, plan and execute
        3. Verify results
        4. Extract skill if successful
        """
        print("[=] MODE: Standard Execution\n")
        
        # Check if we have a skill for this task
        print("[?] Checking skill graph...")
        matching_skill = self.skills.query_skill(task)
        
        if matching_skill and matching_skill.status == SkillStatus.VERIFIED:
            print(f"   [OK] Found verified skill: {matching_skill.name}")
            print(f"   Confidence: {matching_skill.confidence:.1%}, Uses: {matching_skill.usage_count}\n")
            
            # Use the skill
            result = self.skills.use_skill(matching_skill.skill_id, context or {})
            
            # Store in memory as experience
            self.memory.store(
                content=f"Used skill '{matching_skill.name}' for: {task}",
                memory_type=MemoryType.EXPERIENCE,
                metadata={'skill_id': matching_skill.skill_id, 'task': task},
                source='skill_reuse'
            )
            
            # Record outcome
            self.skills.record_skill_outcome(matching_skill.skill_id, success=True)
            
            return {
                'success': True,
                'task': task,
                'mode': 'skill_reuse',
                'skill_used': matching_skill.name,
                'result': result
            }
        else:
            print("   • No verified skill found, planning from scratch\n")
        
        # No skill - run planner
        plan_result = self.planner.run(task, context)
        
        # Verify results
        print("\n[?] Verifying execution results...")
        passed, verification_results = self.evaluator.verify(
            output=plan_result,
            verification_levels=[VerificationLevel.LOGIC, VerificationLevel.OUTPUT],
            context={'task': task}
        )
        
        if passed:
            print("   [OK] Verification passed\n")
            
            # Consider skill extraction
            self._consider_skill_extraction(task, plan_result, verification_results)
        else:
            print("   [!]  Verification failed\n")
            for vr in verification_results:
                if not vr.passed:
                    print(f"      - {vr.message}")
        
        return {
            'success': plan_result.get('success', False),
            'task': task,
            'mode': 'standard',
            'plan_result': plan_result,
            'verification': {
                'passed': passed,
                'results': [vr.to_dict() for vr in verification_results]
            }
        }
    
    def _execute_collaborative(self, task: str, context: Optional[Dict]) -> Dict[str, Any]:
        """
        Collaborative multi-agent execution mode:
        1. Planner agent creates plan
        2. Executor agent implements
        3. Critic agent reviews
        4. Revision cycles if needed
        """
        print("[=] MODE: Collaborative Multi-Agent\n")
        
        # Use multi-agent orchestrator with tools for real execution
        workflow_result = self.orchestrator.execute_workflow(
            workflow_name="collaborative_task",
            initial_task=task,
            required_roles=[AgentRole.PLANNER, AgentRole.EXECUTOR, AgentRole.CRITIC],
            tools=self.tools  # Pass tools so executor can actually execute
        )
        
        # Store workflow in memory
        self.memory.store(
            content=f"Collaborative workflow for: {task}",
            memory_type=MemoryType.EXPERIENCE,
            metadata={
                'workflow': 'collaborative',
                'success': workflow_result.get('success'),
                'revisions': workflow_result.get('revisions', 0),
                'message_count': workflow_result.get('message_count', 0)
            },
            confidence=1.0 if workflow_result.get('success') else 0.7,
            source='multi_agent'
        )
        
        return {
            'success': workflow_result.get('success', False),
            'task': task,
            'mode': 'collaborative',
            'workflow_result': workflow_result
        }
    
    def _execute_incremental(self, task: str, context: Optional[Dict]) -> Dict[str, Any]:
        """
        Incremental execution mode with step-by-step verification:
        1. Generate reasoning steps
        2. Verify each step
        3. Halt if verification fails
        4. Provide detailed trace
        """
        print("[=] MODE: Incremental Execution with Verification\n")
        
        # Generate reasoning steps
        print("[*]  Generating reasoning steps...\n")
        prompt = f"""
Break down the following task into clear reasoning steps:

Task: {task}

Provide 5-7 numbered steps showing your reasoning process.
"""
        
        reasoning_text = self.llm.generate(prompt, temperature=0.5, max_tokens=1024)
        
        # Parse steps
        import re
        steps = []
        for line in reasoning_text.split('\n'):
            if re.match(r'^\d+[.)]\s', line.strip()):
                steps.append(line.strip())
        
        if not steps:
            steps = reasoning_text.split('\n')
        
        # Incremental verification
        verification_results = self.evaluator.incremental_verify(steps, task)
        
        # Check if all steps passed
        all_passed = all(passed for _, passed, _ in verification_results)
        
        # Store in memory
        self.memory.store(
            content=f"Incremental reasoning for: {task}\n" + "\n".join(steps),
            memory_type=MemoryType.EXPERIENCE,
            metadata={
                'mode': 'incremental',
                'num_steps': len(steps),
                'all_passed': all_passed
            },
            confidence=1.0 if all_passed else 0.6,
            source='incremental_execution'
        )
        
        return {
            'success': all_passed,
            'task': task,
            'mode': 'incremental',
            'reasoning_steps': steps,
            'verification_results': verification_results
        }
    
    def _consider_skill_extraction(
        self,
        task: str,
        result: Dict[str, Any],
        verification_results: list
    ):
        """
        Determine if successful execution should become a skill
        """
        # Only extract if successful and verified
        if not result.get('success'):
            return
        
        print("[!] Considering skill extraction...")
        
        # Extract solution pattern from plan steps
        steps = result.get('steps', [])
        if not steps:
            return
        
        solution_pattern = "\n".join([
            f"{s['step_id']}. {s['description']}" 
            for s in result.get('steps', [])
            if s.get('status') == 'completed'
        ])
        
        # Determine category (heuristic)
        task_lower = task.lower()
        if any(word in task_lower for word in ['parse', 'process', 'transform', 'data']):
            category = SkillCategory.DATA_PROCESSING
        elif any(word in task_lower for word in ['code', 'write', 'implement', 'function']):
            category = SkillCategory.CODE_GENERATION
        elif any(word in task_lower for word in ['plan', 'break', 'decompose']):
            category = SkillCategory.PLANNING
        else:
            category = SkillCategory.REASONING
        
        # Extract skill
        skill_id = self.skills.extract_skill(
            task_description=task,
            solution=solution_pattern,
            execution_result=result,
            category=category
        )
        
        print(f"   [OK] Skill extracted (ID: {skill_id[:8]}...)")
        print(f"   Status: CANDIDATE - needs verification before reuse\n")
    
    def curate_memory(self, force: bool = False) -> Dict[str, Any]:
        """
        Curate memory using SSM model (Jamba) for efficient long-context processing.

        This compacts old experiences into summaries, preserving information
        while reducing memory footprint. Uses MEMORY_DIGEST task type to route
        to SSM/Mamba model which has O(n) linear scaling vs O(n²) transformers.

        Args:
            force: If True, run curation regardless of threshold

        Returns:
            Dict with curation statistics
        """
        # Check if curation is needed
        current_count = len(self.memory.get_recent_by_type(MemoryType.EXPERIENCE, limit=10000))

        if not force and (current_count - self._last_curation_count) < self._memory_curation_threshold:
            return {'skipped': True, 'reason': 'threshold_not_met', 'current_count': current_count}

        print("\n[~] MEMORY CURATION (using SSM model)")
        print(f"   Experiences: {current_count}")

        # Create a curation-specific LLM wrapper that forces MEMORY_DIGEST routing
        class CurationLLM:
            def __init__(self, llm_interface):
                self.llm = llm_interface

            def generate(self, prompt, max_tokens=100):
                # Force routing to SSM model via task type
                return self.llm.generate(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=0.3,
                    task_type=TaskType.MEMORY_DIGEST
                )

        curation_llm = CurationLLM(self.llm)

        # Run compaction with SSM model
        result = self.memory.compact_old_memories(
            age_threshold_hours=24.0,  # Compact memories older than 24 hours
            cluster_size=3,
            llm_interface=curation_llm
        )

        self._last_curation_count = current_count

        print(f"   Clusters created: {result.get('clusters_created', 0)}")
        print(f"   Memories compacted: {result.get('memories_compacted', 0)}")

        return {
            'skipped': False,
            'clusters_created': result.get('clusters_created', 0),
            'memories_compacted': result.get('memories_compacted', 0),
            'previous_count': current_count,
        }

    def _maybe_curate_memory(self):
        """Check and run memory curation if threshold is met"""
        try:
            result = self.curate_memory(force=False)
            if not result.get('skipped') and result.get('memories_compacted', 0) > 0:
                print(f"   [OK] Memory curation complete: {result.get('memories_compacted', 0)} memories compacted")
        except Exception as e:
            # Log with more context but don't fail the main task
            import traceback
            print(f"   [!] Memory curation failed: {e}")
            print(f"       (This is non-critical - main task continues)")
            # Store the error for later analysis if needed
            if not hasattr(self, '_curation_errors'):
                self._curation_errors = []
            self._curation_errors.append({
                'timestamp': time.time(),
                'error': str(e),
                'traceback': traceback.format_exc()
            })

    def analyze_performance(self) -> Dict[str, Any]:
        """
        Comprehensive system performance analysis
        """
        print("\n" + "="*70)
        print("[#] RAEC SYSTEM PERFORMANCE ANALYSIS")
        print("="*70 + "\n")
        
        # Memory stats
        print("💾 Memory System:")
        memory_stats = {
            'facts': len(self.memory.get_recent_by_type(MemoryType.FACT, limit=1000)),
            'experiences': len(self.memory.get_recent_by_type(MemoryType.EXPERIENCE, limit=1000)),
            'beliefs': len(self.memory.get_recent_by_type(MemoryType.BELIEF, limit=1000)),
            'summaries': len(self.memory.get_recent_by_type(MemoryType.SUMMARY, limit=1000))
        }
        
        for mem_type, count in memory_stats.items():
            print(f"   {mem_type.capitalize()}: {count}")
        
        # Skill stats
        print("\n[o] Skill Graph:")
        skill_stats = self.skills.get_stats()
        print(f"   Total skills: {skill_stats['total_skills']}")
        print(f"   Verified: {skill_stats['verified_count']}")
        print(f"   Avg confidence: {skill_stats['avg_confidence']:.1%}")
        print(f"   Total usage: {skill_stats['total_usage']}")
        
        if skill_stats['by_status']:
            print("   By status:")
            for status, count in skill_stats['by_status'].items():
                print(f"     {status}: {count}")
        
        # Tool stats
        print("\n[T] Tool System:")
        tool_stats = self.tools.get_tool_stats()
        print(f"   Total tools: {tool_stats['total_tools']}")
        print(f"   Verified: {tool_stats['verified']}")
        print(f"   Active: {tool_stats['active']}")
        print(f"   Total executions: {tool_stats['total_executions']}")
        print(f"   Avg success rate: {tool_stats['avg_success_rate']:.1%}")
        
        # Agent stats
        print("\n[R] Multi-Agent System:")
        agent_stats = self.orchestrator.get_agent_stats()
        print(f"   Total agents: {agent_stats['total_agents']}")
        print(f"   Messages processed: {agent_stats['total_messages_processed']}")
        print(f"   Tasks completed: {agent_stats['total_tasks_completed']}")
        
        if agent_stats['by_role']:
            print("   By role:")
            for role, count in agent_stats['by_role'].items():
                print(f"     {role}: {count}")
        
        # Verification stats
        print("\n[OK] Verification System:")
        verification_stats = self.evaluator.get_verification_stats()
        print(f"   Total verifications: {verification_stats['total_verifications']}")
        if verification_stats['total_verifications'] > 0:
            print(f"   Pass rate: {verification_stats['pass_rate']:.1%}")
        
        # Bottleneck detection
        print("\n[!]  Bottleneck Analysis:")
        bottlenecks = self.tools.detect_bottlenecks(threshold_ms=500)

        # Swarm stats (if enabled)
        swarm_stats = None
        if self.llm.swarm:
            print("\n[S] Model Swarm:")
            swarm_stats = self.llm.get_swarm_stats()
            if swarm_stats:
                print(f"   Available models: {len(swarm_stats.get('available_models', []))}")
                if swarm_stats.get('calls_by_model'):
                    print("   Calls by model:")
                    for model, count in swarm_stats['calls_by_model'].items():
                        avg_lat = swarm_stats.get('avg_latency_by_model', {}).get(model, 0)
                        print(f"     {model}: {count} calls ({avg_lat:.2f}s avg)")

        print("\n" + "="*70 + "\n")

        return {
            'memory': memory_stats,
            'skills': skill_stats,
            'tools': tool_stats,
            'agents': agent_stats,
            'verification': verification_stats,
            'bottlenecks': bottlenecks,
            'swarm': swarm_stats
        }
    
    def close(self):
        """Clean shutdown"""
        self.memory.close()
        print("\n[OK] Raec system shutdown complete")


def main():
    """Example usage"""
    try:
        # Initialize Raec
        raec = Raec()
        
        # Example task
        task = "Create a function to calculate fibonacci numbers"
        
        # Try standard mode
        print("\n" + "="*70)
        print("EXAMPLE: Standard Mode")
        print("="*70)
        result = raec.execute_task(task, mode="standard")
        print(f"\nResult: {result.get('success')}")
        
        # Analyze performance
        raec.analyze_performance()
        
        # Clean shutdown
        raec.close()
        
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
