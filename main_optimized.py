"""
Raec - Autonomous Reasoning and Execution Core
OPTIMIZED VERSION with tool integration and enhanced monitoring

Improvements:
- Tool-enabled planner integrated
- Network access support (configurable)
- Enhanced error handling
- Performance tracking
- Resource management
"""
import sys
import os
import yaml
import time
import threading
from typing import Dict, Any, Optional, List
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

# Path Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Core imports
from raec_core.llm_interface import LLMInterface
from planner.planner_tools import ToolEnabledPlanner
from memory.memory_db import HierarchicalMemoryDB, MemoryType
from skills.skill_graph import SkillGraph, SkillCategory, SkillStatus
from tools.executor import ToolExecutor, ToolType
from tools.tool_interface import ToolInterface
from agents.orchestrator import MultiAgentOrchestrator, AgentRole
from evaluators.logic_checker import LogicChecker, VerificationLevel


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics"""
    tasks_executed: int = 0
    tasks_successful: int = 0
    tasks_failed: int = 0
    total_execution_time: float = 0.0
    avg_execution_time: float = 0.0
    skills_extracted: int = 0
    skills_reused: int = 0
    tools_used: int = 0
    memory_queries: int = 0
    agent_messages: int = 0
    verifications_passed: int = 0
    verifications_failed: int = 0
    recent_tasks: deque = field(default_factory=lambda: deque(maxlen=10))
    
    @property
    def success_rate(self) -> float:
        if self.tasks_executed == 0:
            return 0.0
        return self.tasks_successful / self.tasks_executed


class Raec:
    """
    OPTIMIZED Raec system with:
    - Tool-enabled planning
    - Real-time performance tracking
    - Enhanced error handling
    - Network access (configurable)
    - Resource management
    """
    
    def __init__(self, config_path: str = "config.yaml", enable_network: bool = False):
        """
        Initialize Raec system
        
        Args:
            config_path: Path to configuration file
            enable_network: Enable network access for web tools (default: False for safety)
        """
        print("\n" + "="*70)
        print("🧠 INITIALIZING RAEC SYSTEM (OPTIMIZED)")
        print("="*70 + "\n")
        
        # Performance tracking
        self.metrics = PerformanceMetrics()
        self._start_time = time.time()
        self.enable_network = enable_network
        
        # Load configuration
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_config_path = os.path.join(base_dir, config_path)
        
        try:
            with open(full_config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            print(f"❌ Configuration error: {e}")
            raise
        
        # Initialize LLM
        print("⚙️  Initializing LLM interface...")
        try:
            self.llm = LLMInterface(model=self.config['model']['name'])
            print(f"   ✓ Connected to model: {self.config['model']['name']}")
        except Exception as e:
            print(f"   ❌ LLM initialization failed: {e}")
            raise
        
        # Initialize subsystems with error handling
        db_path = os.path.join(base_dir, self.config['memory']['db_path'])
        skill_path = os.path.join(base_dir, self.config.get('skills', {}).get('storage_path', 'skills/skill_db.json'))
        
        print("\n⚙️  Initializing subsystems...")
        
        # Memory
        try:
            self.memory = HierarchicalMemoryDB(db_path=db_path)
            print("   ✓ Hierarchical Memory initialized")
        except Exception as e:
            print(f"   ⚠️  Memory initialization warning: {e}")
            self.memory = None
        
        # Skills
        try:
            self.skills = SkillGraph(storage_path=skill_path)
            print("   ✓ Skill Graph loaded")
        except Exception as e:
            print(f"   ⚠️  Skill Graph initialization warning: {e}")
            self.skills = None
        
        # Tool System - both executor and interface
        try:
            self.tool_executor = ToolExecutor(
                python_timeout=self.config['tools'].get('python_timeout', 60)
            )
            self.tools = ToolInterface()
            print("   ✓ Tool System ready (30+ tools available)")
        except Exception as e:
            print(f"   ⚠️  Tool system warning: {e}")
            self.tool_executor = None
            self.tools = None
        
        # Multi-Agent Orchestrator
        try:
            self.orchestrator = MultiAgentOrchestrator(self.llm)
            self._initialize_agents()
            print("   ✓ Multi-Agent Orchestrator configured")
        except Exception as e:
            print(f"   ⚠️  Orchestrator warning: {e}")
            self.orchestrator = None
        
        # Verification
        try:
            self.evaluator = LogicChecker(self.llm)
            print("   ✓ Logic Checker enabled")
        except Exception as e:
            print(f"   ⚠️  Evaluator warning: {e}")
            self.evaluator = None
        
        # Tool-Enabled Planner (NEW!)
        try:
            if self.memory and self.skills and self.tools:
                self.planner = ToolEnabledPlanner(
                    self.llm,
                    self.memory,
                    self.skills,
                    tools=self.tools,
                    max_steps=self.config['planner'].get('max_steps', 10)
                )
                print("   ✓ Tool-Enabled Planner initialized")
            else:
                print("   ⚠️  Planner requires memory, skills, and tools - using basic planner")
                from planner.planner import Planner
                self.planner = Planner(self.llm, self.memory, self.skills)
        except Exception as e:
            print(f"   ⚠️  Planner warning: {e}")
            self.planner = None
        
        # System directive
        self.logic_directive = "You are a technical reasoning engine. Provide objective analysis."
        
        # Network status
        if enable_network:
            print("\n   ⚠️  Network access ENABLED (web tools will work)")
        else:
            print("\n   ℹ️  Network access DISABLED (enable with enable_network=True)")
        
        print("\n" + "="*70)
        print("✅ RAEC SYSTEM ONLINE")
        print(f"   Uptime: {time.time() - self._start_time:.2f}s")
        print("="*70 + "\n")
    
    def _initialize_agents(self):
        """Set up default agents for multi-agent mode"""
        if not self.orchestrator:
            return
        
        try:
            self.orchestrator.create_agent(
                AgentRole.PLANNER,
                capabilities=["task_decomposition", "planning"],
                description="Plans and breaks down complex tasks"
            )
            
            self.orchestrator.create_agent(
                AgentRole.EXECUTOR,
                capabilities=["execution", "implementation", "tool_usage"],
                description="Executes plans and implements solutions using tools"
            )
            
            self.orchestrator.create_agent(
                AgentRole.CRITIC,
                capabilities=["review", "quality_assurance", "verification"],
                description="Reviews and validates work quality"
            )
        except Exception as e:
            print(f"   ⚠️  Agent initialization warning: {e}")
    
    def process_input(self, task: str, mode: str = "standard") -> str:
        """
        Main entry point for processing user input
        
        Args:
            task: User's task
            mode: Execution mode (standard, collaborative, incremental)
            
        Returns:
            Result string
        """
        start_time = time.time()
        
        try:
            # Execute with specified mode
            full_task = f"{self.logic_directive}\n\nTask: {task}"
            result = self.execute_task(full_task, mode=mode)
            
            # Update metrics
            exec_time = time.time() - start_time
            self.metrics.tasks_executed += 1
            self.metrics.total_execution_time += exec_time
            self.metrics.avg_execution_time = self.metrics.total_execution_time / self.metrics.tasks_executed
            
            if result.get('success'):
                self.metrics.tasks_successful += 1
            else:
                self.metrics.tasks_failed += 1
            
            self.metrics.recent_tasks.append({
                'task': task[:50],
                'mode': mode,
                'success': result.get('success'),
                'time': exec_time,
                'timestamp': datetime.now()
            })
            
            if not result.get('success'):
                return f"Error: {result.get('error', 'Execution failed')}"
            
            # Synthesize output
            synthesis_prompt = f"""
Task: {task}
Execution Data: {str(result.get('steps', []))[:500]}

Based on the execution data, provide a direct, concise response.
"""
            
            return self.llm.generate(synthesis_prompt, temperature=0.5, max_tokens=512)
            
        except Exception as e:
            self.metrics.tasks_failed += 1
            return f"Error during execution: {e}"
    
    def execute_task(
        self,
        task: str,
        mode: str = "standard",
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Execute a task with specified mode
        
        Modes:
        - standard: Normal planning and execution with memory/skills/tools
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
        print(f"🎯 EXECUTING TASK (Mode: {mode})")
        print(f"{'='*70}")
        print(f"Task: {task}\n")
        
        try:
            if mode == "standard":
                return self._execute_standard(task, context)
            elif mode == "collaborative":
                return self._execute_collaborative(task, context)
            elif mode == "incremental":
                return self._execute_incremental(task, context)
            else:
                return {
                    'success': False,
                    'error': f"Unknown mode: {mode}",
                    'task': task
                }
        except Exception as e:
            print(f"❌ Execution error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'task': task
            }
    
    def _execute_standard(self, task: str, context: Optional[Dict]) -> Dict[str, Any]:
        """
        Standard execution mode with tools:
        1. Check for existing skills
        2. If no skill, plan with tool assignment
        3. Execute using actual tools
        4. Verify results
        5. Extract skill if successful
        """
        print("📋 MODE: Standard Execution with Tools\n")
        
        # Check if we have a skill for this task
        if self.skills:
            print("🔍 Checking skill graph...")
            matching_skill = self.skills.query_skill(task)
            
            if matching_skill and matching_skill.status == SkillStatus.VERIFIED:
                print(f"   ✓ Found verified skill: {matching_skill.name}")
                print(f"   Confidence: {matching_skill.confidence:.1%}, Uses: {matching_skill.usage_count}\n")
                
                # Use the skill
                result = self.skills.use_skill(matching_skill.skill_id, context or {})
                
                # Update metrics
                self.metrics.skills_reused += 1
                
                # Store in memory
                if self.memory:
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
        
        # No skill - run tool-enabled planner
        if not self.planner:
            return {'success': False, 'error': 'Planner not initialized'}
        
        plan_result = self.planner.run(task, context)
        
        # Update tool usage metrics
        if 'steps' in plan_result:
            for step in plan_result['steps']:
                if step.get('tool'):
                    self.metrics.tools_used += 1
        
        # Verify results
        if self.evaluator:
            print("\n🔍 Verifying execution results...")
            passed, verification_results = self.evaluator.verify(
                output=plan_result,
                verification_levels=[VerificationLevel.LOGIC, VerificationLevel.OUTPUT],
                context={'task': task}
            )
            
            # Update metrics
            if passed:
                self.metrics.verifications_passed += 1
                print("   ✅ Verification passed\n")
                self._consider_skill_extraction(task, plan_result, verification_results)
            else:
                self.metrics.verifications_failed += 1
                print("   ⚠️  Verification failed\n")
                for vr in verification_results:
                    if not vr.passed:
                        print(f"      - {vr.message}")
        else:
            passed = True
            verification_results = []
        
        return {
            'success': plan_result.get('success', False),
            'task': task,
            'mode': 'standard',
            'plan_result': plan_result,
            'verification': {
                'passed': passed,
                'results': [vr.to_dict() for vr in verification_results] if verification_results else []
            }
        }
    
    def _execute_collaborative(self, task: str, context: Optional[Dict]) -> Dict[str, Any]:
        """
        Collaborative multi-agent execution mode
        """
        print("📋 MODE: Collaborative Multi-Agent\n")
        
        if not self.orchestrator:
            return {'success': False, 'error': 'Orchestrator not initialized'}
        
        # Use multi-agent orchestrator
        workflow_result = self.orchestrator.execute_workflow(
            workflow_name="collaborative_task",
            initial_task=task,
            required_roles=[AgentRole.PLANNER, AgentRole.EXECUTOR, AgentRole.CRITIC]
        )
        
        # Update metrics
        self.metrics.agent_messages += workflow_result.get('message_count', 0)
        
        # Store workflow in memory
        if self.memory:
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
        Incremental execution mode with step-by-step verification
        """
        print("📋 MODE: Incremental Execution with Verification\n")
        
        # Generate reasoning steps
        print("⚙️  Generating reasoning steps...\n")
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
        if self.evaluator:
            verification_results = self.evaluator.incremental_verify(steps, task)
            all_passed = all(passed for _, passed, _ in verification_results)
            
            # Update metrics
            if all_passed:
                self.metrics.verifications_passed += 1
            else:
                self.metrics.verifications_failed += 1
        else:
            verification_results = []
            all_passed = True
        
        # Store in memory
        if self.memory:
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
        """Extract successful execution as a skill"""
        if not self.skills or not result.get('success'):
            return
        
        print("💡 Considering skill extraction...")
        
        # Extract solution pattern from plan steps
        steps = result.get('steps', [])
        if not steps:
            return
        
        solution_pattern = "\n".join([
            f"{s.get('step_id', i)}. {s.get('description', str(s))}" 
            for i, s in enumerate(steps, 1)
            if isinstance(s, dict) and s.get('status') in ['completed', 'COMPLETED']
        ])
        
        if not solution_pattern:
            return
        
        # Determine category
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
        try:
            skill_id = self.skills.extract_skill(
                task_description=task,
                solution=solution_pattern,
                execution_result=result,
                category=category
            )
            
            self.metrics.skills_extracted += 1
            print(f"   ✓ Skill extracted (ID: {skill_id[:8]}...)")
            print(f"   Status: CANDIDATE - needs verification before reuse\n")
        except Exception as e:
            print(f"   ⚠️  Skill extraction failed: {e}\n")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get real-time performance metrics"""
        uptime = time.time() - self._start_time
        
        return {
            'uptime_seconds': uptime,
            'tasks': {
                'total': self.metrics.tasks_executed,
                'successful': self.metrics.tasks_successful,
                'failed': self.metrics.tasks_failed,
                'success_rate': self.metrics.success_rate
            },
            'timing': {
                'total_execution_time': self.metrics.total_execution_time,
                'avg_execution_time': self.metrics.avg_execution_time
            },
            'skills': {
                'extracted': self.metrics.skills_extracted,
                'reused': self.metrics.skills_reused
            },
            'tools': {
                'used': self.metrics.tools_used
            },
            'verification': {
                'passed': self.metrics.verifications_passed,
                'failed': self.metrics.verifications_failed
            },
            'agents': {
                'messages': self.metrics.agent_messages
            },
            'recent_tasks': list(self.metrics.recent_tasks)
        }
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Comprehensive system performance analysis"""
        print("\n" + "="*70)
        print("📊 RAEC SYSTEM PERFORMANCE ANALYSIS")
        print("="*70 + "\n")
        
        # Real-time metrics
        perf = self.get_performance_metrics()
        print("⚡ Real-Time Performance:")
        print(f"   Uptime: {perf['uptime_seconds']:.2f}s")
        print(f"   Tasks: {perf['tasks']['total']} ({perf['tasks']['success_rate']:.1%} success)")
        print(f"   Avg execution time: {perf['timing']['avg_execution_time']:.2f}s")
        print(f"   Skills extracted: {perf['skills']['extracted']}")
        print(f"   Skills reused: {perf['skills']['reused']}")
        print(f"   Tools used: {perf['tools']['used']}")
        
        # Memory stats
        if self.memory:
            print("\n💾 Memory System:")
            memory_stats = {
                'facts': len(self.memory.get_recent_by_type(MemoryType.FACT, limit=1000)),
                'experiences': len(self.memory.get_recent_by_type(MemoryType.EXPERIENCE, limit=1000)),
                'beliefs': len(self.memory.get_recent_by_type(MemoryType.BELIEF, limit=1000)),
                'summaries': len(self.memory.get_recent_by_type(MemoryType.SUMMARY, limit=1000))
            }
            
            for mem_type, count in memory_stats.items():
                print(f"   {mem_type.capitalize()}: {count}")
        
        # Skill stats
        if self.skills:
            print("\n🎯 Skill Graph:")
            skill_stats = self.skills.get_stats()
            print(f"   Total skills: {skill_stats['total_skills']}")
            print(f"   Verified: {skill_stats['verified_count']}")
            print(f"   Avg confidence: {skill_stats['avg_confidence']:.1%}")
            print(f"   Total usage: {skill_stats['total_usage']}")
            
            if skill_stats.get('by_status'):
                print("   By status:")
                for status, count in skill_stats['by_status'].items():
                    print(f"     {status}: {count}")
        
        # Tool stats
        if self.tool_executor:
            print("\n🔧 Tool System:")
            tool_stats = self.tool_executor.get_tool_stats()
            print(f"   Total tools: {tool_stats['total_tools']}")
            print(f"   Verified: {tool_stats['verified']}")
            print(f"   Active: {tool_stats['active']}")
            print(f"   Total executions: {tool_stats['total_executions']}")
            if tool_stats['total_executions'] > 0:
                print(f"   Avg success rate: {tool_stats['avg_success_rate']:.1%}")
        
        # Agent stats
        if self.orchestrator:
            print("\n🤖 Multi-Agent System:")
            agent_stats = self.orchestrator.get_agent_stats()
            print(f"   Total agents: {agent_stats['total_agents']}")
            print(f"   Messages processed: {agent_stats['total_messages_processed']}")
            print(f"   Tasks completed: {agent_stats['total_tasks_completed']}")
            
            if agent_stats.get('by_role'):
                print("   By role:")
                for role, count in agent_stats['by_role'].items():
                    print(f"     {role}: {count}")
        
        # Verification stats
        if self.evaluator:
            print("\n✅ Verification System:")
            verification_stats = self.evaluator.get_verification_stats()
            print(f"   Total verifications: {verification_stats['total_verifications']}")
            if verification_stats['total_verifications'] > 0:
                print(f"   Pass rate: {verification_stats['pass_rate']:.1%}")
        
        # Bottleneck detection
        if self.tool_executor:
            print("\n⚠️  Bottleneck Analysis:")
            bottlenecks = self.tool_executor.detect_bottlenecks(threshold_ms=500)
            if not bottlenecks:
                print("   ✓ No bottlenecks detected")
        
        print("\n" + "="*70 + "\n")
        
        return {
            'realtime': perf,
            'memory': memory_stats if self.memory else {},
            'skills': skill_stats if self.skills else {},
            'tools': tool_stats if self.tool_executor else {},
            'agents': agent_stats if self.orchestrator else {},
            'verification': verification_stats if self.evaluator else {},
            'bottlenecks': bottlenecks if self.tool_executor else []
        }
    
    def close(self):
        """Clean shutdown with resource cleanup"""
        print("\n🔄 Shutting down Raec system...")
        
        try:
            if self.memory:
                self.memory.close()
                print("   ✓ Memory closed")
        except Exception as e:
            print(f"   ⚠️  Memory close warning: {e}")
        
        uptime = time.time() - self._start_time
        print(f"\n✓ Raec system shutdown complete")
        print(f"   Total uptime: {uptime:.2f}s")
        print(f"   Tasks executed: {self.metrics.tasks_executed}")
        print(f"   Success rate: {self.metrics.success_rate:.1%}")


def main():
    """Example usage"""
    try:
        # Initialize Raec with network disabled (safer)
        raec = Raec(enable_network=False)
        
        # Example task
        task = "Count the words in this sentence"
        
        # Try standard mode with tools
        print("\n" + "="*70)
        print("EXAMPLE: Standard Mode with Tools")
        print("="*70)
        result = raec.execute_task(task, mode="standard")
        print(f"\nResult: {result.get('success')}")
        
        # Analyze performance
        raec.analyze_performance()
        
        # Clean shutdown
        raec.close()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
