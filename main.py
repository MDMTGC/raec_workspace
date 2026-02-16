"""
Raec - Autonomous Reasoning and Execution Core
Complete integration with all upgraded components

Fixed Integration:
- All imports corrected
- Method calls aligned
- Proper error handling
- Three execution modes working
- Identity and conversation persistence
- Intent classification for routing
"""
import sys
import os
import yaml
import time
from typing import Dict, Any, List, Optional

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

# Identity and conversation systems
from identity import SelfModel
from conversation import ConversationManager, IntentClassifier, Intent

# Web access (transparent, logged)
from web import WebFetcher, WebSearch, ActivityLog

# Curiosity engine (autonomous exploration)
from curiosity import CuriosityEngine, QuestionQueue, IdleLoop

# Agency systems
from goals import GoalManager, GoalType, GoalStatus
from preferences import PreferenceManager, PreferenceType
from evaluation import SelfEvaluator, EvaluationType
from toolsmith import ToolForge
from proactive import Notifier, NotificationType
from context import ContextManager
from uncertainty import ConfidenceTracker


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

        # Identity system (persistent personality)
        identity_path = os.path.join(base_dir, "identity/identity.json")
        self.identity = SelfModel(identity_path=identity_path)
        print(f"   [OK] Identity loaded: {self.identity.identity.name}")

        # Conversation manager (session continuity)
        conv_path = os.path.join(base_dir, "conversation/conversation_state.json")
        self.conversation = ConversationManager(state_path=conv_path)
        print(f"   [OK] Conversation manager active (session: {self.conversation.current_session.session_id})")

        # Intent classifier (routing)
        self.intent_classifier = IntentClassifier()
        print("   [OK] Intent classifier ready")

        # Web access (transparent, all activity logged)
        self.web_activity = ActivityLog()
        self.web_fetcher = WebFetcher(activity_log=self.web_activity)
        self.web_searcher = WebSearch(activity_log=self.web_activity)
        print("   [OK] Web access enabled (transparent mode)")

        # Curiosity engine (autonomous exploration)
        self.question_queue = QuestionQueue()
        self.curiosity = CuriosityEngine(
            question_queue=self.question_queue,
            llm_interface=self.llm,
            web_search=self.search_web,
            web_fetch=self.web_fetch,
            memory_store=self._store_curiosity_finding
        )
        self.idle_loop = IdleLoop(
            curiosity_engine=self.curiosity,
            idle_threshold=60.0,           # Start investigating after 1 min idle
            investigation_interval=120.0,  # 2 min between investigations
            max_investigations_per_session=5,
            on_state_change=self._on_curiosity_state_change,
            on_investigation_complete=self._on_investigation_complete
        )
        print(f"   [OK] Curiosity engine ready ({self.curiosity.get_pending_count()} pending questions)")

        # Agency systems
        self.goals = GoalManager()
        print(f"   [OK] Goals manager ({self.goals.get_stats()['active']} active goals)")

        self.preferences = PreferenceManager()
        print(f"   [OK] Preferences ({self.preferences.get_stats()['total_preferences']} learned)")

        self.self_eval = SelfEvaluator()
        print(f"   [OK] Self-evaluator active")

        self.toolsmith = ToolForge(llm_interface=self.llm)
        print(f"   [OK] Toolsmith ready ({self.toolsmith.get_stats()['deployed']} deployed tools)")

        self.notifier = Notifier()
        print(f"   [OK] Proactive notifier ({self.notifier.get_stats()['pending']} pending)")

        self.context = ContextManager()
        print(f"   [OK] Context awareness active")

        self.confidence = ConfidenceTracker()
        print(f"   [OK] Uncertainty quantification ready")

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
        self._curation_errors = []  # Track non-critical errors for later review

        # Start the curiosity loop (autonomous exploration)
        self.idle_loop.start()

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
    
    def process_input(self, user_input: str, mode: str = "auto") -> str:
        """
        Main entry point for processing user input.

        Routes input based on intent classification:
        - CHAT: Conversational response using personality
        - QUERY: Information retrieval response
        - TASK: Full planning and execution
        - META: System commands about RAEC itself

        Args:
            user_input: User's message
            mode: Execution mode (auto, standard, collaborative, incremental)

        Returns:
            Result string
        """
        # Update context awareness
        ctx = self.context.update(user_input)

        # Track conversation
        self.conversation.add_user_message(user_input)
        self.identity.record_interaction()

        # Learn preferences from explicit statements
        if any(phrase in user_input.lower() for phrase in ['i prefer', 'i like', 'always ', 'never ', "don't "]):
            pref = self.preferences.learn_from_explicit(user_input, self.llm)
            if pref:
                print(f"   [+] Learned preference: {pref.name}")

        # Classify intent
        classification = self.intent_classifier.classify(user_input)
        intent = classification.intent

        print(f"\n[?] Intent: {intent.value} (confidence: {classification.confidence:.0%})")
        if ctx.user_state.urgency.value > 2:
            print(f"   [!] Urgency detected: {ctx.user_state.urgency.name}")

        # Route based on intent
        if intent == Intent.CHAT:
            response = self._handle_chat(user_input)
        elif intent == Intent.QUERY:
            response = self._handle_query(user_input)
        elif intent == Intent.META:
            response = self._handle_meta(user_input)
        else:  # Intent.TASK
            response = self._handle_task(user_input, mode if mode != "auto" else "standard")

        # Assess confidence in response
        confidence = self.confidence.assess_confidence(response, task_type=intent.value)
        if self.confidence.should_express_uncertainty(confidence):
            print(f"   [~] Low confidence: {confidence.score:.0%}")

        # Analyze response for curiosity triggers (uncertainty, knowledge gaps)
        questions_added = self.curiosity.analyze_response(
            response=response,
            user_input=user_input,
            session_id=self.conversation.current_session.session_id
        )
        if questions_added:
            print(f"   [?] Added {len(questions_added)} questions to investigate")

        # Record user activity (resets idle timer)
        self.idle_loop.record_user_activity()

        # Track response
        self.conversation.add_assistant_message(response)
        self.conversation.save()
        self.identity.save()

        return response

    def _handle_chat(self, user_input: str) -> str:
        """Handle casual conversation"""
        identity_context = self.identity.get_system_prompt_context()
        recent_messages = self.conversation.get_context_messages(limit=5)

        # Build conversation context
        conv_context = "\n".join([
            f"{m['role']}: {m['content'][:200]}"
            for m in recent_messages[:-1]  # Exclude current message
        ])

        # Get applicable preferences
        pref_context = self.preferences.build_preference_prompt("chat")

        # Get context hints
        hints = self.context.get_response_hints()
        style_hint = ""
        if hints.get("verbosity") == "minimal":
            style_hint = "Keep response very brief."
        elif hints.get("mood_aware") and hints.get("formality") == "empathetic":
            style_hint = "The user may be frustrated. Be understanding."

        prompt = f"""{identity_context}

{pref_context}

Recent conversation:
{conv_context}

User: {user_input}

Reply in first person ("I", "my"). Be conversational, concise, and direct. No narration, no third-person.
{style_hint}
"""
        return self.llm.generate(prompt, temperature=0.7, max_tokens=256)

    def _handle_query(self, user_input: str) -> str:
        """Handle information requests"""
        identity_context = self.identity.get_system_prompt_context()

        # Check memory for relevant facts
        relevant_memories = self.memory.query(user_input, k=3)
        memory_context = ""
        if relevant_memories:
            memory_context = "\nRelevant knowledge:\n" + "\n".join([
                f"- {m['content'][:150]}" for m in relevant_memories
            ])

        prompt = f"""{identity_context}
{memory_context}

Question: {user_input}

Answer in first person ("I", "my"). Be direct and concise. No narration or third-person.
"""
        response = self.llm.generate(prompt, temperature=0.5, max_tokens=512)

        # Track topic
        self.conversation.add_topic(user_input[:50])

        return response

    def _handle_meta(self, user_input: str) -> str:
        """Handle system commands about RAEC"""
        input_lower = user_input.lower()

        # Direct meta commands
        if '/help' in input_lower:
            return self._get_help_text()
        elif '/status' in input_lower:
            return self._get_status_text()
        elif '/skills' in input_lower:
            return self._get_skills_text()
        elif '/curiosity' in input_lower:
            return self._get_curiosity_text()
        elif '/learned' in input_lower or 'what did you learn' in input_lower:
            return self.what_did_you_learn()
        elif '/questions' in input_lower or 'what are you wondering' in input_lower:
            return self.get_pending_questions()
        elif '/web' in input_lower or 'web activity' in input_lower:
            return self.get_web_activity()
        elif '/goals' in input_lower:
            return self.goals.format_active_goals()
        elif '/preferences' in input_lower or '/prefs' in input_lower:
            return self.preferences.format_preferences()
        elif '/performance' in input_lower or '/eval' in input_lower:
            return self.self_eval.format_performance_report()
        elif '/tools' in input_lower or '/toolsmith' in input_lower:
            return self.toolsmith.format_tools()
        elif '/notifications' in input_lower or '/notify' in input_lower:
            return self.notifier.format_pending()
        elif '/confidence' in input_lower or '/calibration' in input_lower:
            return self.confidence.format_calibration_report()
        elif 'who are you' in input_lower or 'what are you' in input_lower:
            return self._get_identity_text()
        elif 'what can you do' in input_lower:
            return self._get_capabilities_text()
        else:
            # General meta query
            identity_context = self.identity.get_system_prompt_context()
            prompt = f"""{identity_context}

The user is asking about me: {user_input}

Answer in first person about my capabilities, identity, or status. Be direct. No third-person narration.
"""
            return self.llm.generate(prompt, temperature=0.5, max_tokens=256)

    def _handle_task(self, task: str, mode: str) -> str:
        """Handle action/execution requests"""
        # Execute with specified mode
        full_task = f"{self.logic_directive}\n\nTask: {task}"
        result = self.execute_task(full_task, mode=mode)

        if not result.get('success'):
            return f"Error: {result.get('error', 'Execution failed')}"

        # Track outcome
        self.conversation.add_outcome(f"Completed: {task[:50]}")

        # Synthesize output - confirm what was done
        steps_completed = [s for s in result.get('steps', []) if s.get('status') == 'completed']
        step_summary = "\n".join([
            f"- {s.get('description', 'Unknown step')}: {str(s.get('result', ''))[:100]}"
            for s in steps_completed
        ])

        identity_context = self.identity.get_system_prompt_context()
        synthesis_prompt = f"""{identity_context}

Task requested: {task}
Status: COMPLETED SUCCESSFULLY
Steps executed:
{step_summary}

Confirm in first person what I accomplished (1-2 sentences). Start with "Done:" or "Completed:". No narration.
"""

        return self.llm.generate(synthesis_prompt, temperature=0.5, max_tokens=256)

    def _get_help_text(self) -> str:
        """Return help text"""
        return """RAEC - Reflective Agentic Ecosystem Composer

Commands:
  /help         - Show this help
  /status       - System status
  /skills       - List learned skills
  /curiosity    - Curiosity engine status
  /learned      - What I learned autonomously
  /questions    - What I'm wondering about
  /web          - Recent web activity
  /goals        - Active goals I'm pursuing
  /preferences  - Learned preferences
  /performance  - Self-evaluation report
  /tools        - Generated tools
  /notifications- Pending notifications
  /confidence   - Calibration report

Modes:
  standard      - Normal planning and execution
  collaborative - Multi-agent workflow
  incremental   - Step-by-step with verification

Just tell me what you need - I'll figure out how to help."""

    def _get_status_text(self) -> str:
        """Return system status"""
        skill_stats = self.skills.get_stats()
        memory_count = len(self.memory.get_recent_by_type(MemoryType.EXPERIENCE, limit=1000))
        session_msgs = self.conversation.message_count()
        curiosity_stats = self.get_curiosity_stats()
        goal_stats = self.goals.get_stats()
        pref_stats = self.preferences.get_stats()
        eval_stats = self.self_eval.get_stats()

        return f"""RAEC Status:
  Session: {self.conversation.current_session.session_id}
  Messages this session: {session_msgs}
  Trust level: {self.identity.identity.trust_level}
  Total interactions: {self.identity.identity.interactions_count}

Knowledge:
  Skills: {skill_stats['total_skills']} ({skill_stats['verified_count']} verified)
  Experiences: {memory_count}
  Preferences: {pref_stats['total_preferences']} learned

Agency:
  Goals: {goal_stats['active']} active ({goal_stats['completed']} completed)
  Curiosity: {curiosity_stats['idle_loop']['state']} ({curiosity_stats['questions']['unresolved']} pending)
  Performance: {eval_stats['success_rate']:.0%} success rate ({eval_stats['trend']})"""

    def _get_skills_text(self) -> str:
        """Return skills summary"""
        skill_stats = self.skills.get_stats()
        return f"""Skills: {skill_stats['total_skills']} total
  Verified: {skill_stats['verified_count']}
  Avg confidence: {skill_stats['avg_confidence']:.0%}
  Total uses: {skill_stats['total_usage']}"""

    def _get_identity_text(self) -> str:
        """Return identity information"""
        identity = self.identity.identity
        name = identity.name
        traits = identity.traits
        values = identity.core_values

        return f"""I'm {name} - Reflective Agentic Ecosystem Composer.

Traits: {', '.join(traits)}
Core values: {', '.join(values)}

I'm designed to be a persistent, evolving AI assistant that learns from our interactions and adapts over time."""

    def _get_capabilities_text(self) -> str:
        """Return capabilities information"""
        return """I can help with:

• File operations - create, read, write, organize files
• Code execution - run Python scripts, shell commands
• Planning - break down complex tasks into steps
• Learning - remember patterns and improve over time
• Web search - find current information online
• URL fetching - read and extract content from web pages
• Data processing - parse, transform, analyze data

Just describe what you need and I'll plan the approach."""

    # ═══════════════════════════════════════════════════════════════════════════
    # WEB ACCESS - Transparent internet capabilities
    # ═══════════════════════════════════════════════════════════════════════════

    def web_fetch(self, url: str, reason: str, autonomous: bool = False) -> dict:
        """Fetch content from a URL with full transparency."""
        triggered_by = "autonomous" if autonomous else "user"
        result = self.web_fetcher.fetch(url, reason, triggered_by)

        if result.success and autonomous:
            self.memory.store(
                content=f"Fetched {url}: {result.title or 'No title'}",
                memory_type=MemoryType.EXPERIENCE,
                metadata={'url': url, 'reason': reason, 'autonomous': True},
                source='web_fetch'
            )

        return {
            'success': result.success, 'url': result.url, 'title': result.title,
            'content': result.content, 'links': result.links, 'error': result.error
        }

    def search_web(self, query: str, reason: str, autonomous: bool = False) -> dict:
        """Search the web with full transparency."""
        triggered_by = "autonomous" if autonomous else "user"
        results = self.web_searcher.search(query, reason, triggered_by)

        if results.success and autonomous:
            self.memory.store(
                content=f"Searched for '{query}': {len(results.results)} results",
                memory_type=MemoryType.EXPERIENCE,
                metadata={'query': query, 'reason': reason, 'autonomous': True},
                source='web_search'
            )

        return {
            'success': results.success, 'query': results.query,
            'results': [{'title': r.title, 'url': r.url, 'snippet': r.snippet} for r in results.results],
            'error': results.error
        }

    def get_web_activity(self, limit: int = 20) -> str:
        """Get formatted web activity log for transparency."""
        return self.web_activity.format_recent(limit)

    def get_web_stats(self) -> dict:
        """Get web activity statistics."""
        return self.web_activity.get_stats()

    # ═══════════════════════════════════════════════════════════════════════════
    # CURIOSITY - Autonomous exploration and learning
    # ═══════════════════════════════════════════════════════════════════════════

    def _store_curiosity_finding(self, content: str, context: dict):
        """Store a curiosity finding in memory"""
        self.memory.store(
            content=content,
            memory_type=MemoryType.EXPERIENCE,
            metadata=context,
            source='curiosity'
        )

    def _on_curiosity_state_change(self, state):
        """Called when curiosity state changes (for GUI integration)"""
        print(f"   [~] Curiosity: {state.value}")

    def _on_investigation_complete(self, result: dict):
        """Called when an investigation completes"""
        if result.get('success'):
            print(f"   [!] Learned: {result.get('findings', '')[:100]}...")

    def start_curiosity(self):
        """Start the background curiosity loop"""
        self.idle_loop.start()
        print("   [OK] Curiosity loop started")

    def stop_curiosity(self):
        """Stop the background curiosity loop"""
        self.idle_loop.stop()
        print("   [OK] Curiosity loop stopped")

    def pause_curiosity(self):
        """Temporarily pause curiosity"""
        self.idle_loop.pause()

    def resume_curiosity(self):
        """Resume curiosity"""
        self.idle_loop.resume()

    def get_curiosity_stats(self) -> dict:
        """Get curiosity statistics"""
        return {
            "idle_loop": self.idle_loop.get_stats(),
            "questions": self.curiosity.get_stats()
        }

    def what_did_you_learn(self) -> str:
        """Get summary of what RAEC learned autonomously"""
        return self.curiosity.format_what_i_learned()

    def get_pending_questions(self) -> str:
        """Get formatted list of pending questions"""
        return self.question_queue.format_pending()

    def add_question(self, question: str, context: str = "User suggested"):
        """Manually add a question for RAEC to investigate"""
        from curiosity.questions import QuestionType, QuestionPriority
        self.question_queue.add(
            question=question,
            question_type=QuestionType.USER_INTEREST,
            context=context,
            priority=QuestionPriority.HIGH,
            source_conversation=self.conversation.current_session.session_id
        )

    def _get_curiosity_text(self) -> str:
        """Return curiosity engine status"""
        stats = self.get_curiosity_stats()
        idle_stats = stats['idle_loop']
        q_stats = stats['questions']

        status_lines = [
            "Curiosity Engine:",
            f"  State: {idle_stats['state']}",
            f"  Pending questions: {q_stats['unresolved']}",
            f"  Resolved: {q_stats['resolved']}",
            f"  Investigations this session: {idle_stats['investigations_this_session']}/{idle_stats['max_investigations']}",
            f"  Idle threshold: {idle_stats['idle_threshold']}s",
        ]

        if q_stats.get('by_type'):
            status_lines.append("  Questions by type:")
            for qtype, count in q_stats['by_type'].items():
                status_lines.append(f"    {qtype}: {count}")

        return "\n".join(status_lines)

    # ═══════════════════════════════════════════════════════════════════════════
    # AGENCY - Goals, preferences, self-evaluation, tool creation
    # ═══════════════════════════════════════════════════════════════════════════

    def add_goal(self, name: str, description: str, goal_type: str = "user_given", priority: str = "medium"):
        """Add a goal for RAEC to pursue"""
        from goals.goal_manager import GoalPriority
        gt = GoalType[goal_type.upper()] if goal_type.upper() in GoalType.__members__ else GoalType.USER_GIVEN
        gp = GoalPriority[priority.upper()] if priority.upper() in GoalPriority.__members__ else GoalPriority.MEDIUM
        return self.goals.create_goal(name=name, description=description, goal_type=gt, priority=gp, source="user")

    def update_goal_progress(self, goal_id: int, delta: float, description: str):
        """Update progress on a goal"""
        return self.goals.update_progress(goal_id, delta, description)

    def get_active_goals(self) -> list:
        """Get active goals"""
        return self.goals.get_active_goals()

    def create_tool(self, name: str, description: str, purpose: str, inputs: dict, output: str):
        """Create a new tool for RAEC to use"""
        from toolsmith.tool_forge import ToolSpec
        spec = ToolSpec(
            name=name, description=description, purpose=purpose,
            inputs=inputs, output=output, examples=[], constraints=["Be safe", "Validate inputs"]
        )
        tool = self.toolsmith.generate_tool(spec, created_by="user")
        if self.toolsmith.test_tool(tool.id):
            self.toolsmith.deploy_tool(tool.id)
            return f"Tool '{name}' created and deployed successfully"
        return f"Tool '{name}' created but failed testing"

    def use_generated_tool(self, name: str, **kwargs):
        """Use a generated tool"""
        return self.toolsmith.use_tool(name, **kwargs)

    def record_task_outcome(self, task: str, succeeded: bool, confidence: float = 0.5):
        """Record outcome of a task for self-evaluation"""
        from evaluation.self_evaluator import OutcomeRating
        self.self_eval.record_task_outcome(task=task, confidence=confidence, succeeded=succeeded)

    def get_session_greeting(self) -> str:
        """Get greeting with pending notifications"""
        greeting = self.notifier.format_session_greeting()
        if greeting:
            return f"Welcome back. {greeting}"
        return "Welcome back."

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
        
        # Verify results (structural + semantic)
        print("\n[?] Verifying execution results...")
        passed, verification_results = self.evaluator.verify(
            output=plan_result,
            verification_levels=[
                VerificationLevel.LOGIC,
                VerificationLevel.OUTPUT,
                VerificationLevel.SEMANTIC,  # W5: task-objective check
            ],
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
        
        # W7: Pass planner so executor agent can use ToolEnabledPlanner
        workflow_result = self.orchestrator.execute_workflow(
            workflow_name="collaborative_task",
            initial_task=task,
            required_roles=[AgentRole.PLANNER, AgentRole.EXECUTOR, AgentRole.CRITIC],
            tools=self.tools,
            planner=self.planner
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
        Determine if successful execution should become a skill.
        W12: Also extract step-level skills from partial successes.
        """
        steps = result.get('steps', [])
        if not steps:
            return

        completed = [s for s in steps if s.get('status') == 'completed']

        if result.get('success') and completed:
            # Full success — extract whole-plan skill
            print("[!] Considering skill extraction...")

            solution_pattern = "\n".join([
                f"{s['step_id']}. {s['description']}"
                for s in completed
            ])

            category = self._categorize_task(task)

            skill_id = self.skills.extract_skill(
                task_description=task,
                solution=solution_pattern,
                execution_result=result,
                category=category
            )

            print(f"   [OK] Skill extracted (ID: {skill_id[:8]}...)")
            print(f"   Status: CANDIDATE - needs verification before reuse\n")

        elif not result.get('success') and len(completed) >= 2:
            # W12: Partial success — extract step-level skills from
            # completed tool-based steps (at least 2 to be meaningful)
            self._extract_step_level_skills(task, completed)

    def _categorize_task(self, task: str) -> SkillCategory:
        """Determine skill category from task description."""
        task_lower = task.lower()
        if any(word in task_lower for word in ['parse', 'process', 'transform', 'data']):
            return SkillCategory.DATA_PROCESSING
        elif any(word in task_lower for word in ['code', 'write', 'implement', 'function']):
            return SkillCategory.CODE_GENERATION
        elif any(word in task_lower for word in ['plan', 'break', 'decompose']):
            return SkillCategory.PLANNING
        elif any(word in task_lower for word in ['fetch', 'http', 'web', 'url', 'api']):
            return SkillCategory.TOOL_USAGE
        else:
            return SkillCategory.REASONING

    def _extract_step_level_skills(self, task: str, completed_steps: List[Dict]):
        """
        W12: Extract skills from individual successful steps in a partially-failed plan.
        Only extracts from steps that used tools (not LLM reasoning fallbacks).
        """
        print("[!] Extracting step-level skills from partial success...")

        extracted = 0
        for step in completed_steps:
            tool = step.get('tool')
            if not tool:
                continue  # Skip LLM-only steps

            desc = step.get('description', '')
            result_preview = str(step.get('result', ''))[:200]

            # Build a mini solution pattern from this single step
            solution = f"TOOL: {tool}\nPARAMS: {step.get('params', {})}"

            category = self._categorize_task(desc)

            try:
                self.skills.extract_skill(
                    task_description=f"Step from '{task[:80]}': {desc}",
                    solution=solution,
                    execution_result={
                        'success': True,
                        'step_result': result_preview,
                        'tool': tool,
                    },
                    category=category,
                    name=f"Step: {desc[:40]}"
                )
                extracted += 1
            except Exception:
                pass  # Don't fail other extractions

        if extracted:
            print(f"   [OK] Extracted {extracted} step-level skill(s)\n")
    
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
            # Store the error for later analysis
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
    
    def reflect(self, trigger: str = "session_end") -> Optional[str]:
        """
        Self-reflection after significant interactions.

        Analyzes recent interactions and updates identity/beliefs accordingly.

        Args:
            trigger: What triggered the reflection (session_end, error, milestone)

        Returns:
            Reflection insight if any
        """
        recent_messages = self.conversation.get_context_messages(limit=10)
        if len(recent_messages) < 3:
            return None

        # Build reflection prompt
        conversation_summary = "\n".join([
            f"{m['role']}: {m['content'][:100]}"
            for m in recent_messages
        ])

        prompt = f"""Analyze this recent interaction as RAEC:

{conversation_summary}

Consider:
1. What went well?
2. What could be improved?
3. Any patterns or preferences to remember?

Provide ONE brief insight (1-2 sentences) worth remembering.
If nothing notable, respond with "No significant insight."
"""

        insight = self.llm.generate(prompt, temperature=0.5, max_tokens=128)

        if "no significant insight" not in insight.lower():
            self.identity.add_reflection(
                trigger=trigger,
                insight=insight,
                category="interaction",
                impact="low"
            )
            return insight

        return None

    def close(self):
        """Clean shutdown with state persistence"""
        print("\n" + "="*70)
        print("[*] RAEC SHUTTING DOWN")
        print("="*70)

        # Stop curiosity loop
        if self.idle_loop.state.value != "stopped":
            self.stop_curiosity()

            # Show what was learned this session
            findings = self.idle_loop.get_session_findings()
            if findings:
                print(f"\n[~] Curiosity session: investigated {len(findings)} questions")

        # Expire old notifications
        expired = self.notifier.expire_old()
        if expired:
            print(f"   [~] Expired {expired} old notifications")

        # Record session performance
        eval_stats = self.self_eval.get_stats()
        if eval_stats['total_evaluations'] > 0:
            print(f"   [~] Session performance: {eval_stats['success_rate']:.0%} success rate")

        # Run final reflection
        insight = self.reflect(trigger="session_end")
        if insight:
            print(f"\n[~] Session reflection: {insight}")

        # End conversation session
        self.conversation.end_session(
            summary=self.conversation.get_conversation_summary(),
            outcomes=self.conversation.current_session.outcomes if self.conversation.current_session else []
        )

        # Reset context for next session
        self.context.reset_session()

        # Save all state
        self.identity.save()
        self.conversation.save()
        self.memory.close()

        print("\n[OK] Raec system shutdown complete (state saved)")


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
