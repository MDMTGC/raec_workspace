"""
Multi-Agent Orchestration System
Based on AutoGen and CrewAI patterns

Enables:
- Collaborative multi-agent workflows
- Role-based agent specialization
- Internal negotiation and coordination
- Self-correction loops
"""
import time
import json
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid


class AgentRole(Enum):
    """Specialized agent roles"""
    PLANNER = "planner"          # Task decomposition and planning
    EXECUTOR = "executor"        # Task execution
    CRITIC = "critic"            # Quality assurance and review
    RESEARCHER = "researcher"    # Information gathering
    SYNTHESIZER = "synthesizer"  # Result aggregation
    SPECIALIST = "specialist"    # Domain-specific tasks


class MessageType(Enum):
    """Types of inter-agent messages"""
    TASK = "task"
    QUESTION = "question"
    RESPONSE = "response"
    CRITIQUE = "critique"
    PROPOSAL = "proposal"
    APPROVAL = "approval"
    REJECTION = "rejection"


@dataclass
class Message:
    """Message passed between agents"""
    id: str
    sender: str
    receiver: str
    message_type: MessageType
    content: Any
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(cls, sender: str, receiver: str, msg_type: MessageType, content: Any, **metadata):
        return cls(
            id=str(uuid.uuid4())[:8],
            sender=sender,
            receiver=receiver,
            message_type=msg_type,
            content=content,
            timestamp=time.time(),
            metadata=metadata
        )


@dataclass
class AgentState:
    """Current state of an agent"""
    busy: bool = False
    current_task: Optional[str] = None
    messages_processed: int = 0
    tasks_completed: int = 0
    last_active: Optional[float] = None


class Agent:
    """
    Base agent with role-specific behavior
    """
    
    def __init__(
        self,
        agent_id: str,
        role: AgentRole,
        llm_interface,
        capabilities: List[str],
        description: str = ""
    ):
        self.agent_id = agent_id
        self.role = role
        self.llm = llm_interface
        self.capabilities = capabilities
        self.description = description
        self.state = AgentState()
        
        # Message queue
        self.inbox: List[Message] = []
        self.sent_messages: List[Message] = []
        
        # Conversation history for context
        self.conversation_history: List[Dict] = []
    
    def receive_message(self, message: Message):
        """Receive a message from another agent"""
        self.inbox.append(message)
    
    def process_message(self, message: Message, context: Dict[str, Any]) -> Optional[Message]:
        """
        Process a message and optionally send a response
        
        Args:
            message: Incoming message
            context: Shared context/state
            
        Returns:
            Response message or None
        """
        self.state.last_active = time.time()
        self.state.messages_processed += 1
        
        # Add to conversation history
        self.conversation_history.append({
            'role': 'user',
            'content': f"[{message.sender}] {message.content}"
        })
        
        # Role-specific processing
        if self.role == AgentRole.PLANNER:
            return self._process_as_planner(message, context)
        elif self.role == AgentRole.EXECUTOR:
            return self._process_as_executor(message, context)
        elif self.role == AgentRole.CRITIC:
            return self._process_as_critic(message, context)
        elif self.role == AgentRole.RESEARCHER:
            return self._process_as_researcher(message, context)
        elif self.role == AgentRole.SYNTHESIZER:
            return self._process_as_synthesizer(message, context)
        else:
            return self._process_generic(message, context)
    
    def _process_as_planner(self, message: Message, context: Dict) -> Optional[Message]:
        """Process message as a planner agent"""
        if message.message_type == MessageType.TASK:
            # Create a plan
            prompt = f"""
You are a planning agent. Break down the following task into clear, actionable steps:

Task: {message.content}

Context: {json.dumps(context, indent=2)}

Provide a numbered plan with clear steps.
"""
            plan = self.llm.generate(prompt, temperature=0.7)
            
            self.conversation_history.append({
                'role': 'assistant',
                'content': plan
            })
            
            return Message.create(
                sender=self.agent_id,
                receiver=message.sender,
                msg_type=MessageType.RESPONSE,
                content=plan,
                plan_created=True
            )
        
        return None
    
    def _process_as_executor(self, message: Message, context: Dict) -> Optional[Message]:
        """Process message as an executor agent"""
        if message.message_type == MessageType.TASK:
            # Execute the task
            prompt = f"""
You are an execution agent. Complete the following task:

Task: {message.content}

Available tools: {self.capabilities}

Describe your execution approach and results.
"""
            result = self.llm.generate(prompt, temperature=0.5)
            
            self.state.tasks_completed += 1
            
            return Message.create(
                sender=self.agent_id,
                receiver=message.sender,
                msg_type=MessageType.RESPONSE,
                content=result,
                completed=True
            )
        
        return None
    
    def _process_as_critic(self, message: Message, context: Dict) -> Optional[Message]:
        """Process message as a critic/reviewer agent"""
        if message.message_type == MessageType.RESPONSE:
            # Critique the response
            prompt = f"""
You are a quality assurance critic. Review the following work:

Work: {message.content}

Original task: {message.metadata.get('original_task', 'Unknown')}

Provide constructive criticism and suggest improvements. Be specific about:
1. What works well
2. What could be improved
3. Any errors or issues

Format: [APPROVE] or [REVISE] followed by your critique.
"""
            critique = self.llm.generate(prompt, temperature=0.3)
            
            # Determine if approved
            approved = critique.strip().startswith('[APPROVE]')
            
            return Message.create(
                sender=self.agent_id,
                receiver=message.sender,
                msg_type=MessageType.APPROVAL if approved else MessageType.CRITIQUE,
                content=critique,
                approved=approved
            )
        
        return None
    
    def _process_as_researcher(self, message: Message, context: Dict) -> Optional[Message]:
        """Process message as a researcher agent"""
        if message.message_type == MessageType.QUESTION:
            # Research the question
            prompt = f"""
You are a research agent. Investigate and answer the following question:

Question: {message.content}

Context: {json.dumps(context, indent=2)}

Provide a detailed, well-sourced answer.
"""
            research = self.llm.generate(prompt, temperature=0.6)
            
            return Message.create(
                sender=self.agent_id,
                receiver=message.sender,
                msg_type=MessageType.RESPONSE,
                content=research,
                research_completed=True
            )
        
        return None
    
    def _process_as_synthesizer(self, message: Message, context: Dict) -> Optional[Message]:
        """Process message as a synthesizer agent"""
        # Synthesize multiple inputs
        if 'synthesis_data' in context:
            prompt = f"""
You are a synthesis agent. Combine and summarize the following information:

{json.dumps(context['synthesis_data'], indent=2)}

Create a coherent, comprehensive summary.
"""
            synthesis = self.llm.generate(prompt, temperature=0.5)
            
            return Message.create(
                sender=self.agent_id,
                receiver=message.sender,
                msg_type=MessageType.RESPONSE,
                content=synthesis,
                synthesis_complete=True
            )
        
        return None
    
    def _process_generic(self, message: Message, context: Dict) -> Optional[Message]:
        """Generic message processing"""
        prompt = f"""
You are an agent with the following capabilities: {', '.join(self.capabilities)}

Message from {message.sender}: {message.content}

How do you respond?
"""
        response = self.llm.generate(prompt, temperature=0.6)
        
        return Message.create(
            sender=self.agent_id,
            receiver=message.sender,
            msg_type=MessageType.RESPONSE,
            content=response
        )
    
    def send_message(self, message: Message):
        """Record a sent message"""
        self.sent_messages.append(message)


class MultiAgentOrchestrator:
    """
    Orchestrates collaboration between multiple agents
    
    Features:
    - Dynamic agent team formation
    - Message routing and coordination
    - Workflow execution with roles
    - Self-correction through critic agents
    """
    
    def __init__(self, llm_interface):
        self.llm = llm_interface
        self.agents: Dict[str, Agent] = {}
        self.message_bus: List[Message] = []
        self.workflows: Dict[str, Dict] = {}
        
        # Shared context accessible to all agents
        self.shared_context: Dict[str, Any] = {}
    
    def register_agent(self, agent: Agent):
        """Add an agent to the orchestrator"""
        self.agents[agent.agent_id] = agent
        print(f"✓ Registered agent: {agent.agent_id} (Role: {agent.role.value})")
    
    def create_agent(
        self,
        role: AgentRole,
        capabilities: List[str],
        description: str = ""
    ) -> Agent:
        """Create and register a new agent"""
        agent_id = f"{role.value}_{len([a for a in self.agents.values() if a.role == role])}"
        agent = Agent(agent_id, role, self.llm, capabilities, description)
        self.register_agent(agent)
        return agent
    
    def send_message(self, message: Message):
        """Route a message to the appropriate agent"""
        self.message_bus.append(message)
        
        if message.receiver in self.agents:
            self.agents[message.receiver].receive_message(message)
            
            # Also update sender's sent messages
            if message.sender in self.agents:
                self.agents[message.sender].send_message(message)
    
    def process_messages(self, max_rounds: int = 10) -> List[Message]:
        """
        Process all pending messages with multi-round communication
        
        Args:
            max_rounds: Maximum communication rounds
            
        Returns:
            All messages exchanged
        """
        round_num = 0
        
        while round_num < max_rounds:
            round_num += 1
            
            # Check if any agent has pending messages
            has_pending = any(agent.inbox for agent in self.agents.values())
            
            if not has_pending:
                break
            
            print(f"\n--- Communication Round {round_num} ---")
            
            # Process each agent's inbox
            for agent in self.agents.values():
                if not agent.inbox:
                    continue
                
                # Process one message at a time
                message = agent.inbox.pop(0)
                
                print(f"  {agent.agent_id} processing message from {message.sender}")
                
                response = agent.process_message(message, self.shared_context)
                
                if response:
                    print(f"    → {agent.agent_id} sending response to {response.receiver}")
                    self.send_message(response)
        
        return self.message_bus
    
    def execute_workflow(
        self,
        workflow_name: str,
        initial_task: str,
        required_roles: Optional[List[AgentRole]] = None
    ) -> Dict[str, Any]:
        """
        Execute a collaborative workflow
        
        Args:
            workflow_name: Name of the workflow
            initial_task: Starting task
            required_roles: Agents needed for the workflow
            
        Returns:
            Workflow results
        """
        print(f"\n{'='*70}")
        print(f"🤝 EXECUTING MULTI-AGENT WORKFLOW: {workflow_name}")
        print(f"{'='*70}\n")
        print(f"Task: {initial_task}\n")
        
        # Ensure required agents exist
        if required_roles:
            for role in required_roles:
                if not any(a.role == role for a in self.agents.values()):
                    self.create_agent(role, capabilities=[f"{role.value}_ops"])
        
        # Initialize workflow context
        self.shared_context['workflow'] = workflow_name
        self.shared_context['initial_task'] = initial_task
        self.shared_context['start_time'] = time.time()
        
        # Clear message bus
        self.message_bus = []
        
        # Standard workflow: Planner → Executor → Critic → (Revise or Approve)
        workflow_result = self._execute_standard_workflow(initial_task)
        
        workflow_result['duration'] = time.time() - self.shared_context['start_time']
        workflow_result['message_count'] = len(self.message_bus)
        
        print(f"\n{'='*70}")
        print(f"✅ WORKFLOW COMPLETE")
        print(f"{'='*70}")
        print(f"Duration: {workflow_result['duration']:.2f}s")
        print(f"Messages: {workflow_result['message_count']}")
        print(f"Success: {workflow_result.get('success', False)}")
        print()
        
        return workflow_result
    
    def _execute_standard_workflow(self, task: str) -> Dict[str, Any]:
        """Execute standard Plan → Execute → Critique workflow"""
        
        # Get specialized agents
        planner = self._get_agent_by_role(AgentRole.PLANNER)
        executor = self._get_agent_by_role(AgentRole.EXECUTOR)
        critic = self._get_agent_by_role(AgentRole.CRITIC)
        
        if not all([planner, executor, critic]):
            return {'success': False, 'error': 'Missing required agents'}
        
        results = {'steps': [], 'success': False}
        
        # Step 1: Planning
        print("📋 Step 1: Planning")
        plan_msg = Message.create(
            sender='orchestrator',
            receiver=planner.agent_id,
            msg_type=MessageType.TASK,
            content=task
        )
        self.send_message(plan_msg)
        self.process_messages(max_rounds=2)
        
        # Get plan from planner's response
        plan_responses = [m for m in self.message_bus if m.sender == planner.agent_id]
        if not plan_responses:
            return {'success': False, 'error': 'Planning failed'}
        
        plan = plan_responses[-1].content
        results['steps'].append({'phase': 'planning', 'output': plan})
        print(f"✓ Plan created\n")
        
        # Step 2: Execution
        print("⚙️  Step 2: Execution")
        exec_msg = Message.create(
            sender='orchestrator',
            receiver=executor.agent_id,
            msg_type=MessageType.TASK,
            content=f"Execute this plan:\n{plan}"
        )
        self.send_message(exec_msg)
        self.process_messages(max_rounds=2)
        
        exec_responses = [m for m in self.message_bus if m.sender == executor.agent_id]
        if not exec_responses:
            return {'success': False, 'error': 'Execution failed'}
        
        execution_result = exec_responses[-1].content
        results['steps'].append({'phase': 'execution', 'output': execution_result})
        print(f"✓ Execution complete\n")
        
        # Step 3: Critique with possible revision
        max_revisions = 2
        revision_count = 0
        approved = False
        
        while not approved and revision_count < max_revisions:
            print(f"🔍 Step 3: Critique (Iteration {revision_count + 1})")
            
            critique_msg = Message.create(
                sender='orchestrator',
                receiver=critic.agent_id,
                msg_type=MessageType.RESPONSE,
                content=execution_result,
                original_task=task
            )
            self.send_message(critique_msg)
            self.process_messages(max_rounds=2)
            
            critic_responses = [m for m in self.message_bus if m.sender == critic.agent_id and m.timestamp > critique_msg.timestamp]
            if not critic_responses:
                break
            
            critique = critic_responses[-1]
            results['steps'].append({'phase': f'critique_{revision_count}', 'output': critique.content})
            
            if critique.message_type == MessageType.APPROVAL:
                approved = True
                print(f"✅ Approved\n")
            else:
                print(f"🔄 Revision requested\n")
                revision_count += 1
                
                if revision_count < max_revisions:
                    # Send back to executor for revision
                    print(f"⚙️  Revision {revision_count}")
                    revise_msg = Message.create(
                        sender='orchestrator',
                        receiver=executor.agent_id,
                        msg_type=MessageType.TASK,
                        content=f"Revise your work based on this feedback:\n{critique.content}\n\nOriginal work:\n{execution_result}"
                    )
                    self.send_message(revise_msg)
                    self.process_messages(max_rounds=2)
                    
                    # Get revised result
                    new_exec_responses = [m for m in self.message_bus if m.sender == executor.agent_id and m.timestamp > revise_msg.timestamp]
                    if new_exec_responses:
                        execution_result = new_exec_responses[-1].content
                        results['steps'].append({'phase': f'revision_{revision_count}', 'output': execution_result})
                        print(f"✓ Revision complete\n")
        
        results['success'] = approved
        results['final_output'] = execution_result
        results['revisions'] = revision_count
        
        return results
    
    def _get_agent_by_role(self, role: AgentRole) -> Optional[Agent]:
        """Get first agent with specified role"""
        for agent in self.agents.values():
            if agent.role == role:
                return agent
        return None
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get statistics about agents"""
        stats = {
            'total_agents': len(self.agents),
            'by_role': {},
            'total_messages_processed': 0,
            'total_tasks_completed': 0
        }
        
        for agent in self.agents.values():
            role = agent.role.value
            stats['by_role'][role] = stats['by_role'].get(role, 0) + 1
            stats['total_messages_processed'] += agent.state.messages_processed
            stats['total_tasks_completed'] += agent.state.tasks_completed
        
        return stats
