# Raec - Autonomous Reasoning and Execution Core

## 🧠 Overview

Raec is a state-of-the-art autonomous agent system implementing cutting-edge research from 2025-2026 in:
- **Hierarchical Reflective Memory** (Hindsight, Membox, MemEvolve)
- **Audited Skill Graphs** (ASG-SI) for safe self-improvement
- **Runtime Tool Evolution** (Live-SWE-agent patterns)
- **Multi-Agent Orchestration** (AutoGen, CrewAI)
- **Advanced Verification** (ToolReflection, incremental inference)

## 🏗️ Architecture

### Core Components

```
raec_workspace/
├── raec_core/           # LLM interface and core utilities
├── memory/              # Hierarchical memory system
├── skills/              # Audited skill graph
├── tools/               # Runtime tool evolution
├── agents/              # Multi-agent orchestration
├── evaluators/          # Logic checking and verification
├── planner/             # Advanced planning system
└── main.py              # Integrated system entry point
```

### 1. Hierarchical Memory System (`memory/`)

**Features:**
- Four memory types: Facts, Experiences, Beliefs, Summaries
- Separate FAISS indices per type for specialized retrieval
- Belief evolution with evidence tracking
- Temporal context and narrative continuity
- Memory linking for causal chains

**Key Classes:**
- `HierarchicalMemoryDB`: Main memory interface
- `MemoryType`: Enum for memory categories
- `MemoryEntry`: Structured memory with metadata

**Usage:**
```python
from memory.memory_db import HierarchicalMemoryDB, MemoryType

memory = HierarchicalMemoryDB()

# Store a fact
fact_id = memory.store(
    content="Python is dynamically typed",
    memory_type=MemoryType.FACT,
    confidence=1.0
)

# Store an experience
exp_id = memory.store(
    content="Successfully implemented vector search",
    memory_type=MemoryType.EXPERIENCE,
    metadata={"task": "search", "success": True}
)

# Query by type
results = memory.query(
    "vector search",
    memory_types=[MemoryType.EXPERIENCE],
    k=5
)

# Evolve a belief
new_belief_id = memory.evolve_belief(
    belief_id=old_belief_id,
    new_content="Updated hypothesis",
    evidence="New experimental data",
    confidence_delta=0.2
)
```

### 2. Audited Skill Graph (`skills/`)

**Features:**
- Extract skills from successful task completions
- Verify skills with test cases
- Track skill dependencies
- Audit trail for all changes
- Performance metrics (usage, success rate)

**Key Classes:**
- `SkillGraph`: Main skill management
- `Skill`: Verified capability with evidence
- `SkillCategory`: Types of skills
- `SkillStatus`: Verification states

**Usage:**
```python
from skills.skill_graph import SkillGraph, SkillCategory

skills = SkillGraph()

# Extract skill from successful execution
skill_id = skills.extract_skill(
    task_description="Parse CSV and compute statistics",
    solution="def process_csv(data): ...",
    execution_result={"success": True, "time": 0.5},
    category=SkillCategory.DATA_PROCESSING
)

# Verify with test cases
test_cases = [
    {"description": "Test basic parsing", "expected": True},
    {"description": "Test error handling", "expected": True}
]
verified = skills.verify_skill(skill_id, test_cases)

# Query for relevant skill
matching_skill = skills.query_skill(
    "process CSV data",
    category=SkillCategory.DATA_PROCESSING
)

# Use skill
if matching_skill:
    result = skills.use_skill(matching_skill.skill_id, {"file": "data.csv"})
```

### 3. Runtime Tool Evolution (`tools/`)

**Features:**
- Generate new tools at runtime using LLM
- Verify tools with test cases
- Detect bottlenecks in execution
- Optimize existing tools
- Performance tracking

**Key Classes:**
- `ToolExecutor`: Main tool system
- `Tool`: Dynamically created tool
- `ToolType`: Categories of tools
- `ToolMetrics`: Performance tracking

**Usage:**
```python
from tools.executor import ToolExecutor, ToolType

tools = ToolExecutor()

# Generate a new tool
tool_id = tools.generate_tool(
    task_description="Extract email addresses from text",
    requirements=["Use regex", "Handle edge cases"],
    llm_interface=llm,
    tool_type=ToolType.DATA_PROCESSOR
)

# Verify tool
test_cases = [
    {"input": {"text": "Email: test@example.com"}, "expected": ["test@example.com"]}
]
verified = tools.verify_tool(tool_id, test_cases)

# Execute tool
result = tools.execute_tool(tool_id, {"text": "Contact: user@domain.com"})

# Detect bottlenecks
bottlenecks = tools.detect_bottlenecks(threshold_ms=1000)

# Optimize slow tool
optimized_id = tools.optimize_tool(
    tool_id=slow_tool_id,
    optimization_goal="reduce runtime",
    llm_interface=llm
)
```

### 4. Multi-Agent Orchestration (`agents/`)

**Features:**
- Specialized agent roles (Planner, Executor, Critic, Researcher, Synthesizer)
- Message-based communication
- Collaborative workflows
- Self-correction loops via Critic agents
- Internal negotiation

**Key Classes:**
- `MultiAgentOrchestrator`: Coordinates agents
- `Agent`: Individual agent with role
- `AgentRole`: Specializations
- `Message`: Inter-agent communication

**Usage:**
```python
from agents.orchestrator import MultiAgentOrchestrator, AgentRole

orchestrator = MultiAgentOrchestrator(llm)

# Create agents
orchestrator.create_agent(AgentRole.PLANNER, ["task_decomposition"])
orchestrator.create_agent(AgentRole.EXECUTOR, ["task_execution"])
orchestrator.create_agent(AgentRole.CRITIC, ["quality_review"])

# Execute collaborative workflow
result = orchestrator.execute_workflow(
    workflow_name="complex_task",
    initial_task="Build a web scraper with error handling",
    required_roles=[AgentRole.PLANNER, AgentRole.EXECUTOR, AgentRole.CRITIC]
)

# Result includes:
# - Plan from Planner
# - Execution from Executor
# - Critique and approval from Critic
# - Revision cycles if needed
```

### 5. Advanced Verification (`evaluators/`)

**Features:**
- Multi-level verification (syntax, logic, output, semantic, performance)
- Error correction suggestions
- Incremental reasoning verification
- LLM-powered semantic checks

**Key Classes:**
- `LogicChecker`: Main verification system
- `VerificationLevel`: Types of checks
- `VerificationResult`: Check outcomes

**Usage:**
```python
from evaluators.logic_checker import LogicChecker, VerificationLevel

checker = LogicChecker(llm)

# Verify output
passed, results = checker.verify(
    output=some_output,
    expected=expected_output,
    verification_levels=[
        VerificationLevel.SYNTAX,
        VerificationLevel.LOGIC,
        VerificationLevel.OUTPUT
    ],
    context={"task": "original task"}
)

# Get correction suggestions if failed
if not passed:
    correction = checker.suggest_correction(some_output, results, context)

# Incremental verification
reasoning_steps = [
    "Step 1: Analyze the problem",
    "Step 2: Design the solution",
    "Step 3: Implement the design"
]
step_results = checker.incremental_verify(reasoning_steps, "Build a calculator")
```

### 6. Advanced Planner (`planner/`)

**Features:**
- Memory-augmented planning
- Multi-step reasoning
- Dependency resolution
- Execution tracking
- Skill integration

**Key Classes:**
- `Planner`: Main planning system
- `PlanStep`: Individual step
- `PlanStatus`: Execution states

**Usage:**
```python
from planner.planner import Planner

planner = Planner(llm, memory, skills)

# Execute task with planning
result = planner.run(
    task="Create a data processing pipeline",
    context={"data_format": "CSV", "output": "JSON"}
)

# Result includes:
# - Generated plan
# - Execution results per step
# - Success/failure tracking
# - Memory storage of plan and results
```

## 🚀 Getting Started

### Installation

```bash
# Install dependencies
pip install -r requirements.txt --break-system-packages

# Ensure Ollama is running with your model
ollama run raec:latest
```

### Basic Usage

```python
from main import Raec

# Initialize Raec
raec = Raec()

# Execute a task
result = raec.execute_task(
    "Create a web scraper for product data",
    mode="standard"  # or "collaborative" or "incremental"
)

# Analyze system performance
raec.analyze_performance()
```

### Execution Modes

1. **Standard Mode**: Normal planning and execution
   ```python
   result = raec.execute_task(task, mode="standard")
   ```

2. **Collaborative Mode**: Multi-agent workflow with self-correction
   ```python
   result = raec.execute_task(task, mode="collaborative")
   ```

3. **Incremental Mode**: Step-by-step with verification
   ```python
   result = raec.execute_task(task, mode="incremental")
   ```

## 📊 System Flow

```
Task Input
    ↓
Check Skills (query skill graph)
    ↓
[Skill Found] → Use Skill → Execute → Verify → Record Outcome
    ↓
[No Skill] → Planning (memory-augmented)
    ↓
Execute Plan (with tools)
    ↓
Verify Results (multi-level)
    ↓
[Success] → Extract Skill → Verify Skill → Add to Graph
    ↓
Store in Memory (experiences, summaries)
    ↓
Return Results
```

## 🔬 Advanced Features

### Memory Evolution

```python
# Beliefs evolve based on new evidence
belief_id = memory.store(
    "Hypothesis: X leads to Y",
    memory_type=MemoryType.BELIEF,
    confidence=0.6
)

# Later, after testing...
new_belief_id = memory.evolve_belief(
    belief_id,
    "Updated: X leads to Y under conditions Z",
    evidence="3 experiments confirm Z is critical",
    confidence_delta=0.3  # Now 0.9 confidence
)
```

### Skill Dependencies

```python
# Build skill chains
basic_skill_id = skills.extract_skill(...)
advanced_skill_id = skills.extract_skill(...)

skills.add_skill_dependency(advanced_skill_id, basic_skill_id)

# Get full chain
chain = skills.get_skill_chain(advanced_skill_id)
```

### Dynamic Tool Creation

```python
# Raec detects it's doing the same operation repeatedly
# Automatically generates a specialized tool
tool_id = tools.generate_tool(
    "Parse nested JSON with specific schema",
    requirements=["Handle missing fields", "Validate types"],
    llm_interface=llm
)

# Tool is verified before use
verified = tools.verify_tool(tool_id, test_cases)

# If verified, tool is integrated into system
if verified:
    result = tools.execute_tool(tool_id, parameters)
```

### Multi-Agent Collaboration

```python
# Agents negotiate and self-correct
# Planner → Executor → Critic → [Revise] → Executor → Critic → Approve

result = orchestrator.execute_workflow("complex_analysis", task)

# Result shows full conversation:
# - Planner creates plan
# - Executor executes
# - Critic reviews (may request revision)
# - Executor revises
# - Critic approves
```

## 🧪 Testing

### Memory System Test
```bash
python -m memory.demo_memory
```

### Full System Test
```bash
python main.py
```

## 📈 Performance Monitoring

```python
# Get comprehensive stats
raec.analyze_performance()
```

Output includes:
- Memory utilization by type
- Skill graph metrics (verified, usage, confidence)
- Tool performance (execution time, success rate)
- Agent activity (messages, tasks completed)
- Verification pass rates
- Bottleneck detection

## 🔧 Configuration

Edit `config.yaml`:

```yaml
model:
  name: Qwen-distill-q3_K_M  # Your LLM model
  device: cuda

memory:
  db_path: data/embeddings/raec_memory.db

tools:
  python_timeout: 10
  max_api_fetch_chars: 1000

planner:
  max_steps: 5

skills:
  adapters_path: skills/adapters/

logs:
  tasks: logs/tasks/
  memory: logs/memory_snapshots/
```

## 🎯 Key Innovations

1. **Hierarchical Memory**: Not just vector retrieval - facts, experiences, beliefs, summaries with evolution
2. **Audited Skills**: Safe self-improvement with verification and audit trails
3. **Runtime Tools**: System creates tools on-demand for bottlenecks
4. **Multi-Agent**: Collaborative workflows with self-correction
5. **Advanced Verification**: Multi-level checks with LLM-powered semantic analysis

## 🔄 Self-Improvement Loop

```
Execute Task
    ↓
[Success] → Extract Pattern
    ↓
Verify Pattern → [Pass] → Store as Skill
    ↓
Use Skill in Future → Track Performance
    ↓
[Low Performance] → Deprecate or Optimize
    ↓
[Bottleneck Detected] → Generate Specialized Tool
    ↓
Verify Tool → Integrate
    ↓
Continuous Learning
```

## 📚 Research Foundations

- **Hindsight Memory**: Hierarchical organization with beliefs and experiences
- **ASG-SI**: Audited skill graph self-improvement
- **Live-SWE-agent**: Runtime tool evolution
- **AutoGen/CrewAI**: Multi-agent orchestration patterns
- **ToolReflection**: Error-aware reasoning with verification
- **Incremental Inference**: Step-by-step reasoning with checks

## 🚧 Future Enhancements

- [ ] Multi-modal memory (images, code)
- [ ] Distributed skill graph across instances
- [ ] Meta-learning for memory structure
- [ ] Advanced tool optimization with profiling
- [ ] External knowledge base integration
- [ ] Cross-agent skill sharing
- [ ] Automated test generation for skills

## 📄 License

Research prototype - use for experimentation and learning.

## 🤝 Contributing

This is a research implementation. Contributions welcome for:
- Additional agent roles
- New skill categories
- Tool optimization strategies
- Verification methods
- Integration with external systems
