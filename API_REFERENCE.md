# Raec API Reference

## Table of Contents

1. [Memory System API](#memory-system-api)
2. [Skill Graph API](#skill-graph-api)
3. [Tool Executor API](#tool-executor-api)
4. [Multi-Agent API](#multi-agent-api)
5. [Verification API](#verification-api)
6. [Planner API](#planner-api)
7. [Main System API](#main-system-api)

---

## Memory System API

### `HierarchicalMemoryDB`

```python
from memory.memory_db import HierarchicalMemoryDB, MemoryType

memory = HierarchicalMemoryDB(db_path="data/embeddings/raec_memory.db")
```

#### Methods

##### `store(content, memory_type, metadata=None, confidence=1.0, source=None, linked_to=None)`
Store a new memory entry.

**Parameters:**
- `content` (str): Memory content
- `memory_type` (MemoryType): FACT, EXPERIENCE, BELIEF, or SUMMARY
- `metadata` (dict, optional): Additional structured data
- `confidence` (float): Confidence score 0-1
- `source` (str, optional): Origin of memory
- `linked_to` (list, optional): IDs of related memories

**Returns:** `int` - Memory ID

**Example:**
```python
memory_id = memory.store(
    content="PyTorch is a deep learning framework",
    memory_type=MemoryType.FACT,
    confidence=1.0,
    source="documentation"
)
```

##### `query(query_text, memory_types=None, k=5, min_confidence=0.0, time_range=None, include_links=False)`
Semantic search across memory.

**Parameters:**
- `query_text` (str): Search query
- `memory_types` (list[MemoryType], optional): Filter by types
- `k` (int): Number of results
- `min_confidence` (float): Minimum confidence threshold
- `time_range` (tuple, optional): (start_time, end_time)
- `include_links` (bool): Include linked memories

**Returns:** `list[dict]` - Memory entries with metadata

**Example:**
```python
results = memory.query(
    "machine learning frameworks",
    memory_types=[MemoryType.FACT, MemoryType.EXPERIENCE],
    k=10,
    min_confidence=0.7
)
```

##### `evolve_belief(belief_id, new_content, evidence, confidence_delta=0.0)`
Update a belief based on new evidence.

**Parameters:**
- `belief_id` (int): ID of belief to update
- `new_content` (str): Updated belief
- `evidence` (str): Supporting evidence
- `confidence_delta` (float): Change in confidence (-1 to +1)

**Returns:** `int` - New belief ID

**Example:**
```python
new_belief_id = memory.evolve_belief(
    belief_id=42,
    new_content="Framework choice depends on project context",
    evidence="Successfully used both in various scenarios",
    confidence_delta=0.3
)
```

##### `add_link(from_id, to_id, link_type="related", strength=1.0)`
Create a link between memories.

**Example:**
```python
memory.add_link(experience_id, fact_id, "uses_knowledge", strength=0.9)
```

##### `create_summary(memory_ids, summary_content, topic=None)`
Create a summary from multiple memories.

**Example:**
```python
summary_id = memory.create_summary(
    memory_ids=[1, 2, 3],
    summary_content="Weekly progress on ML project",
    topic="ml_development"
)
```

##### `get_temporal_context(around_time, window=3600)`
Get memories within a time window.

**Example:**
```python
recent = memory.get_temporal_context(
    around_time=time.time(),
    window=3600  # 1 hour
)
```

---

## Skill Graph API

### `SkillGraph`

```python
from skills.skill_graph import SkillGraph, SkillCategory, SkillStatus

skills = SkillGraph(storage_path="skills/skill_db.json")
```

#### Methods

##### `extract_skill(task_description, solution, execution_result, category, name=None, prerequisites=None)`
Extract a new skill from successful execution.

**Parameters:**
- `task_description` (str): What task was solved
- `solution` (str): How it was solved
- `execution_result` (dict): Execution outcomes
- `category` (SkillCategory): Skill type
- `name` (str, optional): Skill name
- `prerequisites` (list, optional): Required skills

**Returns:** `str` - Skill ID

**Example:**
```python
skill_id = skills.extract_skill(
    task_description="Parse CSV and compute statistics",
    solution="def process(file): ...",
    execution_result={"success": True, "time": 0.5},
    category=SkillCategory.DATA_PROCESSING,
    name="CSV Statistics Processor"
)
```

##### `verify_skill(skill_id, test_cases, verifier_fn=None)`
Verify a skill with test cases.

**Parameters:**
- `skill_id` (str): Skill to verify
- `test_cases` (list[dict]): Test scenarios
- `verifier_fn` (callable, optional): Custom verifier

**Returns:** `bool` - True if verified

**Example:**
```python
test_cases = [
    {"description": "Test basic", "expected": True, "actual": True},
    {"description": "Test error", "expected": True, "actual": True}
]
verified = skills.verify_skill(skill_id, test_cases)
```

##### `query_skill(task_description, category=None)`
Find a verified skill matching the task.

**Example:**
```python
matching_skill = skills.query_skill(
    "process CSV data",
    category=SkillCategory.DATA_PROCESSING
)
```

##### `use_skill(skill_id, context)`
Apply a skill to new context.

**Example:**
```python
result = skills.use_skill(skill_id, {"file": "data.csv"})
```

##### `record_skill_outcome(skill_id, success, result=None)`
Record outcome of using a skill.

**Example:**
```python
skills.record_skill_outcome(skill_id, success=True, result=data)
```

---

## Tool Executor API

### `ToolExecutor`

```python
from tools.executor import ToolExecutor, ToolType

tools = ToolExecutor(
    python_timeout=10,
    max_fetch_chars=1000
)
```

#### Methods

##### `run_python(script, description="")`
Execute Python code safely.

**Example:**
```python
result = tools.run_python(
    "print('Hello')\nresult = 42",
    "Test script"
)
```

##### `generate_tool(task_description, requirements, llm_interface, tool_type=ToolType.PYTHON_SCRIPT)`
Generate a new tool at runtime.

**Example:**
```python
tool_id = tools.generate_tool(
    task_description="Extract emails from text",
    requirements=["Use regex", "Handle edge cases"],
    llm_interface=llm,
    tool_type=ToolType.DATA_PROCESSOR
)
```

##### `verify_tool(tool_id, test_cases)`
Verify a generated tool.

**Example:**
```python
test_cases = [
    {"input": {"text": "test@example.com"}, "expected": ["test@example.com"]}
]
verified = tools.verify_tool(tool_id, test_cases)
```

##### `execute_tool(tool_id, parameters)`
Execute a registered tool.

**Example:**
```python
result = tools.execute_tool(tool_id, {"text": "Email: user@domain.com"})
```

##### `detect_bottlenecks(threshold_ms=1000)`
Analyze execution history for bottlenecks.

**Example:**
```python
bottlenecks = tools.detect_bottlenecks(threshold_ms=500)
```

##### `optimize_tool(tool_id, optimization_goal, llm_interface)`
Optimize an existing tool.

**Example:**
```python
optimized_id = tools.optimize_tool(
    tool_id=slow_tool_id,
    optimization_goal="reduce memory usage",
    llm_interface=llm
)
```

---

## Multi-Agent API

### `MultiAgentOrchestrator`

```python
from agents.orchestrator import MultiAgentOrchestrator, AgentRole

orchestrator = MultiAgentOrchestrator(llm_interface)
```

#### Methods

##### `create_agent(role, capabilities, description="")`
Create and register a new agent.

**Example:**
```python
agent = orchestrator.create_agent(
    role=AgentRole.PLANNER,
    capabilities=["task_decomposition", "strategy"],
    description="Plans complex tasks"
)
```

##### `execute_workflow(workflow_name, initial_task, required_roles=None)`
Execute a collaborative workflow.

**Parameters:**
- `workflow_name` (str): Workflow identifier
- `initial_task` (str): Starting task
- `required_roles` (list[AgentRole], optional): Required agents

**Returns:** `dict` - Workflow results

**Example:**
```python
result = orchestrator.execute_workflow(
    workflow_name="build_feature",
    initial_task="Create authentication system",
    required_roles=[
        AgentRole.PLANNER,
        AgentRole.EXECUTOR,
        AgentRole.CRITIC
    ]
)
```

##### `get_agent_stats()`
Get statistics about agents.

**Example:**
```python
stats = orchestrator.get_agent_stats()
# Returns: {total_agents, by_role, messages_processed, tasks_completed}
```

---

## Verification API

### `LogicChecker`

```python
from evaluators.logic_checker import LogicChecker, VerificationLevel

checker = LogicChecker(llm_interface)
```

#### Methods

##### `verify(output, expected=None, verification_levels=None, context=None)`
Comprehensive verification.

**Parameters:**
- `output` (any): Output to verify
- `expected` (any, optional): Expected output
- `verification_levels` (list, optional): Which checks to run
- `context` (dict, optional): Additional context

**Returns:** `tuple` - (passed: bool, results: list[VerificationResult])

**Example:**
```python
passed, results = checker.verify(
    output=code,
    verification_levels=[
        VerificationLevel.SYNTAX,
        VerificationLevel.LOGIC,
        VerificationLevel.PERFORMANCE
    ],
    context={"task": "process data"}
)
```

##### `suggest_correction(output, verification_results, context=None)`
Generate correction suggestions.

**Example:**
```python
if not passed:
    correction = checker.suggest_correction(output, results, context)
```

##### `incremental_verify(reasoning_steps, task)`
Verify each step of reasoning.

**Example:**
```python
steps = [
    "Analyze requirements",
    "Design solution",
    "Implement components"
]
step_results = checker.incremental_verify(steps, "Build system")
```

---

## Planner API

### `Planner`

```python
from planner.planner import Planner

planner = Planner(llm, memory, skills, max_steps=10)
```

#### Methods

##### `run(task, context=None)`
Execute task with planning.

**Parameters:**
- `task` (str): Task description
- `context` (dict, optional): Additional context

**Returns:** `dict` - Execution results

**Example:**
```python
result = planner.run(
    task="Build data pipeline",
    context={"data_format": "CSV", "output": "JSON"}
)
```

**Result Structure:**
```python
{
    'task': str,
    'plan_id': int,
    'steps': list[PlanStep],
    'results': {
        'success': bool,
        'steps': list,
        'errors': list
    },
    'success': bool
}
```

---

## Main System API

### `Raec`

```python
from main import Raec

raec = Raec(config_path="config.yaml")
```

#### Methods

##### `execute_task(task, mode="standard")`
Execute a task with full capabilities.

**Parameters:**
- `task` (str): Task description
- `mode` (str): Execution mode
  - `"standard"`: Normal planning
  - `"collaborative"`: Multi-agent workflow
  - `"incremental"`: Step-by-step verification

**Returns:** `dict` - Execution results

**Example:**
```python
result = raec.execute_task(
    "Create web scraper with error handling",
    mode="collaborative"
)
```

##### `analyze_performance()`
Analyze system performance and bottlenecks.

**Example:**
```python
raec.analyze_performance()
```

**Output:**
```
Memory stats: {total, by_type, ...}
Skill stats: {verified, confidence, usage, ...}
Tool stats: {total, verified, executions, ...}
Agent stats: {total, messages, tasks, ...}
Bottlenecks: [...]
```

---

## Common Patterns

### 1. Learning and Reusing Knowledge

```python
# Store knowledge as you learn
fact_id = memory.store(
    "API endpoint: /v1/users",
    memory_type=MemoryType.FACT
)

# Store experiences
exp_id = memory.store(
    "Successfully called API with rate limiting",
    memory_type=MemoryType.EXPERIENCE,
    metadata={"api": "users", "success": True}
)

# Later: Query relevant knowledge
relevant = memory.query("API call", k=5)
```

### 2. Building and Using Skills

```python
# Extract after success
skill_id = skills.extract_skill(
    task_description="API integration pattern",
    solution=implementation_code,
    execution_result={"success": True},
    category=SkillCategory.INTEGRATION
)

# Verify skill
verified = skills.verify_skill(skill_id, test_cases)

# Reuse later
if skills.has_skill("API integration"):
    matching = skills.query_skill("integrate external API")
    result = skills.use_skill(matching.skill_id, context)
```

### 3. Multi-Agent Problem Solving

```python
# Create specialized team
orchestrator.create_agent(AgentRole.PLANNER, ["planning"])
orchestrator.create_agent(AgentRole.EXECUTOR, ["execution"])
orchestrator.create_agent(AgentRole.CRITIC, ["review"])

# Execute with collaboration
result = orchestrator.execute_workflow(
    "complex_task",
    "Build robust system with error handling"
)
```

### 4. Quality Assurance

```python
# Multi-level verification
passed, results = checker.verify(
    output=implementation,
    verification_levels=[
        VerificationLevel.SYNTAX,
        VerificationLevel.LOGIC,
        VerificationLevel.PERFORMANCE
    ]
)

# Get corrections if failed
if not passed:
    suggestions = checker.suggest_correction(implementation, results)
```

### 5. Continuous Improvement

```python
# Detect bottlenecks
bottlenecks = tools.detect_bottlenecks()

# Generate optimized tool
if bottlenecks:
    tool_id = tools.generate_tool(
        task_description=f"Optimize {bottlenecks[0]['description']}",
        requirements=["Fast", "Memory efficient"],
        llm_interface=llm
    )
```

---

## Error Handling

All major APIs include error handling. Common patterns:

```python
try:
    result = raec.execute_task(task)
    if result['success']:
        # Handle success
        pass
    else:
        # Handle execution failure
        errors = result.get('errors', [])
except Exception as e:
    # Handle system error
    print(f"System error: {e}")
```

---

## Configuration

### config.yaml Structure

```yaml
model:
  name: your-model-name
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

---

## Examples

See `demo.py` for comprehensive usage examples of all features.

See `test_suite.py` for testing patterns and assertions.
