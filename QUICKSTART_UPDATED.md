# Raec Quick Start Guide

## 🚀 Running Raec

### Option 1: Direct Python
```bash
cd raec_workspace
python main.py
```

### Option 2: GUI Interface
```bash
cd raec_workspace
python raec_gui.py
```

### Option 3: Import as Module
```python
from main import Raec

raec = Raec()
result = raec.execute_task("Your task here")
raec.close()
```

---

## 📋 Execution Modes

### Standard Mode (Default)
Checks skills → Plans → Executes → Verifies → Learns
```python
raec.execute_task("Calculate fibonacci", mode="standard")
```

### Collaborative Mode
Multi-agent with Planner → Executor → Critic loop
```python
raec.execute_task("Build a web scraper", mode="collaborative")
```

### Incremental Mode
Step-by-step reasoning with verification
```python
raec.execute_task("Analyze this algorithm", mode="incremental")
```

---

## 🔧 Configuration

Edit `config.yaml`:
```yaml
model:
  name: raec:latest  # Your Ollama model
  device: cuda

memory:
  db_path: data/embeddings/raec_memory.db

tools:
  python_timeout: 60

planner:
  max_steps: 10
```

---

## 📊 Performance Analysis

```python
raec = Raec()

# ... do some tasks ...

# Get comprehensive stats
stats = raec.analyze_performance()
print(stats)
```

Shows:
- Memory usage by type
- Skill graph statistics
- Tool performance
- Agent activity
- Verification pass rates
- Detected bottlenecks

---

## 💾 Memory System

### Store Different Types
```python
from memory.memory_db import MemoryType

# Store a fact
raec.memory.store(
    "Python uses dynamic typing",
    memory_type=MemoryType.FACT,
    confidence=1.0
)

# Store an experience
raec.memory.store(
    "Completed data pipeline successfully",
    memory_type=MemoryType.EXPERIENCE,
    metadata={'success': True}
)

# Store a belief
raec.memory.store(
    "Smaller models work better for simple tasks",
    memory_type=MemoryType.BELIEF,
    confidence=0.7
)
```

### Query Memory
```python
# Query specific types
results = raec.memory.query(
    "data processing",
    memory_types=[MemoryType.EXPERIENCE],
    k=5
)

# Query with filters
results = raec.memory.query(
    "machine learning",
    min_confidence=0.8,
    include_links=True
)
```

### Evolve Beliefs
```python
# Update a belief with new evidence
new_id = raec.memory.evolve_belief(
    belief_id=old_id,
    new_content="Updated hypothesis based on experiments",
    evidence="3 tests confirmed the new approach",
    confidence_delta=0.2  # Increase confidence
)
```

---

## 🎯 Skill System

### Extract Skills
```python
from skills.skill_graph import SkillCategory

skill_id = raec.skills.extract_skill(
    task_description="Parse CSV and compute statistics",
    solution="<solution code or pattern>",
    execution_result={'success': True},
    category=SkillCategory.DATA_PROCESSING
)
```

### Verify Skills
```python
test_cases = [
    {'description': 'Basic test', 'expected': True},
    {'description': 'Edge case', 'expected': True}
]

verified = raec.skills.verify_skill(skill_id, test_cases)
```

### Query Skills
```python
skill = raec.skills.query_skill(
    "process CSV data",
    category=SkillCategory.DATA_PROCESSING
)

if skill:
    result = raec.skills.use_skill(skill.skill_id, {'file': 'data.csv'})
```

---

## 🔧 Tool System

### Generate Tools
```python
from tools.executor import ToolType

tool_id = raec.tools.generate_tool(
    task_description="Extract email addresses",
    requirements=["Use regex", "Handle edge cases"],
    llm_interface=raec.llm,
    tool_type=ToolType.DATA_PROCESSOR
)
```

### Verify Tools
```python
test_cases = [
    {
        'input': {'text': 'Contact: test@example.com'},
        'expected': ['test@example.com']
    }
]

verified = raec.tools.verify_tool(tool_id, test_cases)
```

### Execute Tools
```python
result = raec.tools.execute_tool(
    tool_id,
    {'text': 'Email me at user@domain.com'}
)
```

### Detect Bottlenecks
```python
bottlenecks = raec.tools.detect_bottlenecks(threshold_ms=1000)
for b in bottlenecks:
    print(f"{b['type']}: {b['suggestion']}")
```

---

## 🤖 Multi-Agent System

### Direct Workflow Execution
```python
from agents.orchestrator import AgentRole

result = raec.orchestrator.execute_workflow(
    workflow_name="analysis_task",
    initial_task="Analyze this dataset",
    required_roles=[
        AgentRole.PLANNER,
        AgentRole.EXECUTOR,
        AgentRole.CRITIC
    ]
)
```

### Create Custom Agents
```python
raec.orchestrator.create_agent(
    AgentRole.RESEARCHER,
    capabilities=["web_search", "data_gathering"],
    description="Researches topics and gathers information"
)
```

---

## ✅ Verification System

### Multi-Level Verification
```python
from evaluators.logic_checker import VerificationLevel

passed, results = raec.evaluator.verify(
    output=some_output,
    verification_levels=[
        VerificationLevel.SYNTAX,
        VerificationLevel.LOGIC,
        VerificationLevel.OUTPUT,
        VerificationLevel.SEMANTIC
    ],
    context={'task': 'original task'}
)
```

### Incremental Reasoning Verification
```python
steps = [
    "Step 1: Analyze the problem",
    "Step 2: Design solution",
    "Step 3: Implement"
]

step_results = raec.evaluator.incremental_verify(
    steps,
    "Build a calculator"
)

for step_num, passed, feedback in step_results:
    print(f"Step {step_num}: {'✓' if passed else '✗'} {feedback}")
```

### Get Corrections
```python
if not passed:
    correction = raec.evaluator.suggest_correction(
        output,
        results,
        context
    )
    print(correction)
```

---

## 📈 Statistics

### Memory Stats
```python
facts = raec.memory.get_recent_by_type(MemoryType.FACT, limit=10)
experiences = raec.memory.get_recent_by_type(MemoryType.EXPERIENCE, limit=10)
```

### Skill Stats
```python
stats = raec.skills.get_stats()
# Returns: total, by_status, by_category, confidence, usage
```

### Tool Stats
```python
stats = raec.tools.get_tool_stats()
# Returns: total, verified, active, executions, success_rate
```

### Agent Stats
```python
stats = raec.orchestrator.get_agent_stats()
# Returns: total_agents, by_role, messages, tasks
```

### Verification Stats
```python
stats = raec.evaluator.get_verification_stats()
# Returns: total_verifications, pass_rate
```

---

## 🐛 Troubleshooting

### Ollama Not Running
```bash
# Start Ollama
ollama serve

# In another terminal, check model
ollama list
ollama run raec:latest
```

### Import Errors
```bash
# Install dependencies
pip install -r requirements.txt --break-system-packages
```

### Database Errors
```bash
# Reset databases (WARNING: loses data)
rm data/embeddings/*.db
rm skills/skill_db.json
```

### Model Issues
```bash
# Rebuild clean model
python build_clean_raec.py
```

---

## 💡 Best Practices

1. **Start Simple**: Use standard mode first
2. **Let it Learn**: Run similar tasks multiple times to build skills
3. **Check Stats**: Use `analyze_performance()` to see what's working
4. **Verify First**: Always verify extracted skills before relying on them
5. **Monitor Bottlenecks**: Check for slow operations regularly
6. **Clean Prompts**: Clear, specific tasks get better results

---

## 🎯 Example Workflows

### Learning Workflow
```python
raec = Raec()

# First time - learns
raec.execute_task("Parse CSV and find average")

# Second time - might find similar skill
raec.execute_task("Parse CSV and find sum")

# Third time - uses verified skill
raec.execute_task("Parse CSV and find max")
```

### Collaborative Workflow
```python
# Complex task requiring quality assurance
result = raec.execute_task(
    "Build a data validation pipeline with error handling",
    mode="collaborative"
)

# Result includes:
# - Initial plan from Planner
# - Implementation from Executor
# - Review from Critic
# - Revision cycles (if needed)
# - Final approval
```

### Incremental Workflow
```python
# Task requiring careful reasoning
result = raec.execute_task(
    "Explain why this algorithm is O(n log n)",
    mode="incremental"
)

# Each reasoning step is verified
# Halts if logical error detected
```

---

## 🔗 Related Files

- **Main Documentation**: `README.md`
- **API Reference**: `API_REFERENCE.md`
- **Memory Guide**: `memory/README.md`
- **Project Status**: `PROJECT_STATUS.md`
- **Integration Summary**: `INTEGRATION_COMPLETE.md`

---

## ⚡ Quick Commands

```bash
# Run main system
python main.py

# Run GUI
python raec_gui.py

# Run tests (if available)
python test_suite.py

# Run memory demo
python -m memory.demo_memory

# Analyze performance
python -c "from main import Raec; r=Raec(); r.analyze_performance()"
```

---

**Ready to use! 🚀**

The dark wizard is gone, the system is operational, and all components are working together. No more cosmic entropy monologues - just clean technical reasoning.
