# Raec Quickstart Guide

## 🚀 Getting Started in 5 Minutes

### Prerequisites

- Python 3.8+
- Ollama running locally with a model
- Basic understanding of LLMs and agents

### Installation

```bash
# 1. Clone or navigate to raec_workspace
cd raec_workspace

# 2. Install dependencies
pip install -r requirements.txt

# 3. Ensure Ollama is running
ollama run raec:latest
# Or use any compatible model and update config.yaml
```

### First Run

```python
from main import Raec

# Initialize Raec (loads all systems)
raec = Raec()

# Execute a simple task
result = raec.execute_task(
    "Create a function to calculate fibonacci numbers",
    mode="standard"
)

print(f"Success: {result['success']}")
```

---

## 💡 Core Concepts in 2 Minutes

### 1. Memory = Knowledge Base

```python
from memory.memory_db import HierarchicalMemoryDB, MemoryType

memory = HierarchicalMemoryDB()

# Store facts (verified knowledge)
memory.store("Python is dynamically typed", MemoryType.FACT)

# Store experiences (what you've done)
memory.store("Built API successfully", MemoryType.EXPERIENCE)

# Store beliefs (hypotheses that evolve)
memory.store("TDD improves code quality", MemoryType.BELIEF, confidence=0.8)

# Query when you need
results = memory.query("Python programming")
```

### 2. Skills = Reusable Capabilities

```python
from skills.skill_graph import SkillGraph, SkillCategory

skills = SkillGraph()

# After solving something well...
skill_id = skills.extract_skill(
    task_description="Parse JSON and validate schema",
    solution="def parse_json(data): ...",
    execution_result={"success": True},
    category=SkillCategory.DATA_PROCESSING
)

# Verify it works
skills.verify_skill(skill_id, test_cases)

# Next time you need it...
matching_skill = skills.query_skill("JSON parsing")
if matching_skill:
    result = skills.use_skill(matching_skill.skill_id, context)
```

### 3. Multi-Agent = Team Problem Solving

```python
from agents.orchestrator import MultiAgentOrchestrator, AgentRole

orchestrator = MultiAgentOrchestrator(llm)

# Planner makes the plan
# Executor does the work
# Critic checks quality
result = orchestrator.execute_workflow(
    workflow_name="build_feature",
    initial_task="Create authentication system"
)
```

### 4. Tools = Dynamic Optimization

```python
from tools.executor import ToolExecutor

tools = ToolExecutor()

# System detects you're doing something repeatedly
bottlenecks = tools.detect_bottlenecks()

# Automatically generates specialized tool
tool_id = tools.generate_tool(
    task_description="Extract phone numbers",
    requirements=["Fast regex", "Multiple formats"],
    llm_interface=llm
)

# Verifies it works
tools.verify_tool(tool_id, test_cases)

# Now it's available for reuse
result = tools.execute_tool(tool_id, {"text": "Call 555-1234"})
```

---

## 🎯 Common Use Cases

### Use Case 1: Research Assistant

```python
raec = Raec()

# Research mode with multi-agent collaboration
result = raec.execute_task(
    "Research best practices for API design and create a summary",
    mode="collaborative"
)

# Knowledge is stored in memory automatically
# Can query later: memory.query("API design best practices")
```

### Use Case 2: Code Helper

```python
# Incremental mode for step-by-step verification
result = raec.execute_task(
    "Create a Python class for managing database connections",
    mode="incremental"
)

# Each step is verified
# If successful, skill is extracted for reuse
```

### Use Case 3: Data Processing Pipeline

```python
# Standard mode with skill reuse
result = raec.execute_task(
    "Process CSV files and generate statistical reports",
    mode="standard"
)

# If similar task exists in skills, it's reused
# Otherwise, new skill is created after success
```

### Use Case 4: System Analysis

```python
# Analyze what the system has learned
raec.analyze_performance()

# Shows:
# - Memory: Facts, experiences, beliefs learned
# - Skills: Verified capabilities available
# - Tools: Generated tools and their performance
# - Agents: Activity and collaboration stats
```

---

## 🔧 Customization

### Change LLM Model

Edit `config.yaml`:

```yaml
model:
  name: your-model-name  # e.g., llama2, mistral, etc.
  device: cuda  # or cpu
```

### Adjust Memory Settings

```yaml
memory:
  db_path: data/embeddings/custom_memory.db
```

### Configure Tools

```yaml
tools:
  python_timeout: 30  # Increase for longer scripts
  max_api_fetch_chars: 5000  # More data per fetch
```

---

## 📊 Monitoring Progress

### Check Memory Growth

```python
from memory.memory_db import HierarchicalMemoryDB, MemoryType

memory = HierarchicalMemoryDB()

# See what's been learned
recent_facts = memory.get_recent_by_type(MemoryType.FACT, limit=10)
recent_experiences = memory.get_recent_by_type(MemoryType.EXPERIENCE, limit=10)
recent_beliefs = memory.get_recent_by_type(MemoryType.BELIEF, limit=10)

print(f"Facts learned: {len(recent_facts)}")
print(f"Experiences recorded: {len(recent_experiences)}")
print(f"Beliefs formed: {len(recent_beliefs)}")
```

### Check Skill Accumulation

```python
from skills.skill_graph import SkillGraph

skills = SkillGraph()

stats = skills.get_stats()
print(f"Total skills: {stats['total_skills']}")
print(f"Verified skills: {stats['verified_count']}")
print(f"Average confidence: {stats['avg_confidence']:.1%}")
print(f"Total usage: {stats['total_usage']}")
```

### Check Tool Evolution

```python
from tools.executor import ToolExecutor

tools = ToolExecutor()

stats = tools.get_tool_stats()
print(f"Generated tools: {stats['total_tools']}")
print(f"Verified tools: {stats['verified']}")
print(f"Total executions: {stats['total_executions']}")
print(f"Success rate: {stats['avg_success_rate']:.1%}")
```

---

## 🐛 Troubleshooting

### Ollama Connection Error

```
Error: Cannot connect to Ollama at localhost:11434
```

**Solution:**
1. Check Ollama is running: `ollama list`
2. Start your model: `ollama run your-model-name`
3. Verify in config.yaml the correct model name

### Memory Database Lock

```
Error: database is locked
```

**Solution:**
```python
# Close all connections
memory.close()

# Or delete and reinitialize
import os
os.remove("data/embeddings/raec_memory.db")
memory = HierarchicalMemoryDB()
```

### Slow Performance

**Check bottlenecks:**
```python
tools = ToolExecutor()
bottlenecks = tools.detect_bottlenecks(threshold_ms=500)

for b in bottlenecks:
    print(f"Slow: {b['description']}")
    print(f"Avg: {b['avg_runtime_ms']}ms")
    print(f"Suggestion: {b['suggestion']}")
```

---

## 🎓 Learning Path

### Day 1: Basics
1. Run `demo.py` to see all features
2. Try `test_suite.py` to understand APIs
3. Read `README.md` for architecture

### Day 2: Memory & Skills
1. Work through memory demo: `python -m memory.demo_memory`
2. Experiment with skill extraction
3. Build a small project using both

### Day 3: Multi-Agent & Tools
1. Create custom agent workflows
2. Generate specialized tools
3. Optimize a real bottleneck

### Day 4: Integration
1. Connect to your own data sources
2. Build domain-specific skills
3. Create custom verification rules

### Week 2: Advanced
1. Implement custom skill categories
2. Create specialized agent roles
3. Build tool optimization strategies
4. Integrate external knowledge bases

---

## 📚 Next Steps

1. **Read the Research Context** (`RESEARCH_CONTEXT.md`)
   - Understand the cutting-edge techniques implemented

2. **Explore API Reference** (`API_REFERENCE.md`)
   - Detailed API documentation for all components

3. **Run Demonstrations** (`demo.py`)
   - See practical examples of each feature

4. **Execute Tests** (`test_suite.py`)
   - Verify everything works correctly

5. **Build Something**
   - Start with a small project
   - Let Raec learn from your work
   - Watch skills and tools accumulate

---

## 💬 Getting Help

### Check Documentation
- `README.md` - Architecture overview
- `API_REFERENCE.md` - Complete API docs
- `demo.py` - Working examples
- Component-specific READMEs in subdirectories

### Debug Mode
```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

raec = Raec()
```

### Community
- Share your use cases
- Contribute improvements
- Report issues with minimal reproduction

---

## ✅ Quick Checklist

- [ ] Ollama running with model
- [ ] Dependencies installed
- [ ] Config.yaml updated
- [ ] Run demo.py successfully
- [ ] Run test_suite.py all pass
- [ ] First task executed
- [ ] Memory storing correctly
- [ ] Skills being extracted
- [ ] Ready to build!

**You're ready to use Raec!** 🎉

Start with simple tasks and watch the system learn and improve. The more you use it, the better it gets at similar tasks through skill accumulation and memory growth.
