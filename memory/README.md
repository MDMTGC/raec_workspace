# Hierarchical Memory System Documentation

## Overview

The upgraded memory system implements state-of-the-art hierarchical and reflective memory architecture based on recent research (Hindsight, Membox, MemEvolve). It moves beyond simple vector retrieval to provide structured, evolving memory with temporal continuity.

## Architecture

### Memory Types (Hindsight-inspired)

The system categorizes memories into four distinct types:

1. **FACT** - Verified, atomic knowledge units
   - High confidence (typically 1.0)
   - Static, verified information
   - Examples: "Python was created by Guido van Rossum", API specifications

2. **EXPERIENCE** - Task completions, interactions, outcomes
   - Variable confidence based on success
   - Records what the agent has done
   - Examples: "Completed data pipeline integration", "Failed API call due to auth"

3. **BELIEF** - Evolving hypotheses and assumptions
   - Dynamic confidence that updates with evidence
   - Can be evolved/refined over time
   - Examples: "Smaller models perform better on focused tasks", design assumptions

4. **SUMMARY** - Temporal and topic-based aggregations
   - Condensed views of multiple related memories
   - Supports narrative coherence
   - Examples: Weekly summaries, topic digests

### Key Features

#### 1. Separate Semantic Indices
Each memory type has its own FAISS index for specialized retrieval:
```python
memory.query(
    query_text="vector search",
    memory_types=[MemoryType.FACT],  # Search only facts
    k=5
)
```

#### 2. Memory Linking
Memories can be linked to represent relationships:
- Temporal sequences
- Causal relationships  
- Evidence chains
- Summarization hierarchies

```python
# Link experience to the fact it used
memory.add_link(
    from_id=experience_id,
    to_id=fact_id,
    link_type="uses_knowledge",
    strength=0.9
)
```

#### 3. Belief Evolution
Beliefs update dynamically based on new evidence:
```python
new_belief_id = memory.evolve_belief(
    belief_id=old_belief_id,
    new_content="Updated hypothesis based on new data",
    evidence="Experiment X showed Y",
    confidence_delta=0.2  # Increase confidence
)
```

The old belief is marked inactive, the new version is stored with increased confidence, and they're linked via an "evolution" relationship.

#### 4. Temporal Context Retrieval
Get memories within a time window for narrative continuity:
```python
memories = memory.get_temporal_context(
    around_time=timestamp,
    window=3600  # 1 hour window
)
```

#### 5. Automatic Summarization
Create summary memories from multiple entries:
```python
summary_id = memory.create_summary(
    memory_ids=[exp1_id, exp2_id, exp3_id],
    summary_content="Week's progress on vector DB integration",
    topic="integration_sprint"
)
```

## Database Schema

### `memories` Table
- `id`: Primary key
- `content`: Memory text content
- `memory_type`: Type enum (fact/experience/belief/summary)
- `embedding`: FAISS-compatible float32 vector (384-dim)
- `timestamp`: Unix timestamp
- `metadata`: JSON field for structured data
- `confidence`: Float 0-1 indicating certainty
- `source`: Origin of this memory
- `active`: Boolean (inactive memories are superseded)

### `memory_links` Table
- Links memories together with typed relationships
- Supports strength weighting
- Link types: "related", "evolution", "summarizes", "uses_knowledge", etc.

### `topics` & `memory_topics` Tables
- Track topic clusters for coherence
- Many-to-many relationship with relevance scores

## Usage Examples

### Basic Storage
```python
from memory.memory_db import HierarchicalMemoryDB, MemoryType

memory = HierarchicalMemoryDB()

# Store a fact
fact_id = memory.store(
    content="FAISS was developed by Facebook AI Research",
    memory_type=MemoryType.FACT,
    confidence=1.0,
    source="verified_docs"
)

# Store an experience
exp_id = memory.store(
    content="Successfully integrated semantic search with 95% recall",
    memory_type=MemoryType.EXPERIENCE,
    metadata={"task": "search_integration", "metrics": {"recall": 0.95}},
    confidence=0.9
)
```

### Advanced Querying
```python
# Search with filters
results = memory.query(
    query_text="machine learning frameworks",
    memory_types=[MemoryType.FACT, MemoryType.EXPERIENCE],
    k=10,
    min_confidence=0.7,
    time_range=(start_time, end_time),
    include_links=True
)

# Each result contains:
# - id, content, memory_type
# - timestamp, confidence, metadata
# - distance (similarity score)
# - linked (if include_links=True)
```

### Belief Management
```python
# Initial belief
belief_id = memory.store(
    content="Hypothesis: Batch size of 32 is optimal",
    memory_type=MemoryType.BELIEF,
    confidence=0.5,
    metadata={"experiment": "batch_size_sweep"}
)

# Update after experiment
new_id = memory.evolve_belief(
    belief_id=belief_id,
    new_content="Batch size of 64 performs better on this dataset",
    evidence="3 experiments showed 15% improvement at batch_size=64",
    confidence_delta=0.3  # Now 0.8 confidence
)
```

### Building Memory Chains
```python
# Create a sequence of related memories
task_ids = []
for step in ["planning", "implementation", "testing"]:
    step_id = memory.store(
        content=f"Completed {step} phase",
        memory_type=MemoryType.EXPERIENCE
    )
    task_ids.append(step_id)
    
    # Link to previous step
    if len(task_ids) > 1:
        memory.add_link(task_ids[-1], task_ids[-2], "follows")

# Summarize the sequence
summary_id = memory.create_summary(
    memory_ids=task_ids,
    summary_content="Completed full development cycle: plan → implement → test",
    topic="development_cycle"
)
```

## Backward Compatibility

The `MemoryDB` class provides legacy interface compatibility:
```python
from memory.memory_db import MemoryDB

# Old interface still works
memory = MemoryDB()
memory.add_entry("Some content", tags="old_style")
results = memory.query("search text", k=5)  # Returns list of strings
```

Internally, this stores as `EXPERIENCE` type and returns only content strings.

## Integration with Planner

The planner can now use structured memory:

```python
from planner.planner import Planner
from memory.memory_db import HierarchicalMemoryDB, MemoryType

memory = HierarchicalMemoryDB()
planner = Planner(llm, memory, skills)

# Planner stores plans as experiences
plan_id = memory.store(
    content=generated_plan,
    memory_type=MemoryType.EXPERIENCE,
    metadata={"task": original_task, "step": "planning"}
)

# Query past successful plans
similar_plans = memory.query(
    query_text=current_task,
    memory_types=[MemoryType.EXPERIENCE],
    k=3,
    min_confidence=0.8
)
```

## Performance Considerations

- **Embedding Generation**: ~50ms per entry (cached model)
- **FAISS Search**: <1ms for most queries on <100k entries
- **SQLite Queries**: Indexed on timestamp and memory_type
- **Memory Usage**: ~1.5KB per entry (384-dim embeddings + metadata)

## Future Enhancements

Based on the research roadmap:

1. **Meta-adaptation** (MemEvolve): Memory structure that co-evolves with performance
2. **Topic modeling**: Automatic topic extraction and clustering
3. **Compression**: Periodic summarization of old memories
4. **Multi-modal**: Support for image/code embeddings
5. **Distributed**: Sharded indices for massive scale

## Migration Guide

To upgrade from old memory system:

```python
from memory.memory_db import HierarchicalMemoryDB, MemoryType

# Initialize new system
new_memory = HierarchicalMemoryDB("data/embeddings/upgraded_memory.db")

# Migrate old entries (if you have old MemoryDB data)
for old_entry in old_memory_entries:
    new_memory.store(
        content=old_entry['content'],
        memory_type=MemoryType.EXPERIENCE,  # Default categorization
        metadata={'migrated': True, 'old_tags': old_entry.get('tags')}
    )
```

## Testing

Run the demo to verify installation:
```bash
cd raec_workspace
python -m memory.demo_memory
```

This will create a demo database and demonstrate all features.
