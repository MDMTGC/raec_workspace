"""
Hierarchical Reflective Memory Module
Based on state-of-the-art memory architectures (Hindsight, Membox)

Memory is organized into:
- Facts: Verified, atomic knowledge units
- Experiences: Task completions, interactions, outcomes
- Beliefs: Evolving hypotheses and assumptions
- Summaries: Temporal and topic-based aggregations
"""
import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import time
import json
from typing import List, Dict, Optional, Tuple
from enum import Enum


class MemoryType(Enum):
    """Categories of memory following Hindsight architecture"""
    FACT = "fact"               # Verified atomic knowledge
    EXPERIENCE = "experience"   # Task outcomes, interactions
    BELIEF = "belief"           # Evolving hypotheses
    SUMMARY = "summary"         # Temporal/topic aggregations


class MemoryEntry:
    """Structured memory entry with metadata"""
    def __init__(
        self,
        content: str,
        memory_type: MemoryType,
        timestamp: float,
        metadata: Optional[Dict] = None,
        linked_entries: Optional[List[int]] = None,
        confidence: float = 1.0,
        source: Optional[str] = None
    ):
        self.content = content
        self.memory_type = memory_type
        self.timestamp = timestamp
        self.metadata = metadata or {}
        self.linked_entries = linked_entries or []
        self.confidence = confidence
        self.source = source


class HierarchicalMemoryDB:
    """
    Advanced memory system with:
    - Hierarchical organization (facts, experiences, beliefs, summaries)
    - Temporal linking and topic continuity
    - Reflective updates and belief evolution
    - Multi-index semantic search by type
    """
    
    def __init__(self, db_path="data/embeddings/raec_memory.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        
        # Initialize schema
        self._init_schema()
        
        # Embedding model
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.dim = 384
        
        # Separate FAISS indices for each memory type
        self.indices = {
            MemoryType.FACT: faiss.IndexFlatL2(self.dim),
            MemoryType.EXPERIENCE: faiss.IndexFlatL2(self.dim),
            MemoryType.BELIEF: faiss.IndexFlatL2(self.dim),
            MemoryType.SUMMARY: faiss.IndexFlatL2(self.dim)
        }
        
        # ID mappings for each index
        self.id_maps = {
            MemoryType.FACT: [],
            MemoryType.EXPERIENCE: [],
            MemoryType.BELIEF: [],
            MemoryType.SUMMARY: []
        }
        
        # Load existing memories into indices
        self._rebuild_indices()

    def _init_schema(self):
        """Initialize database schema with hierarchical structure"""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                embedding BLOB NOT NULL,
                timestamp REAL NOT NULL,
                metadata TEXT,
                confidence REAL DEFAULT 1.0,
                source TEXT,
                active INTEGER DEFAULT 1
            )
        """)
        
        # Links between memories (for temporal and causal relationships)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_memory_id INTEGER NOT NULL,
                to_memory_id INTEGER NOT NULL,
                link_type TEXT NOT NULL,
                strength REAL DEFAULT 1.0,
                FOREIGN KEY (from_memory_id) REFERENCES memories(id),
                FOREIGN KEY (to_memory_id) REFERENCES memories(id)
            )
        """)
        
        # Topic clusters for coherence tracking
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                created_at REAL NOT NULL
            )
        """)
        
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_topics (
                memory_id INTEGER NOT NULL,
                topic_id INTEGER NOT NULL,
                relevance REAL DEFAULT 1.0,
                FOREIGN KEY (memory_id) REFERENCES memories(id),
                FOREIGN KEY (topic_id) REFERENCES topics(id),
                PRIMARY KEY (memory_id, topic_id)
            )
        """)
        
        self.conn.commit()

    def _rebuild_indices(self):
        """Load existing memories into FAISS indices"""
        for mem_type in MemoryType:
            self.cursor.execute(
                "SELECT id, embedding FROM memories WHERE memory_type=? AND active=1",
                (mem_type.value,)
            )
            rows = self.cursor.fetchall()
            
            embeddings = []
            ids = []
            for row_id, emb_blob in rows:
                emb = np.frombuffer(emb_blob, dtype='float32')
                embeddings.append(emb)
                ids.append(row_id)
            
            if embeddings:
                embeddings_array = np.vstack(embeddings)
                self.indices[mem_type].add(embeddings_array)
                self.id_maps[mem_type] = ids

    def store(
        self,
        content: str,
        memory_type: MemoryType,
        metadata: Optional[Dict] = None,
        confidence: float = 1.0,
        source: Optional[str] = None,
        linked_to: Optional[List[int]] = None
    ) -> int:
        """
        Store a new memory entry with hierarchical categorization
        
        Args:
            content: The memory content
            memory_type: Category (FACT, EXPERIENCE, BELIEF, SUMMARY)
            metadata: Additional structured data
            confidence: Confidence score (0-1)
            source: Source of this memory
            linked_to: IDs of related memories
            
        Returns:
            ID of stored memory
        """
        # Generate embedding
        emb = self.embedder.encode([content])[0].astype('float32')
        
        # Store in database
        self.cursor.execute("""
            INSERT INTO memories (content, memory_type, embedding, timestamp, metadata, confidence, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            content,
            memory_type.value,
            emb.tobytes(),
            time.time(),
            json.dumps(metadata) if metadata else None,
            confidence,
            source
        ))
        
        memory_id = self.cursor.lastrowid
        
        # Add links if specified
        if linked_to:
            for target_id in linked_to:
                self.add_link(memory_id, target_id, "related")
        
        self.conn.commit()
        
        # Add to FAISS index
        self.indices[memory_type].add(emb.reshape(1, -1))
        self.id_maps[memory_type].append(memory_id)
        
        return memory_id

    def add_link(self, from_id: int, to_id: int, link_type: str = "related", strength: float = 1.0):
        """Create a link between two memories"""
        self.cursor.execute("""
            INSERT INTO memory_links (from_memory_id, to_memory_id, link_type, strength)
            VALUES (?, ?, ?, ?)
        """, (from_id, to_id, link_type, strength))
        self.conn.commit()

    def query(
        self,
        query_text: str,
        memory_types: Optional[List[MemoryType]] = None,
        k: int = 5,
        min_confidence: float = 0.0,
        time_range: Optional[Tuple[float, float]] = None,
        include_links: bool = False
    ) -> List[Dict]:
        """
        Semantic search across memory with filtering
        
        Args:
            query_text: Search query
            memory_types: Filter by types (None = all types)
            k: Number of results per type
            min_confidence: Minimum confidence threshold
            time_range: (start_time, end_time) tuple
            include_links: Include linked memories in results
            
        Returns:
            List of memory dictionaries with content, metadata, and links
        """
        if memory_types is None:
            memory_types = list(MemoryType)
        
        q_emb = self.embedder.encode([query_text])[0].astype('float32')
        
        all_results = []
        
        for mem_type in memory_types:
            if self.indices[mem_type].ntotal == 0:
                continue
            
            # Search in FAISS
            distances, indices = self.indices[mem_type].search(q_emb.reshape(1, -1), k)
            
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= len(self.id_maps[mem_type]):
                    continue
                
                memory_id = self.id_maps[mem_type][idx]
                
                # Retrieve from database with filters
                query_parts = ["SELECT * FROM memories WHERE id=?"]
                params = [memory_id]
                
                if min_confidence > 0:
                    query_parts.append("AND confidence >= ?")
                    params.append(min_confidence)
                
                if time_range:
                    query_parts.append("AND timestamp BETWEEN ? AND ?")
                    params.extend(time_range)
                
                self.cursor.execute(" ".join(query_parts), params)
                row = self.cursor.fetchone()
                
                if row:
                    result = {
                        'id': row[0],
                        'content': row[1],
                        'memory_type': row[2],
                        'timestamp': row[4],
                        'metadata': json.loads(row[5]) if row[5] else {},
                        'confidence': row[6],
                        'source': row[7],
                        'distance': float(dist)
                    }
                    
                    # Add linked memories if requested
                    if include_links:
                        result['linked'] = self._get_linked_memories(memory_id)
                    
                    all_results.append(result)
        
        # Sort by distance (most relevant first)
        all_results.sort(key=lambda x: x['distance'])
        
        return all_results[:k * len(memory_types)]

    def _get_linked_memories(self, memory_id: int) -> List[Dict]:
        """Retrieve memories linked to a given memory"""
        self.cursor.execute("""
            SELECT m.*, ml.link_type, ml.strength 
            FROM memories m
            JOIN memory_links ml ON m.id = ml.to_memory_id
            WHERE ml.from_memory_id = ?
        """, (memory_id,))
        
        linked = []
        for row in self.cursor.fetchall():
            linked.append({
                'id': row[0],
                'content': row[1],
                'memory_type': row[2],
                'link_type': row[-2],
                'strength': row[-1]
            })
        
        return linked

    def evolve_belief(self, belief_id: int, new_content: str, evidence: str, confidence_delta: float = 0.0):
        """
        Update a belief based on new evidence (memory evolution)
        
        Args:
            belief_id: ID of belief to update
            new_content: Updated belief content
            evidence: Supporting evidence for the update
            confidence_delta: Change in confidence (-1 to +1)
        """
        # Get current belief
        self.cursor.execute("SELECT confidence FROM memories WHERE id=?", (belief_id,))
        row = self.cursor.fetchone()
        if not row:
            return
        
        current_confidence = row[0]
        new_confidence = max(0.0, min(1.0, current_confidence + confidence_delta))
        
        # Mark old belief as inactive
        self.cursor.execute("UPDATE memories SET active=0 WHERE id=?", (belief_id,))
        
        # Store new version
        new_id = self.store(
            content=new_content,
            memory_type=MemoryType.BELIEF,
            confidence=new_confidence,
            source=f"evolved_from:{belief_id}",
            metadata={'evidence': evidence, 'previous_version': belief_id}
        )
        
        # Link to previous version
        self.add_link(new_id, belief_id, "evolution", strength=1.0)
        
        return new_id

    def create_summary(self, memory_ids: List[int], summary_content: str, topic: Optional[str] = None) -> int:
        """
        Create a summary memory from multiple related memories
        
        Args:
            memory_ids: IDs of memories to summarize
            summary_content: The summary text
            topic: Optional topic label
            
        Returns:
            ID of summary memory
        """
        metadata = {
            'summarized_memories': memory_ids,
            'topic': topic
        }
        
        summary_id = self.store(
            content=summary_content,
            memory_type=MemoryType.SUMMARY,
            metadata=metadata,
            source="auto_summarization"
        )
        
        # Link to all summarized memories
        for mem_id in memory_ids:
            self.add_link(summary_id, mem_id, "summarizes")
        
        return summary_id

    def get_recent_by_type(self, memory_type: MemoryType, limit: int = 10) -> List[Dict]:
        """Get most recent memories of a specific type"""
        self.cursor.execute("""
            SELECT id, content, timestamp, confidence, metadata
            FROM memories
            WHERE memory_type=? AND active=1
            ORDER BY timestamp DESC
            LIMIT ?
        """, (memory_type.value, limit))
        
        results = []
        for row in self.cursor.fetchall():
            results.append({
                'id': row[0],
                'content': row[1],
                'timestamp': row[2],
                'confidence': row[3],
                'metadata': json.loads(row[4]) if row[4] else {}
            })
        
        return results

    def get_temporal_context(self, around_time: float, window: float = 3600) -> List[Dict]:
        """
        Get memories within a time window (for temporal continuity)
        
        Args:
            around_time: Center timestamp
            window: Time window in seconds (default: 1 hour)
        """
        start_time = around_time - window / 2
        end_time = around_time + window / 2
        
        self.cursor.execute("""
            SELECT id, content, memory_type, timestamp, confidence
            FROM memories
            WHERE timestamp BETWEEN ? AND ? AND active=1
            ORDER BY timestamp
        """, (start_time, end_time))
        
        results = []
        for row in self.cursor.fetchall():
            results.append({
                'id': row[0],
                'content': row[1],
                'memory_type': row[2],
                'timestamp': row[3],
                'confidence': row[4]
            })
        
        return results

    def close(self):
        """Close database connection"""
        self.conn.close()


# Backward compatibility wrapper
class MemoryDB(HierarchicalMemoryDB):
    """Legacy interface for backward compatibility"""
    
    def add_entry(self, content: str, tags: str = ""):
        """Legacy method - stores as EXPERIENCE by default"""
        metadata = {'tags': tags} if tags else None
        return self.store(
            content=content,
            memory_type=MemoryType.EXPERIENCE,
            metadata=metadata
        )
    
    def query(self, query_text: str, k=5):
        """Legacy method - returns just content strings"""
        results = super().query(query_text, k=k)
        return [r['content'] for r in results]
