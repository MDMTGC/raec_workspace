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

    # ------------------------------------------------------------------
    # MEMORY DECAY SYSTEM
    # ------------------------------------------------------------------
    # Implements natural decay for low-stakes data while preserving
    # critical developmental knowledge. Aligns with RAEC's principle
    # that identity comes from immutable rules, not memory accumulation.
    # ------------------------------------------------------------------

    def decay_memories(
        self,
        decay_rate: float = 0.01,
        min_confidence: float = 0.1,
        protected_types: Optional[List[MemoryType]] = None,
        age_threshold_hours: float = 24.0
    ) -> Dict:
        """
        Apply time-based confidence decay to memories.

        Low-stakes memories (EXPERIENCE, BELIEF) decay over time.
        FACT and SUMMARY are protected by default but can be configured.
        Memories below min_confidence are deactivated (soft-deleted).

        Args:
            decay_rate: Confidence lost per day of age (default 0.01 = 1%/day)
            min_confidence: Threshold below which memories are deactivated
            protected_types: Memory types exempt from decay (default: FACT, SUMMARY)
            age_threshold_hours: Only decay memories older than this

        Returns:
            Dict with stats: decayed_count, deactivated_count, protected_count
        """
        if protected_types is None:
            protected_types = [MemoryType.FACT, MemoryType.SUMMARY]

        now = time.time()
        age_threshold_secs = age_threshold_hours * 3600

        # Get all active, non-protected memories older than threshold
        type_placeholders = ','.join('?' * len(protected_types))
        protected_values = [t.value for t in protected_types]

        self.cursor.execute(f"""
            SELECT id, memory_type, confidence, timestamp, metadata
            FROM memories
            WHERE active = 1
              AND memory_type NOT IN ({type_placeholders})
              AND (? - timestamp) > ?
        """, (*protected_values, now, age_threshold_secs))

        rows = self.cursor.fetchall()

        decayed = 0
        deactivated = 0
        protected = 0

        for row in rows:
            mem_id, mem_type, conf, ts, meta_json = row
            age_days = (now - ts) / 86400.0

            # Check if memory is protected by high access count
            meta = json.loads(meta_json) if meta_json else {}
            access_count = meta.get('access_count', 0)
            if access_count >= 10:
                # High-access memories are protected
                protected += 1
                continue

            # Calculate decayed confidence
            decay_amount = decay_rate * age_days
            new_conf = max(0.0, conf - decay_amount)

            if new_conf < min_confidence:
                # Deactivate memory
                self.cursor.execute(
                    "UPDATE memories SET active = 0, confidence = ? WHERE id = ?",
                    (new_conf, mem_id)
                )
                deactivated += 1
            else:
                # Apply decay
                self.cursor.execute(
                    "UPDATE memories SET confidence = ? WHERE id = ?",
                    (new_conf, mem_id)
                )
                decayed += 1

        self.conn.commit()

        return {
            'decayed_count': decayed,
            'deactivated_count': deactivated,
            'protected_count': protected,
            'total_processed': len(rows)
        }

    def compact_old_memories(
        self,
        age_threshold_hours: float = 168.0,  # 1 week
        cluster_size: int = 5,
        llm_interface=None
    ) -> Dict:
        """
        Compact old memories by summarizing clusters.

        Groups old memories by topic/similarity and creates SUMMARY entries,
        then deactivates the original entries. Preserves information in
        compressed form.

        Args:
            age_threshold_hours: Only compact memories older than this
            cluster_size: Minimum cluster size to trigger summarization
            llm_interface: Optional LLM for generating summaries

        Returns:
            Dict with stats: clusters_created, memories_compacted
        """
        now = time.time()
        age_threshold_secs = age_threshold_hours * 3600

        # Get old EXPERIENCE memories
        self.cursor.execute("""
            SELECT id, content, timestamp, metadata
            FROM memories
            WHERE active = 1
              AND memory_type = ?
              AND (? - timestamp) > ?
            ORDER BY timestamp
        """, (MemoryType.EXPERIENCE.value, now, age_threshold_secs))

        old_memories = self.cursor.fetchall()

        if len(old_memories) < cluster_size:
            return {'clusters_created': 0, 'memories_compacted': 0}

        # Simple clustering: group by time proximity (within 1 hour)
        clusters = []
        current_cluster = []
        last_ts = None

        for mem in old_memories:
            mem_id, content, ts, meta = mem
            if last_ts is None or (ts - last_ts) < 3600:
                current_cluster.append((mem_id, content, ts, meta))
            else:
                if len(current_cluster) >= cluster_size:
                    clusters.append(current_cluster)
                current_cluster = [(mem_id, content, ts, meta)]
            last_ts = ts

        if len(current_cluster) >= cluster_size:
            clusters.append(current_cluster)

        summaries_created = 0
        memories_compacted = 0

        for cluster in clusters:
            mem_ids = [m[0] for m in cluster]
            contents = [m[1] for m in cluster]

            # Generate summary
            if llm_interface:
                prompt = f"Summarize these related experiences in 1-2 sentences:\n\n"
                prompt += "\n".join(f"- {c}" for c in contents)
                summary_text = llm_interface.generate(prompt, max_tokens=100)
            else:
                # Simple fallback: concatenate first 50 chars of each
                summary_text = "Cluster of experiences: " + "; ".join(
                    c[:50] for c in contents[:3]
                ) + "..."

            # Create summary
            self.create_summary(mem_ids, summary_text, topic="auto_compacted")

            # Deactivate original memories
            for mem_id in mem_ids:
                self.cursor.execute(
                    "UPDATE memories SET active = 0 WHERE id = ?",
                    (mem_id,)
                )
                memories_compacted += 1

            summaries_created += 1

        self.conn.commit()

        return {
            'clusters_created': summaries_created,
            'memories_compacted': memories_compacted
        }

    def fragment_weak_beliefs(
        self,
        confidence_threshold: float = 0.3,
        llm_interface=None
    ) -> Dict:
        """
        Fragment low-confidence beliefs into atomic facts.

        When a belief falls below the threshold, extract any verifiable
        atomic facts from it before deactivating. This preserves valuable
        nuggets while discarding uncertain interpretations.

        Args:
            confidence_threshold: Beliefs below this are fragmented
            llm_interface: Optional LLM for extracting facts

        Returns:
            Dict with stats: beliefs_fragmented, facts_extracted
        """
        self.cursor.execute("""
            SELECT id, content, confidence, metadata
            FROM memories
            WHERE active = 1
              AND memory_type = ?
              AND confidence < ?
        """, (MemoryType.BELIEF.value, confidence_threshold))

        weak_beliefs = self.cursor.fetchall()
        beliefs_fragmented = 0
        facts_extracted = 0

        for belief in weak_beliefs:
            belief_id, content, conf, meta_json = belief

            # Extract facts using LLM or simple heuristic
            if llm_interface:
                prompt = (
                    f"Extract any verifiable facts from this uncertain belief. "
                    f"Return only concrete, atomic facts, one per line:\n\n{content}"
                )
                facts_text = llm_interface.generate(prompt, max_tokens=200)
                facts = [f.strip() for f in facts_text.split('\n') if f.strip()]
            else:
                # Simple heuristic: look for statements with numbers or dates
                import re
                facts = []
                sentences = content.split('.')
                for sent in sentences:
                    if re.search(r'\d', sent):  # Contains numbers
                        facts.append(sent.strip())

            # Store extracted facts
            for fact in facts:
                if len(fact) > 10:  # Skip trivial fragments
                    self.store(
                        content=fact,
                        memory_type=MemoryType.FACT,
                        confidence=0.7,  # Lower confidence since extracted
                        source=f"fragmented_from_belief:{belief_id}",
                        linked_to=[belief_id]
                    )
                    facts_extracted += 1

            # Deactivate the weak belief
            self.cursor.execute(
                "UPDATE memories SET active = 0 WHERE id = ?",
                (belief_id,)
            )
            beliefs_fragmented += 1

        self.conn.commit()

        return {
            'beliefs_fragmented': beliefs_fragmented,
            'facts_extracted': facts_extracted
        }

    def record_access(self, memory_id: int):
        """
        Record that a memory was accessed (used in retrieval).

        High-access memories are protected from decay.
        """
        self.cursor.execute(
            "SELECT metadata FROM memories WHERE id = ?",
            (memory_id,)
        )
        row = self.cursor.fetchone()
        if row:
            meta = json.loads(row[0]) if row[0] else {}
            meta['access_count'] = meta.get('access_count', 0) + 1
            meta['last_accessed'] = time.time()
            self.cursor.execute(
                "UPDATE memories SET metadata = ? WHERE id = ?",
                (json.dumps(meta), memory_id)
            )
            self.conn.commit()

    def run_maintenance(self, llm_interface=None) -> Dict:
        """
        Run full memory maintenance cycle.

        Should be called periodically (e.g., daily) to:
        1. Decay old, low-value memories
        2. Compact clusters of old experiences
        3. Fragment weak beliefs into facts

        Returns:
            Combined stats from all maintenance operations
        """
        stats = {}

        # Step 1: Decay
        decay_stats = self.decay_memories()
        stats['decay'] = decay_stats

        # Step 2: Compact (only if we have many old memories)
        compact_stats = self.compact_old_memories(llm_interface=llm_interface)
        stats['compact'] = compact_stats

        # Step 3: Fragment weak beliefs
        fragment_stats = self.fragment_weak_beliefs(llm_interface=llm_interface)
        stats['fragment'] = fragment_stats

        return stats

    def get_memory_health(self) -> Dict:
        """
        Get statistics about memory system health.

        Returns counts by type, average confidence, decay candidates, etc.
        """
        stats = {}

        # Count by type
        for mem_type in MemoryType:
            self.cursor.execute(
                "SELECT COUNT(*) FROM memories WHERE memory_type = ? AND active = 1",
                (mem_type.value,)
            )
            stats[f'{mem_type.value}_count'] = self.cursor.fetchone()[0]

        # Total active
        self.cursor.execute("SELECT COUNT(*) FROM memories WHERE active = 1")
        stats['total_active'] = self.cursor.fetchone()[0]

        # Total inactive (decayed/deactivated)
        self.cursor.execute("SELECT COUNT(*) FROM memories WHERE active = 0")
        stats['total_inactive'] = self.cursor.fetchone()[0]

        # Average confidence
        self.cursor.execute("SELECT AVG(confidence) FROM memories WHERE active = 1")
        avg = self.cursor.fetchone()[0]
        stats['avg_confidence'] = avg if avg else 0.0

        # Decay candidates (old, low-confidence, non-protected)
        now = time.time()
        self.cursor.execute("""
            SELECT COUNT(*) FROM memories
            WHERE active = 1
              AND memory_type IN (?, ?)
              AND confidence < 0.5
              AND (? - timestamp) > 86400
        """, (MemoryType.EXPERIENCE.value, MemoryType.BELIEF.value, now))
        stats['decay_candidates'] = self.cursor.fetchone()[0]

        return stats

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
