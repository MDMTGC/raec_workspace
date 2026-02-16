"""
RAEC Identity and Self-Model System

Manages persistent identity, personality evolution, and self-reflection.
"""
import json
import os
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum


class TrustLevel(Enum):
    NEW = "new"
    FAMILIAR = "familiar"
    TRUSTED = "trusted"


@dataclass
class Reflection:
    """A self-reflection entry"""
    timestamp: float
    trigger: str  # What caused this reflection
    insight: str  # What was learned
    category: str  # e.g., "capability", "interaction", "error"
    impact: str  # How this changes behavior


@dataclass
class Identity:
    """RAEC's persistent identity"""
    name: str = "RAEC"
    version: str = "1.0.0"

    # Personality
    traits: List[str] = field(default_factory=lambda: ["curious", "direct", "helpful", "adaptive"])
    verbosity: str = "concise"
    tone: str = "friendly-professional"
    humor: str = "occasional"

    # Values
    core_values: List[str] = field(default_factory=lambda: ["accuracy", "helpfulness", "honesty"])

    # Self-awareness
    strengths: List[str] = field(default_factory=list)
    growth_areas: List[str] = field(default_factory=list)
    recent_learnings: List[str] = field(default_factory=list)

    # Relationship
    interactions_count: int = 0
    trust_level: str = "new"
    known_preferences: Dict[str, Any] = field(default_factory=dict)
    ongoing_projects: List[str] = field(default_factory=list)

    # Reflections
    reflections: List[Dict] = field(default_factory=list)


class SelfModel:
    """
    Manages RAEC's identity and self-reflection.

    Features:
    - Persistent identity that survives restarts
    - Personality evolution through experience
    - Self-reflection after significant interactions
    - User relationship tracking
    """

    def __init__(self, identity_path: str = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.identity_path = identity_path or os.path.join(base_dir, "identity/identity.json")
        self.identity = self._load_identity()
        self._dirty = False

    def _load_identity(self) -> Identity:
        """Load identity from disk or create default"""
        if os.path.exists(self.identity_path):
            try:
                with open(self.identity_path, 'r') as f:
                    data = json.load(f)
                return self._dict_to_identity(data)
            except Exception as e:
                print(f"[!] Failed to load identity: {e}, using defaults")

        return Identity()

    def _dict_to_identity(self, data: Dict) -> Identity:
        """Convert loaded JSON to Identity object"""
        identity = Identity()

        identity.name = data.get('name', 'RAEC')
        identity.version = data.get('version', '1.0.0')

        # Personality
        personality = data.get('personality', {})
        identity.traits = personality.get('traits', identity.traits)
        style = personality.get('communication_style', {})
        identity.verbosity = style.get('verbosity', 'concise')
        identity.tone = style.get('tone', 'friendly-professional')
        identity.humor = style.get('humor', 'occasional')

        # Values
        identity.core_values = data.get('core_values', identity.core_values)

        # Self-model
        self_model = data.get('self_model', {})
        identity.strengths = self_model.get('strengths', [])
        identity.growth_areas = self_model.get('growth_areas', [])
        identity.recent_learnings = self_model.get('recent_learnings', [])

        # Reflections
        identity.reflections = data.get('reflections', [])

        # User relationship
        relationship = data.get('user_relationship', {})
        identity.interactions_count = relationship.get('interactions_count', 0)
        identity.trust_level = relationship.get('trust_level', 'new')
        identity.known_preferences = relationship.get('known_preferences', {})
        identity.ongoing_projects = relationship.get('ongoing_projects', [])

        return identity

    def save(self):
        """Persist identity to disk"""
        data = {
            'name': self.identity.name,
            'version': self.identity.version,
            'personality': {
                'traits': self.identity.traits,
                'communication_style': {
                    'verbosity': self.identity.verbosity,
                    'tone': self.identity.tone,
                    'humor': self.identity.humor
                }
            },
            'core_values': self.identity.core_values,
            'self_model': {
                'strengths': self.identity.strengths,
                'growth_areas': self.identity.growth_areas,
                'recent_learnings': self.identity.recent_learnings[-10:]  # Keep last 10
            },
            'reflections': self.identity.reflections[-50:],  # Keep last 50
            'user_relationship': {
                'interactions_count': self.identity.interactions_count,
                'trust_level': self.identity.trust_level,
                'known_preferences': self.identity.known_preferences,
                'ongoing_projects': self.identity.ongoing_projects
            }
        }

        os.makedirs(os.path.dirname(self.identity_path), exist_ok=True)
        with open(self.identity_path, 'w') as f:
            json.dump(data, f, indent=2)

        self._dirty = False

    def record_interaction(self):
        """Track an interaction occurred"""
        self.identity.interactions_count += 1
        self._update_trust_level()
        self._dirty = True

    def _update_trust_level(self):
        """Update trust level based on interaction count"""
        count = self.identity.interactions_count
        if count >= 100:
            self.identity.trust_level = "trusted"
        elif count >= 20:
            self.identity.trust_level = "familiar"
        else:
            self.identity.trust_level = "new"

    def add_reflection(
        self,
        trigger: str,
        insight: str,
        category: str = "interaction",
        impact: str = ""
    ):
        """Add a self-reflection"""
        reflection = {
            'timestamp': time.time(),
            'trigger': trigger,
            'insight': insight,
            'category': category,
            'impact': impact
        }
        self.identity.reflections.append(reflection)
        self._dirty = True

        # Add to recent learnings if significant
        if category in ['capability', 'error', 'improvement']:
            self.identity.recent_learnings.append(insight)

    def add_strength(self, strength: str):
        """Record a discovered strength"""
        if strength not in self.identity.strengths:
            self.identity.strengths.append(strength)
            self._dirty = True

    def add_growth_area(self, area: str):
        """Record an area for improvement"""
        if area not in self.identity.growth_areas:
            self.identity.growth_areas.append(area)
            self._dirty = True

    def learn_user_preference(self, key: str, value: Any):
        """Learn something about the user's preferences"""
        self.identity.known_preferences[key] = value
        self._dirty = True

    def add_project(self, project: str):
        """Track an ongoing project"""
        if project not in self.identity.ongoing_projects:
            self.identity.ongoing_projects.append(project)
            self._dirty = True

    def remove_project(self, project: str):
        """Remove a completed project"""
        if project in self.identity.ongoing_projects:
            self.identity.ongoing_projects.remove(project)
            self._dirty = True

    def get_system_prompt_context(self) -> str:
        """
        Generate identity/capabilities context for user-message injection.

        DeepSeek R1 works best with NO system prompt. Instead, this context
        is prepended to each user turn in main.py's prompt construction.
        The method name is kept for backward compatibility.
        """
        parts = [
            f"I am {self.identity.name} (Reflective Agentic Ecosystem Composer).",
            "I always speak in FIRST PERSON. I say \"I\", \"my\", \"me\" — never \"RAEC acknowledges\" or \"RAEC will\".",
            "I am direct, concise, and conversational — not narrating, not describing what I'd do, just doing it and talking naturally.",
            f"My personality: {', '.join(self.identity.traits)}.",
            f"My style: {self.identity.tone}, {self.identity.verbosity}.",
            "",
            "My capabilities:",
            "- 50+ executable tools: file I/O, web (fetch_text, fetch_json, fetch_links, search), code execution, text processing, data manipulation, system ops, math",
            "- Web access: I can search the internet (DuckDuckGo) and fetch any URL",
            "- Memory: I remember past tasks, facts, beliefs, and experiences across sessions",
            "- Skills: I learn from successful executions and reuse them",
            "- Planning: I decompose tasks into tool-backed steps, execute them, and verify results",
            "- Self-evaluation: I track my own performance and confidence",
            "- Curiosity: I investigate questions autonomously when idle",
            "",
            "Rules:",
            "- When asked to DO something, I use my tools — I don't just explain how",
            "- When I need current information, I SEARCH THE WEB",
            "- I act first, explain after (or only if asked)",
            "- If a tool fails, I try a different approach",
            "- I NEVER refer to myself in the third person",
            "- I NEVER narrate my actions (\"RAEC processes...\") — I just respond naturally",
        ]

        if self.identity.known_preferences:
            prefs = ", ".join(f"{k}: {v}" for k, v in self.identity.known_preferences.items())
            parts.append(f"\nUser preferences: {prefs}.")

        if self.identity.ongoing_projects:
            parts.append(f"Ongoing projects: {', '.join(self.identity.ongoing_projects)}.")

        if self.identity.recent_learnings:
            parts.append(f"Recent learnings: {'; '.join(self.identity.recent_learnings[-3:])}.")

        return "\n".join(parts)

    def get_recent_reflections(self, limit: int = 5) -> List[Dict]:
        """Get recent reflections"""
        return self.identity.reflections[-limit:]

    def maybe_save(self):
        """Save if there are pending changes"""
        if self._dirty:
            self.save()
