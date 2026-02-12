"""
RAEC Conversation Manager

Manages conversation history, session state, and cross-session continuity.
"""
import json
import os
import time
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum


class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Message:
    """A conversation message"""
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Message':
        return cls(
            role=data['role'],
            content=data['content'],
            timestamp=data.get('timestamp', time.time()),
            metadata=data.get('metadata', {})
        )


@dataclass
class Session:
    """A conversation session"""
    session_id: str
    start_time: float
    messages: List[Message] = field(default_factory=list)
    summary: Optional[str] = None
    key_topics: List[str] = field(default_factory=list)
    outcomes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'session_id': self.session_id,
            'start_time': self.start_time,
            'messages': [m.to_dict() for m in self.messages],
            'summary': self.summary,
            'key_topics': self.key_topics,
            'outcomes': self.outcomes
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Session':
        return cls(
            session_id=data['session_id'],
            start_time=data['start_time'],
            messages=[Message.from_dict(m) for m in data.get('messages', [])],
            summary=data.get('summary'),
            key_topics=data.get('key_topics', []),
            outcomes=data.get('outcomes', [])
        )


class ConversationManager:
    """
    Manages conversation state and history.

    Features:
    - Session tracking with message history
    - Cross-session summaries stored for continuity
    - Context window management (compress old messages)
    - Persistence across restarts
    """

    def __init__(self, state_path: str = None, max_context_messages: int = 20):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.state_path = state_path or os.path.join(base_dir, "conversation/conversation_state.json")
        self.max_context_messages = max_context_messages

        self.current_session: Optional[Session] = None
        self.recent_sessions: List[Dict] = []  # Summaries of past sessions

        self._load_state()
        self._ensure_session()

    def _load_state(self):
        """Load conversation state from disk"""
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, 'r') as f:
                    data = json.load(f)

                # Load current session if exists
                if data.get('current_session'):
                    self.current_session = Session.from_dict(data['current_session'])

                # Load recent session summaries
                self.recent_sessions = data.get('recent_sessions', [])

            except Exception as e:
                print(f"[!] Failed to load conversation state: {e}")

    def _ensure_session(self):
        """Ensure we have an active session"""
        if self.current_session is None:
            self.current_session = Session(
                session_id=str(uuid.uuid4())[:8],
                start_time=time.time()
            )

    def save(self):
        """Persist conversation state to disk"""
        data = {
            'current_session': self.current_session.to_dict() if self.current_session else None,
            'recent_sessions': self.recent_sessions[-10:]  # Keep last 10 session summaries
        }

        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, 'w') as f:
            json.dump(data, f, indent=2)

    def add_user_message(self, content: str, metadata: Dict = None) -> Message:
        """Add a user message to the conversation"""
        self._ensure_session()
        message = Message(
            role=MessageRole.USER.value,
            content=content,
            metadata=metadata or {}
        )
        self.current_session.messages.append(message)
        return message

    def add_assistant_message(self, content: str, metadata: Dict = None) -> Message:
        """Add an assistant message to the conversation"""
        self._ensure_session()
        message = Message(
            role=MessageRole.ASSISTANT.value,
            content=content,
            metadata=metadata or {}
        )
        self.current_session.messages.append(message)
        return message

    def add_system_message(self, content: str) -> Message:
        """Add a system message"""
        self._ensure_session()
        message = Message(
            role=MessageRole.SYSTEM.value,
            content=content
        )
        self.current_session.messages.append(message)
        return message

    def get_context_messages(self, limit: int = None) -> List[Dict]:
        """Get recent messages for LLM context"""
        if not self.current_session:
            return []

        limit = limit or self.max_context_messages
        messages = self.current_session.messages[-limit:]

        return [m.to_dict() for m in messages]

    def get_conversation_summary(self) -> str:
        """Get a summary of the current conversation for context"""
        if not self.current_session or not self.current_session.messages:
            return "No conversation history."

        msg_count = len(self.current_session.messages)
        user_msgs = [m for m in self.current_session.messages if m.role == MessageRole.USER.value]

        # Get topics from recent messages
        recent_content = " ".join([m.content[:100] for m in self.current_session.messages[-5:]])

        summary = f"Session {self.current_session.session_id}: {msg_count} messages exchanged."

        if self.current_session.key_topics:
            summary += f" Topics: {', '.join(self.current_session.key_topics)}."

        return summary

    def end_session(self, summary: str = None, outcomes: List[str] = None):
        """End current session and store summary"""
        if not self.current_session:
            return

        # Store session summary
        session_summary = {
            'session_id': self.current_session.session_id,
            'start_time': self.current_session.start_time,
            'end_time': time.time(),
            'message_count': len(self.current_session.messages),
            'summary': summary or self._generate_session_summary(),
            'key_topics': self.current_session.key_topics,
            'outcomes': outcomes or self.current_session.outcomes
        }

        self.recent_sessions.append(session_summary)
        self.save()

        # Start new session
        self.current_session = None
        self._ensure_session()

    def _generate_session_summary(self) -> str:
        """Generate a basic session summary"""
        if not self.current_session:
            return ""

        msg_count = len(self.current_session.messages)
        duration = time.time() - self.current_session.start_time
        duration_min = int(duration / 60)

        return f"Session with {msg_count} messages over {duration_min} minutes."

    def add_topic(self, topic: str):
        """Track a conversation topic"""
        self._ensure_session()
        if topic not in self.current_session.key_topics:
            self.current_session.key_topics.append(topic)

    def add_outcome(self, outcome: str):
        """Track a session outcome"""
        self._ensure_session()
        self.current_session.outcomes.append(outcome)

    def get_recent_session_summaries(self, limit: int = 3) -> List[Dict]:
        """Get summaries of recent past sessions"""
        return self.recent_sessions[-limit:]

    def clear_history(self):
        """Clear conversation history (keeps session summaries)"""
        if self.current_session:
            self.current_session.messages = []

    def message_count(self) -> int:
        """Get current session message count"""
        if self.current_session:
            return len(self.current_session.messages)
        return 0
