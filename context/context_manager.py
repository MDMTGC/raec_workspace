"""
Context Manager - RAEC's situational awareness

Understands the current situation to respond appropriately:
- Temporal context (time, duration, patterns)
- User context (urgency, mood, preferences)
- Task context (what we're working on, history)
- Environmental context (project, files, recent actions)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from enum import Enum
import re


class TimeOfDay(Enum):
    """Rough time of day"""
    EARLY_MORNING = "early_morning"  # 5-8
    MORNING = "morning"              # 8-12
    AFTERNOON = "afternoon"          # 12-17
    EVENING = "evening"              # 17-21
    NIGHT = "night"                  # 21-5


class DayType(Enum):
    """Type of day"""
    WEEKDAY = "weekday"
    WEEKEND = "weekend"


class Urgency(Enum):
    """Perceived urgency level"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class UserMood(Enum):
    """Inferred user mood (used to adjust responses)"""
    UNKNOWN = "unknown"
    RELAXED = "relaxed"
    FOCUSED = "focused"
    FRUSTRATED = "frustrated"
    RUSHED = "rushed"


@dataclass
class UserState:
    """Inferred state of the user"""
    mood: UserMood = UserMood.UNKNOWN
    urgency: Urgency = Urgency.NORMAL
    apparent_expertise: str = "unknown"  # beginner, intermediate, expert
    engagement_level: str = "normal"     # low, normal, high
    signals: List[str] = field(default_factory=list)  # What indicated this state


@dataclass
class Context:
    """Current context snapshot"""
    # Temporal
    timestamp: datetime
    time_of_day: TimeOfDay
    day_type: DayType
    time_since_last_interaction: Optional[timedelta]
    session_duration: timedelta

    # User
    user_state: UserState
    interaction_count_today: int
    interaction_count_session: int

    # Task
    current_topic: Optional[str]
    topic_depth: int  # How deep are we in this topic?
    active_goals: List[str]
    recent_topics: List[str]

    # Environmental
    working_project: Optional[str]
    recent_files: List[str]
    recent_tools_used: List[str]

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "time_of_day": self.time_of_day.value,
            "day_type": self.day_type.value,
            "session_duration_minutes": self.session_duration.total_seconds() / 60,
            "user_state": {
                "mood": self.user_state.mood.value,
                "urgency": self.user_state.urgency.value,
                "expertise": self.user_state.apparent_expertise,
                "engagement": self.user_state.engagement_level
            },
            "interactions_today": self.interaction_count_today,
            "interactions_session": self.interaction_count_session,
            "current_topic": self.current_topic,
            "topic_depth": self.topic_depth,
            "active_goals": self.active_goals
        }


class ContextManager:
    """
    Manages RAEC's awareness of the current context.

    Updates in real-time based on interactions and time.
    """

    # Urgency signal patterns
    URGENCY_PATTERNS = {
        Urgency.CRITICAL: [
            r'\basap\b', r'\burgent\b', r'\bemergency\b', r'\bimmediately\b',
            r'\bcrash\b', r'\bdown\b', r'\bbroken\b', r'\bproduction\b'
        ],
        Urgency.HIGH: [
            r'\bquick\b', r'\bfast\b', r'\bsoon\b', r'\bhurry\b',
            r'\bdeadline\b', r'\btoday\b', r'\bnow\b'
        ],
        Urgency.LOW: [
            r'\bwhenever\b', r'\bno rush\b', r'\beventually\b',
            r'\bwhen you (have|get) (time|chance)\b'
        ]
    }

    # Mood signal patterns
    MOOD_PATTERNS = {
        UserMood.FRUSTRATED: [
            r'\bugh\b', r'\bwhy (won\'t|doesn\'t|isn\'t)\b', r'\bagain\b',
            r'\bstill\b', r'\bnot working\b', r'!{2,}', r'\bfrustrat'
        ],
        UserMood.RUSHED: [
            r'\bquick\b', r'\bfast\b', r'\bjust\b', r'\bonly\b',
            r'\bsimply\b', r'\bdon\'t need'
        ],
        UserMood.RELAXED: [
            r'\bcurious\b', r'\bwondering\b', r'\binteresting\b',
            r'\bno rush\b', r'\bwhen you can\b'
        ]
    }

    def __init__(self):
        self.session_start: datetime = datetime.now()
        self.last_interaction: Optional[datetime] = None
        self.interaction_count_today: int = 0
        self.interaction_count_session: int = 0

        # Topic tracking
        self.current_topic: Optional[str] = None
        self.topic_history: List[str] = []
        self.topic_depth: int = 0

        # User state
        self.user_state = UserState()

        # Environment
        self.working_project: Optional[str] = None
        self.recent_files: List[str] = []
        self.recent_tools: List[str] = []
        self.active_goals: List[str] = []

        # Compile patterns
        self._urgency_patterns = {
            level: [re.compile(p, re.IGNORECASE) for p in patterns]
            for level, patterns in self.URGENCY_PATTERNS.items()
        }
        self._mood_patterns = {
            mood: [re.compile(p, re.IGNORECASE) for p in patterns]
            for mood, patterns in self.MOOD_PATTERNS.items()
        }

    def update(self, user_input: str) -> Context:
        """
        Update context based on new user input.

        Returns the current context snapshot.
        """
        now = datetime.now()

        # Update interaction tracking
        self.interaction_count_session += 1
        self.interaction_count_today += 1

        time_since_last = None
        if self.last_interaction:
            time_since_last = now - self.last_interaction
        self.last_interaction = now

        # Analyze user state from input
        self._analyze_user_input(user_input)

        # Update topic tracking
        self._update_topic(user_input)

        # Build context snapshot
        return Context(
            timestamp=now,
            time_of_day=self._get_time_of_day(now),
            day_type=self._get_day_type(now),
            time_since_last_interaction=time_since_last,
            session_duration=now - self.session_start,
            user_state=self.user_state,
            interaction_count_today=self.interaction_count_today,
            interaction_count_session=self.interaction_count_session,
            current_topic=self.current_topic,
            topic_depth=self.topic_depth,
            active_goals=self.active_goals.copy(),
            recent_topics=self.topic_history[-5:],
            working_project=self.working_project,
            recent_files=self.recent_files[-10:],
            recent_tools_used=self.recent_tools[-10:]
        )

    def _get_time_of_day(self, dt: datetime) -> TimeOfDay:
        """Determine time of day"""
        hour = dt.hour
        if 5 <= hour < 8:
            return TimeOfDay.EARLY_MORNING
        elif 8 <= hour < 12:
            return TimeOfDay.MORNING
        elif 12 <= hour < 17:
            return TimeOfDay.AFTERNOON
        elif 17 <= hour < 21:
            return TimeOfDay.EVENING
        else:
            return TimeOfDay.NIGHT

    def _get_day_type(self, dt: datetime) -> DayType:
        """Determine day type"""
        if dt.weekday() < 5:
            return DayType.WEEKDAY
        return DayType.WEEKEND

    def _analyze_user_input(self, text: str):
        """Analyze user input for state signals"""
        signals = []

        # Check urgency
        detected_urgency = Urgency.NORMAL
        for level, patterns in self._urgency_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    detected_urgency = level
                    signals.append(f"urgency:{level.name}")
                    break
            if detected_urgency != Urgency.NORMAL:
                break

        self.user_state.urgency = detected_urgency

        # Check mood
        detected_mood = UserMood.UNKNOWN
        for mood, patterns in self._mood_patterns.items():
            match_count = sum(1 for p in patterns if p.search(text))
            if match_count >= 2:  # Need multiple signals
                detected_mood = mood
                signals.append(f"mood:{mood.name}")
                break
            elif match_count == 1:
                # Weak signal, don't change from current unless unknown
                if self.user_state.mood == UserMood.UNKNOWN:
                    detected_mood = mood
                    signals.append(f"mood_weak:{mood.name}")

        if detected_mood != UserMood.UNKNOWN:
            self.user_state.mood = detected_mood

        # Check message length/style for engagement
        word_count = len(text.split())
        if word_count < 5:
            self.user_state.engagement_level = "low"
            signals.append("engagement:terse")
        elif word_count > 50:
            self.user_state.engagement_level = "high"
            signals.append("engagement:detailed")
        else:
            self.user_state.engagement_level = "normal"

        self.user_state.signals = signals

    def _update_topic(self, text: str):
        """Update topic tracking"""
        # Simple topic extraction - first few meaningful words
        words = text.lower().split()
        # Filter common words
        stop_words = {'i', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be',
                      'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                      'will', 'would', 'could', 'should', 'may', 'might', 'can',
                      'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                      'it', 'this', 'that', 'what', 'how', 'why', 'when', 'where'}

        meaningful = [w for w in words if w not in stop_words and len(w) > 2][:3]

        if meaningful:
            new_topic = ' '.join(meaningful)

            if new_topic == self.current_topic:
                self.topic_depth += 1
            else:
                if self.current_topic:
                    self.topic_history.append(self.current_topic)
                self.current_topic = new_topic
                self.topic_depth = 1

    def record_file_access(self, file_path: str):
        """Record that a file was accessed"""
        if file_path not in self.recent_files:
            self.recent_files.append(file_path)
            if len(self.recent_files) > 20:
                self.recent_files.pop(0)

    def record_tool_use(self, tool_name: str):
        """Record that a tool was used"""
        self.recent_tools.append(tool_name)
        if len(self.recent_tools) > 20:
            self.recent_tools.pop(0)

    def set_project(self, project_name: str):
        """Set the current working project"""
        self.working_project = project_name

    def set_active_goals(self, goals: List[str]):
        """Set active goals"""
        self.active_goals = goals

    def get_response_hints(self) -> Dict[str, Any]:
        """
        Get hints for how to respond based on current context.

        Returns suggestions for response style and content.
        """
        hints = {
            "verbosity": "normal",
            "formality": "normal",
            "include_explanation": True,
            "include_examples": True,
            "urgency_aware": False,
            "mood_aware": False
        }

        # Adjust for urgency
        if self.user_state.urgency == Urgency.CRITICAL:
            hints["verbosity"] = "minimal"
            hints["include_explanation"] = False
            hints["include_examples"] = False
            hints["urgency_aware"] = True
        elif self.user_state.urgency == Urgency.HIGH:
            hints["verbosity"] = "concise"
            hints["include_examples"] = False
            hints["urgency_aware"] = True

        # Adjust for mood
        if self.user_state.mood == UserMood.FRUSTRATED:
            hints["formality"] = "empathetic"
            hints["mood_aware"] = True
        elif self.user_state.mood == UserMood.RUSHED:
            hints["verbosity"] = "minimal"
        elif self.user_state.mood == UserMood.RELAXED:
            hints["verbosity"] = "thorough"

        # Adjust for engagement
        if self.user_state.engagement_level == "low":
            hints["verbosity"] = "minimal"
        elif self.user_state.engagement_level == "high":
            hints["verbosity"] = "detailed"

        # Adjust for time of day
        tod = self._get_time_of_day(datetime.now())
        if tod in [TimeOfDay.NIGHT, TimeOfDay.EARLY_MORNING]:
            # People working late/early might be rushed
            if hints["verbosity"] == "normal":
                hints["verbosity"] = "concise"

        return hints

    def build_context_prompt(self) -> str:
        """Build a context description for inclusion in prompts"""
        ctx = self.update("")  # Get current context without new input

        lines = ["Current context:"]

        # Time context
        lines.append(f"- Time: {ctx.time_of_day.value}, {ctx.day_type.value}")
        lines.append(f"- Session: {int(ctx.session_duration.total_seconds() / 60)} min, {ctx.interaction_count_session} interactions")

        # User context
        if ctx.user_state.urgency != Urgency.NORMAL:
            lines.append(f"- User urgency: {ctx.user_state.urgency.name}")
        if ctx.user_state.mood != UserMood.UNKNOWN:
            lines.append(f"- User mood: {ctx.user_state.mood.value}")

        # Task context
        if ctx.current_topic:
            lines.append(f"- Current topic: {ctx.current_topic} (depth: {ctx.topic_depth})")
        if ctx.active_goals:
            lines.append(f"- Active goals: {', '.join(ctx.active_goals[:3])}")

        return "\n".join(lines)

    def reset_session(self):
        """Reset for a new session"""
        self.session_start = datetime.now()
        self.interaction_count_session = 0
        self.current_topic = None
        self.topic_depth = 0
        self.user_state = UserState()
