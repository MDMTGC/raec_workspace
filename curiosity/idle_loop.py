"""
Idle Loop - Background curiosity when user isn't active

This is what makes RAEC think when you're not watching.
Investigates questions, explores tangents, learns.
"""

import threading
import time
from typing import Optional, Callable, Any
from datetime import datetime, timedelta
from enum import Enum

from .engine import CuriosityEngine
from .questions import QuestionQueue, QuestionPriority


class IdleState(Enum):
    """States of the idle loop"""
    STOPPED = "stopped"
    IDLE = "idle"           # Waiting, not investigating
    CURIOUS = "curious"     # Actively investigating
    PAUSED = "paused"       # Temporarily paused


class IdleLoop:
    """
    Background loop that investigates questions when RAEC is idle.

    Behavior:
    - Waits for idle_threshold seconds of no user activity
    - Picks a question from the queue
    - Investigates it (search, fetch, synthesize)
    - Records what was learned
    - Waits before investigating another

    Transparency:
    - All activity is logged
    - User can see what was investigated
    - Can be paused/stopped anytime
    """

    def __init__(
        self,
        curiosity_engine: CuriosityEngine,
        idle_threshold: float = 60.0,      # Seconds of inactivity before investigating
        investigation_interval: float = 120.0,  # Seconds between investigations
        max_investigations_per_session: int = 5,  # Don't go overboard
        on_state_change: Optional[Callable[[IdleState], None]] = None,
        on_investigation_complete: Optional[Callable[[dict], None]] = None
    ):
        self.engine = curiosity_engine
        self.idle_threshold = idle_threshold
        self.investigation_interval = investigation_interval
        self.max_investigations = max_investigations_per_session

        # Callbacks
        self.on_state_change = on_state_change
        self.on_investigation_complete = on_investigation_complete

        # State
        self._state = IdleState.STOPPED
        self._last_user_activity = datetime.now()
        self._investigations_this_session = 0
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()

        # Investigation history (for this session)
        self._session_findings = []

    @property
    def state(self) -> IdleState:
        return self._state

    def _set_state(self, new_state: IdleState):
        """Update state and notify callback"""
        if new_state != self._state:
            self._state = new_state
            if self.on_state_change:
                try:
                    self.on_state_change(new_state)
                except Exception:
                    pass  # Don't let callback errors break the loop

    def record_user_activity(self):
        """Called when user interacts - resets idle timer"""
        self._last_user_activity = datetime.now()
        if self._state == IdleState.CURIOUS:
            # Don't interrupt active investigation, but note activity
            pass

    def start(self):
        """Start the idle loop"""
        if self._thread and self._thread.is_alive():
            return  # Already running

        self._stop_event.clear()
        self._pause_event.clear()
        self._investigations_this_session = 0
        self._session_findings = []

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._set_state(IdleState.IDLE)

    def stop(self):
        """Stop the idle loop"""
        self._stop_event.set()
        self._set_state(IdleState.STOPPED)
        if self._thread:
            self._thread.join(timeout=5.0)

    def pause(self):
        """Temporarily pause investigations"""
        self._pause_event.set()
        self._set_state(IdleState.PAUSED)

    def resume(self):
        """Resume investigations"""
        self._pause_event.clear()
        self._set_state(IdleState.IDLE)

    def _run_loop(self):
        """Main loop - runs in background thread"""
        while not self._stop_event.is_set():
            # Check if paused
            if self._pause_event.is_set():
                time.sleep(1.0)
                continue

            # Check if we've hit the investigation limit
            if self._investigations_this_session >= self.max_investigations:
                time.sleep(10.0)  # Just idle, don't investigate more
                continue

            # Check if idle long enough
            idle_duration = (datetime.now() - self._last_user_activity).total_seconds()
            if idle_duration < self.idle_threshold:
                time.sleep(5.0)  # Check again soon
                continue

            # Check if there are questions to investigate
            pending = self.engine.questions.get_next(1)
            if not pending:
                time.sleep(30.0)  # Nothing to do, check later
                continue

            # We're going to investigate!
            self._set_state(IdleState.CURIOUS)

            question = pending[0]
            result = self.engine.investigate(question)

            self._investigations_this_session += 1

            if result.get('success'):
                self._session_findings.append({
                    "question": question.question,
                    "findings": result.get('findings'),
                    "sources": result.get('sources', []),
                    "timestamp": datetime.now().isoformat()
                })

            # Notify callback
            if self.on_investigation_complete:
                try:
                    self.on_investigation_complete(result)
                except Exception:
                    pass

            self._set_state(IdleState.IDLE)

            # Wait before investigating another
            for _ in range(int(self.investigation_interval)):
                if self._stop_event.is_set():
                    break
                time.sleep(1.0)

    def get_session_findings(self) -> list:
        """Get what was learned this session"""
        return self._session_findings.copy()

    def format_session_summary(self) -> str:
        """Format session findings for display"""
        if not self._session_findings:
            return "No investigations completed this session."

        lines = [
            f"Curiosity Session Summary ({len(self._session_findings)} investigations):",
            "=" * 50
        ]

        for f in self._session_findings:
            lines.append(f"\nQ: {f['question']}")
            lines.append(f"Learned: {f['findings']}")
            if f['sources']:
                lines.append(f"Sources: {', '.join(f['sources'][:2])}")

        return "\n".join(lines)

    def get_stats(self) -> dict:
        """Get idle loop statistics"""
        return {
            "state": self._state.value,
            "investigations_this_session": self._investigations_this_session,
            "max_investigations": self.max_investigations,
            "idle_threshold": self.idle_threshold,
            "seconds_since_activity": (datetime.now() - self._last_user_activity).total_seconds(),
            "pending_questions": self.engine.get_pending_count()
        }
