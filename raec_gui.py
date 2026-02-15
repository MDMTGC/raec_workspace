"""
RAEC GUI - Presence Interface
A living, breathing interface for a frontier-class local AI
"""
import customtkinter as ctk
import threading
import traceback
import sys
import os
import math
import time
from datetime import datetime
from typing import Optional, Tuple
from PIL import Image, ImageDraw, ImageFilter

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from main import Raec
except ImportError as e:
    print(f"BOOT ERROR: {e}")
    sys.exit(1)

# Disable default theme - we're going custom
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


# ═══════════════════════════════════════════════════════════════════════════════
# COLOR SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class Colors:
    """RAEC color palette - deep blacks with cyan accent"""
    # Backgrounds (near-black gradient)
    BG_DEEP = "#050508"
    BG_PRIMARY = "#0A0A0F"
    BG_ELEVATED = "#0F0F16"
    BG_SURFACE = "#14141D"

    # Accent (cyan/teal - the RAEC signature)
    ACCENT = "#00D4AA"
    ACCENT_DIM = "#00A080"
    ACCENT_GLOW = "#00FFD0"
    ACCENT_SUBTLE = "#004D40"

    # Text
    TEXT_PRIMARY = "#E8E8EC"
    TEXT_SECONDARY = "#8888A0"
    TEXT_MUTED = "#505068"
    TEXT_DIM = "#303040"

    # States
    PROCESSING = "#FFB020"
    SUCCESS = "#00D060"
    ERROR = "#FF4060"
    CURIOUS = "#A070FF"  # Soft purple when exploring autonomously

    # Borders
    BORDER_SUBTLE = "#1A1A24"
    BORDER_ACTIVE = "#252535"


# ═══════════════════════════════════════════════════════════════════════════════
# BREATHING ORB - The living presence indicator
# ═══════════════════════════════════════════════════════════════════════════════

class BreathingOrb(ctk.CTkCanvas):
    """
    An ambient orb that breathes when idle, pulses when thinking.
    The visual soul of RAEC.
    """

    def __init__(self, parent, size=60):
        super().__init__(
            parent,
            width=size,
            height=size,
            bg=Colors.BG_PRIMARY,
            highlightthickness=0
        )

        self.size = size
        self.center = size // 2
        self.base_radius = size // 4

        # Animation state
        self.phase = 0.0
        self.state = "idle"  # idle, thinking, success, error
        self.intensity = 0.3
        self.target_intensity = 0.3

        # Start animation
        self._animate()

    def set_state(self, state: str):
        """Change orb state: idle, thinking, success, error, curious"""
        self.state = state
        if state == "idle":
            self.target_intensity = 0.3
        elif state == "thinking":
            self.target_intensity = 0.8
        elif state == "success":
            self.target_intensity = 0.6
        elif state == "error":
            self.target_intensity = 0.7
        elif state == "curious":
            self.target_intensity = 0.5

    def _get_color(self) -> str:
        """Get current color based on state"""
        if self.state == "thinking":
            return Colors.PROCESSING
        elif self.state == "success":
            return Colors.SUCCESS
        elif self.state == "error":
            return Colors.ERROR
        elif self.state == "curious":
            return Colors.CURIOUS
        return Colors.ACCENT

    def _animate(self):
        """Render one frame of the breathing animation"""
        self.delete("all")

        # Smooth intensity transition
        self.intensity += (self.target_intensity - self.intensity) * 0.1

        # Calculate breath cycle
        if self.state == "thinking":
            # Faster pulse when thinking
            breath = math.sin(self.phase * 3) * 0.5 + 0.5
            self.phase += 0.15
        elif self.state == "curious":
            # Wandering rhythm when exploring — dual sine for organic feel
            breath = (math.sin(self.phase * 1.3) * 0.3 + math.sin(self.phase * 0.7) * 0.2) + 0.5
            self.phase += 0.06
        else:
            # Slow, calm breathing when idle
            breath = math.sin(self.phase) * 0.5 + 0.5
            self.phase += 0.03

        color = self._get_color()

        # Draw outer glow layers (creates soft halo effect)
        for i in range(5, 0, -1):
            glow_radius = self.base_radius + (i * 4) + (breath * 6)
            alpha = int(self.intensity * 40 * (1 - i/6))
            glow_color = self._blend_color(Colors.BG_PRIMARY, color, alpha/255)

            self.create_oval(
                self.center - glow_radius,
                self.center - glow_radius,
                self.center + glow_radius,
                self.center + glow_radius,
                fill=glow_color,
                outline=""
            )

        # Draw core
        core_radius = self.base_radius + (breath * 3)
        core_alpha = int(self.intensity * 200 + 55)
        core_color = self._blend_color(Colors.BG_ELEVATED, color, core_alpha/255)

        self.create_oval(
            self.center - core_radius,
            self.center - core_radius,
            self.center + core_radius,
            self.center + core_radius,
            fill=core_color,
            outline=""
        )

        # Inner bright spot
        inner_radius = core_radius * 0.4
        self.create_oval(
            self.center - inner_radius,
            self.center - inner_radius,
            self.center + inner_radius,
            self.center + inner_radius,
            fill=color,
            outline=""
        )

        # Schedule next frame
        self.after(33, self._animate)  # ~30 FPS

    def _blend_color(self, base: str, overlay: str, alpha: float) -> str:
        """Blend two hex colors"""
        def hex_to_rgb(h):
            h = h.lstrip('#')
            return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

        def rgb_to_hex(r, g, b):
            return f"#{int(r):02x}{int(g):02x}{int(b):02x}"

        br, bg, bb = hex_to_rgb(base)
        or_, og, ob = hex_to_rgb(overlay)

        nr = br + (or_ - br) * alpha
        ng = bg + (og - bg) * alpha
        nb = bb + (ob - bb) * alpha

        return rgb_to_hex(nr, ng, nb)


# ═══════════════════════════════════════════════════════════════════════════════
# MESSAGE DISPLAY - Clean, conversational message rendering
# ═══════════════════════════════════════════════════════════════════════════════

class MessageBubble(ctk.CTkFrame):
    """A single message in the conversation"""

    def __init__(self, parent, role: str, content: str, timestamp: str = None):
        is_user = role.lower() == "user"

        super().__init__(
            parent,
            fg_color=Colors.BG_SURFACE if is_user else "transparent",
            corner_radius=12
        )

        # Role indicator
        role_text = "You" if is_user else "RAEC"
        role_color = Colors.TEXT_SECONDARY if is_user else Colors.ACCENT

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=16, pady=(12, 4))

        role_label = ctk.CTkLabel(
            header,
            text=role_text,
            font=("Segoe UI", 11, "bold"),
            text_color=role_color
        )
        role_label.pack(side="left")

        if timestamp:
            time_label = ctk.CTkLabel(
                header,
                text=timestamp,
                font=("Segoe UI", 10),
                text_color=Colors.TEXT_DIM
            )
            time_label.pack(side="right")

        # Content
        content_label = ctk.CTkLabel(
            self,
            text=content,
            font=("Segoe UI", 13),
            text_color=Colors.TEXT_PRIMARY,
            wraplength=600,
            justify="left",
            anchor="w"
        )
        content_label.pack(fill="x", padx=16, pady=(0, 12))


class ConversationView(ctk.CTkScrollableFrame):
    """The main conversation display area"""

    def __init__(self, parent):
        super().__init__(
            parent,
            fg_color=Colors.BG_PRIMARY,
            corner_radius=0,
            scrollbar_button_color=Colors.BG_ELEVATED,
            scrollbar_button_hover_color=Colors.BORDER_ACTIVE
        )

        self.messages = []

        # Welcome message
        self._add_system_message("RAEC is waking up...")

    def add_message(self, role: str, content: str):
        """Add a message to the conversation"""
        timestamp = datetime.now().strftime("%H:%M")

        bubble = MessageBubble(self, role, content, timestamp)
        bubble.pack(fill="x", padx=20, pady=6)

        self.messages.append(bubble)

        # Scroll to bottom
        self.after(50, lambda: self._parent_canvas.yview_moveto(1.0))

    def _add_system_message(self, text: str):
        """Add a system status message"""
        label = ctk.CTkLabel(
            self,
            text=text,
            font=("Cascadia Mono", 11),
            text_color=Colors.TEXT_MUTED
        )
        label.pack(pady=20)

    def update_system_message(self, text: str):
        """Update the initial system message"""
        for widget in self.winfo_children():
            if isinstance(widget, ctk.CTkLabel):
                widget.configure(text=text)
                break

    def clear(self):
        """Clear all messages"""
        for widget in self.winfo_children():
            widget.destroy()
        self.messages = []


# ═══════════════════════════════════════════════════════════════════════════════
# INPUT BAR - The command interface
# ═══════════════════════════════════════════════════════════════════════════════

class InputBar(ctk.CTkFrame):
    """The input area with integrated controls"""

    def __init__(self, parent, on_submit):
        super().__init__(parent, fg_color=Colors.BG_ELEVATED, corner_radius=16)

        self.on_submit = on_submit

        # Mode indicator (subtle)
        self.mode_var = ctk.StringVar(value="auto")
        self.mode_btn = ctk.CTkButton(
            self,
            textvariable=self.mode_var,
            width=60,
            height=32,
            fg_color="transparent",
            hover_color=Colors.BG_SURFACE,
            text_color=Colors.TEXT_MUTED,
            font=("Cascadia Mono", 10),
            command=self._cycle_mode
        )
        self.mode_btn.pack(side="left", padx=(12, 0), pady=10)

        # Main input
        self.entry = ctk.CTkEntry(
            self,
            placeholder_text="Say something...",
            font=("Segoe UI", 14),
            height=44,
            fg_color="transparent",
            border_width=0,
            text_color=Colors.TEXT_PRIMARY,
            placeholder_text_color=Colors.TEXT_MUTED
        )
        self.entry.pack(side="left", fill="x", expand=True, padx=12, pady=8)
        self.entry.bind("<Return>", lambda e: self._submit())

        # Submit hint
        self.submit_hint = ctk.CTkLabel(
            self,
            text="⏎",
            font=("Segoe UI", 16),
            text_color=Colors.TEXT_DIM
        )
        self.submit_hint.pack(side="right", padx=16)

    def _cycle_mode(self):
        """Cycle through modes"""
        modes = ["auto", "standard", "collab", "step"]
        current = self.mode_var.get()
        idx = modes.index(current) if current in modes else 0
        self.mode_var.set(modes[(idx + 1) % len(modes)])

    def _submit(self):
        """Handle submission"""
        text = self.entry.get().strip()
        if text:
            self.entry.delete(0, "end")
            mode = self.mode_var.get()
            # Map short names to full names
            mode_map = {"collab": "collaborative", "step": "incremental"}
            self.on_submit(text, mode_map.get(mode, mode))

    def set_enabled(self, enabled: bool):
        """Enable/disable input"""
        state = "normal" if enabled else "disabled"
        self.entry.configure(state=state)
        self.entry.configure(
            placeholder_text="Say something..." if enabled else "Thinking..."
        )

    def focus(self):
        """Focus the input"""
        self.entry.focus()


# ═══════════════════════════════════════════════════════════════════════════════
# STATUS BAR - Minimal, informative
# ═══════════════════════════════════════════════════════════════════════════════

class StatusBar(ctk.CTkFrame):
    """Minimal status bar at the bottom"""

    def __init__(self, parent):
        super().__init__(parent, fg_color=Colors.BG_DEEP, height=28, corner_radius=0)
        self.pack_propagate(False)

        # Left: Status
        self.status = ctk.CTkLabel(
            self,
            text="Initializing...",
            font=("Cascadia Mono", 10),
            text_color=Colors.TEXT_MUTED
        )
        self.status.pack(side="left", padx=16)

        # Right: Session info
        self.session_info = ctk.CTkLabel(
            self,
            text="",
            font=("Cascadia Mono", 10),
            text_color=Colors.TEXT_DIM
        )
        self.session_info.pack(side="right", padx=16)

        # Intent (center-ish)
        self.intent = ctk.CTkLabel(
            self,
            text="",
            font=("Cascadia Mono", 10),
            text_color=Colors.ACCENT_DIM
        )
        self.intent.pack(side="right", padx=16)

    def set_status(self, text: str, color: str = None):
        """Update status text"""
        self.status.configure(
            text=text,
            text_color=color or Colors.TEXT_MUTED
        )

    def set_intent(self, intent: str, confidence: float):
        """Show detected intent"""
        self.intent.configure(text=f"{intent} ({confidence:.0%})")

    def set_session(self, session_id: str, interactions: int):
        """Update session info"""
        self.session_info.configure(text=f"#{session_id[:6]} · {interactions} msgs")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

class RaecGUI(ctk.CTk):
    """
    RAEC Presence Interface

    A minimal, living interface that prioritizes conversation
    while maintaining awareness of system state.
    """

    def __init__(self):
        super().__init__()

        # Window setup
        self.title("RAEC")
        self.geometry("900x700")
        self.minsize(700, 500)
        self.configure(fg_color=Colors.BG_PRIMARY)

        # State
        self.core: Optional[Raec] = None
        self.is_executing = False
        self.command_history = []
        self.history_index = -1

        # Build UI
        self._setup_ui()
        self._bind_events()

        # Initialize core
        threading.Thread(target=self._init_core, daemon=True).start()

    def _setup_ui(self):
        """Build the interface"""
        # Main container
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # ─── Header ───────────────────────────────────────────────────────────
        header = ctk.CTkFrame(self, fg_color="transparent", height=80)
        header.grid(row=0, column=0, sticky="ew", padx=30, pady=(20, 0))
        header.grid_propagate(False)

        # Breathing orb
        self.orb = BreathingOrb(header, size=50)
        self.orb.pack(side="left", padx=(0, 16))

        # Title
        title_frame = ctk.CTkFrame(header, fg_color="transparent")
        title_frame.pack(side="left", fill="y")

        title = ctk.CTkLabel(
            title_frame,
            text="RAEC",
            font=("Segoe UI", 24, "bold"),
            text_color=Colors.TEXT_PRIMARY
        )
        title.pack(anchor="w")

        self.subtitle = ctk.CTkLabel(
            title_frame,
            text="Waking up...",
            font=("Segoe UI", 12),
            text_color=Colors.TEXT_MUTED
        )
        self.subtitle.pack(anchor="w")

        # ─── Conversation ─────────────────────────────────────────────────────
        self.conversation = ConversationView(self)
        self.conversation.grid(row=1, column=0, sticky="nsew", padx=30, pady=20)

        # ─── Input ────────────────────────────────────────────────────────────
        self.input_bar = InputBar(self, self._on_submit)
        self.input_bar.grid(row=2, column=0, sticky="ew", padx=30, pady=(0, 20))

        # ─── Status ───────────────────────────────────────────────────────────
        self.status_bar = StatusBar(self)
        self.status_bar.grid(row=3, column=0, sticky="ew")

    def _bind_events(self):
        """Setup event bindings"""
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Keyboard shortcuts
        self.bind("<Control-l>", lambda e: self.conversation.clear())
        self.bind("<Escape>", lambda e: self.input_bar.focus())

        # History navigation
        self.input_bar.entry.bind("<Up>", self._history_prev)
        self.input_bar.entry.bind("<Down>", self._history_next)

    def _init_core(self):
        """Initialize RAEC core"""
        try:
            self.status_bar.set_status("Loading core...", Colors.PROCESSING)
            self.core = Raec()

            # Update UI
            self.after(0, self._on_core_ready)

        except Exception as e:
            self.after(0, lambda: self._on_core_error(str(e)))
            traceback.print_exc()

    def _on_core_ready(self):
        """Called when core is ready"""
        self.subtitle.configure(text="Ready")
        self.orb.set_state("idle")
        self.status_bar.set_status("Ready", Colors.SUCCESS)

        # Update session info
        if hasattr(self.core, 'conversation'):
            session_id = self.core.conversation.current_session.session_id
            interactions = self.core.identity.identity.interactions_count
            self.status_bar.set_session(session_id, interactions)

        # Wire curiosity state changes into GUI
        if hasattr(self.core, 'idle_loop'):
            self.core.idle_loop.on_state_change = self._on_curiosity_state
            self.core.idle_loop.on_investigation_complete = self._on_curiosity_finding

        # Show greeting with any pending notifications
        greeting = self.core.get_session_greeting() if hasattr(self.core, 'get_session_greeting') else "RAEC is ready. How can I help?"
        self.conversation.update_system_message(greeting)
        self.input_bar.set_enabled(True)
        self.input_bar.focus()

    def _on_core_error(self, error: str):
        """Called when core fails to load"""
        self.subtitle.configure(text="Offline", text_color=Colors.ERROR)
        self.orb.set_state("error")
        self.status_bar.set_status(f"Error: {error}", Colors.ERROR)
        self.conversation.update_system_message(f"Failed to start: {error}")

    def _on_curiosity_state(self, state):
        """Called from background thread when curiosity state changes"""
        state_val = state.value if hasattr(state, 'value') else str(state)
        if state_val == "curious":
            self.after(0, lambda: self.orb.set_state("curious"))
            self.after(0, lambda: self.subtitle.configure(text="Exploring..."))
            self.after(0, lambda: self.status_bar.set_status("Investigating a question...", Colors.CURIOUS))
        elif state_val == "idle" and not self.is_executing:
            self.after(0, lambda: self.orb.set_state("idle"))
            self.after(0, lambda: self.subtitle.configure(text="Ready"))
            self.after(0, lambda: self.status_bar.set_status("Ready", Colors.SUCCESS))

    def _on_curiosity_finding(self, result: dict):
        """Called from background thread when curiosity investigation completes"""
        if result.get('success'):
            question = result.get('question', '')[:80]
            findings = result.get('findings', '')[:200]
            msg = f"I looked into: {question}\nLearned: {findings}"
            self.after(0, lambda: self.conversation.add_message("assistant", msg))

    def _on_submit(self, text: str, mode: str):
        """Handle user input"""
        if self.is_executing or not self.core:
            return

        # Add to history
        self.command_history.append(text)
        self.history_index = -1

        # Show user message
        self.conversation.add_message("user", text)

        # Execute
        threading.Thread(
            target=self._execute,
            args=(text, mode),
            daemon=True
        ).start()

    def _execute(self, text: str, mode: str):
        """Execute in background"""
        self.is_executing = True
        self.after(0, lambda: self.input_bar.set_enabled(False))
        self.after(0, lambda: self.orb.set_state("thinking"))
        self.after(0, lambda: self.status_bar.set_status("Thinking...", Colors.PROCESSING))

        try:
            # Get intent for display
            if hasattr(self.core, 'intent_classifier'):
                classification = self.core.intent_classifier.classify(text)
                self.after(0, lambda: self.status_bar.set_intent(
                    classification.intent.value,
                    classification.confidence
                ))

            # Process
            result = self.core.process_input(text, mode=mode)

            # Show response
            self.after(0, lambda: self.conversation.add_message("assistant", result))
            self.after(0, lambda: self.orb.set_state("success"))
            self.after(0, lambda: self.status_bar.set_status("Ready", Colors.SUCCESS))

            # Update session info
            if hasattr(self.core, 'identity'):
                interactions = self.core.identity.identity.interactions_count
                session_id = self.core.conversation.current_session.session_id
                self.after(0, lambda: self.status_bar.set_session(session_id, interactions))

            # Return to idle after a moment
            self.after(1500, lambda: self.orb.set_state("idle"))

        except Exception as e:
            self.after(0, lambda: self.conversation.add_message(
                "assistant", f"Error: {str(e)}"
            ))
            self.after(0, lambda: self.orb.set_state("error"))
            self.after(0, lambda: self.status_bar.set_status(f"Error", Colors.ERROR))
            self.after(3000, lambda: self.orb.set_state("idle"))

        finally:
            self.is_executing = False
            self.after(0, lambda: self.input_bar.set_enabled(True))
            self.after(0, lambda: self.input_bar.focus())

    def _history_prev(self, event):
        """Navigate history up"""
        if self.command_history and self.history_index < len(self.command_history) - 1:
            self.history_index += 1
            self.input_bar.entry.delete(0, "end")
            self.input_bar.entry.insert(0, self.command_history[-(self.history_index + 1)])

    def _history_next(self, event):
        """Navigate history down"""
        if self.history_index > 0:
            self.history_index -= 1
            self.input_bar.entry.delete(0, "end")
            self.input_bar.entry.insert(0, self.command_history[-(self.history_index + 1)])
        elif self.history_index == 0:
            self.history_index = -1
            self.input_bar.entry.delete(0, "end")

    def _on_close(self):
        """Clean shutdown"""
        try:
            if self.core:
                self.core.close()
        except:
            pass
        self.destroy()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        app = RaecGUI()
        app.mainloop()
    except Exception as e:
        print(f"FATAL: {e}")
        traceback.print_exc()
