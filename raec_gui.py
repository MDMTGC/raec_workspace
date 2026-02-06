"""
RAEC GUI - Unified Control Interface
Enhanced version with stats, mode selection, and improved UX
"""
import customtkinter as ctk
import threading
import traceback
import sys
import os
from datetime import datetime
from typing import Optional

# Path setup for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from main import Raec
except ImportError as e:
    print(f"BOOT ERROR: {e}")
    sys.exit(1)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


class StatsPanel(ctk.CTkFrame):
    """Panel displaying system statistics"""

    def __init__(self, parent):
        super().__init__(parent, fg_color="#111111", corner_radius=8)

        self.title = ctk.CTkLabel(
            self, text="SYSTEM STATS",
            font=("JetBrains Mono", 11, "bold"),
            text_color="#666666"
        )
        self.title.pack(pady=(10, 5))

        # Stats labels
        self.stats_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.stats_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.stat_labels = {}
        stats = [
            ("memory", "Memory", "0"),
            ("skills", "Skills", "0"),
            ("tools", "Tools", "0"),
            ("agents", "Agents", "0"),
            ("verifications", "Verified", "0"),
            ("swarm", "Swarm", "-"),
        ]

        for key, label, default in stats:
            frame = ctk.CTkFrame(self.stats_frame, fg_color="transparent")
            frame.pack(fill="x", pady=2)

            name_label = ctk.CTkLabel(
                frame, text=f"{label}:",
                font=("JetBrains Mono", 10),
                text_color="#555555",
                width=80, anchor="w"
            )
            name_label.pack(side="left")

            value_label = ctk.CTkLabel(
                frame, text=default,
                font=("JetBrains Mono", 10, "bold"),
                text_color="#888888"
            )
            value_label.pack(side="right")

            self.stat_labels[key] = value_label

    def update_stats(self, stats: dict):
        """Update stats display from analyze_performance() output"""
        try:
            if 'memory' in stats:
                mem = stats['memory']
                total = sum(mem.values())
                self.stat_labels['memory'].configure(text=str(total))

            if 'skills' in stats:
                self.stat_labels['skills'].configure(
                    text=str(stats['skills'].get('total_skills', 0))
                )

            if 'tools' in stats:
                self.stat_labels['tools'].configure(
                    text=str(stats['tools'].get('total_tools', 0))
                )

            if 'agents' in stats:
                self.stat_labels['agents'].configure(
                    text=str(stats['agents'].get('total_agents', 0))
                )

            if 'verification' in stats:
                total = stats['verification'].get('total_verifications', 0)
                self.stat_labels['verifications'].configure(text=str(total))

            if 'swarm' in stats and stats['swarm']:
                models = len(stats['swarm'].get('available_models', []))
                self.stat_labels['swarm'].configure(text=f"{models} models")
        except Exception as e:
            print(f"Stats update error: {e}")


class RaecGUI(ctk.CTk):
    """Main RAEC GUI Application"""

    def __init__(self):
        super().__init__()

        self.title("RAEC | Autonomous Reasoning & Execution Core")
        self.geometry("1200x800")
        self.minsize(900, 600)
        self.configure(fg_color="#0A0A0A")

        # State
        self.core: Optional[Raec] = None
        self.is_executing = False
        self.command_history = []
        self.history_index = -1

        self._setup_ui()
        self._bind_events()

        # Start core initialization
        self.update_log("SYSTEM >> Booting sequence initiated...")
        threading.Thread(target=self._init_core, daemon=True).start()

    def _setup_ui(self):
        """Setup the UI layout"""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(0, weight=1)

        # Main content area (left)
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=(20, 10), pady=20)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        # Output display
        self.display = ctk.CTkTextbox(
            self.main_frame,
            fg_color="#0D0D0D",
            text_color="#B0B0B0",
            font=("JetBrains Mono", 12),
            corner_radius=8,
            border_width=1,
            border_color="#1A1A1A"
        )
        self.display.grid(row=0, column=0, sticky="nsew", pady=(0, 10))

        # Input area
        self.input_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.input_frame.grid(row=1, column=0, sticky="ew")
        self.input_frame.grid_columnconfigure(1, weight=1)

        # Mode selector
        self.mode_var = ctk.StringVar(value="standard")
        self.mode_menu = ctk.CTkOptionMenu(
            self.input_frame,
            values=["standard", "collaborative", "incremental"],
            variable=self.mode_var,
            width=130,
            height=40,
            fg_color="#1A1A1A",
            button_color="#252525",
            button_hover_color="#303030",
            dropdown_fg_color="#1A1A1A",
            font=("JetBrains Mono", 11)
        )
        self.mode_menu.grid(row=0, column=0, padx=(0, 10))

        # Input entry
        self.entry = ctk.CTkEntry(
            self.input_frame,
            placeholder_text="Initializing...",
            height=40,
            state="disabled",
            font=("JetBrains Mono", 12),
            fg_color="#0D0D0D",
            border_color="#1A1A1A"
        )
        self.entry.grid(row=0, column=1, sticky="ew", padx=(0, 10))

        # Execute button
        self.exec_btn = ctk.CTkButton(
            self.input_frame,
            text="EXECUTE",
            command=self._submit,
            state="disabled",
            width=100,
            height=40,
            fg_color="#1A1A1A",
            hover_color="#252525",
            font=("JetBrains Mono", 11, "bold")
        )
        self.exec_btn.grid(row=0, column=2)

        # Right sidebar
        self.sidebar = ctk.CTkFrame(self, fg_color="#0D0D0D", width=200, corner_radius=8)
        self.sidebar.grid(row=0, column=1, sticky="ns", padx=(0, 20), pady=20)
        self.sidebar.grid_propagate(False)

        # Stats panel
        self.stats_panel = StatsPanel(self.sidebar)
        self.stats_panel.pack(fill="x", padx=10, pady=10)

        # Action buttons
        self.btn_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.btn_frame.pack(fill="x", padx=10, pady=10)

        self.refresh_btn = ctk.CTkButton(
            self.btn_frame,
            text="Refresh Stats",
            command=self._refresh_stats,
            state="disabled",
            fg_color="#1A1A1A",
            hover_color="#252525",
            font=("JetBrains Mono", 10)
        )
        self.refresh_btn.pack(fill="x", pady=2)

        self.clear_btn = ctk.CTkButton(
            self.btn_frame,
            text="Clear Log",
            command=self._clear_log,
            fg_color="#1A1A1A",
            hover_color="#252525",
            font=("JetBrains Mono", 10)
        )
        self.clear_btn.pack(fill="x", pady=2)

        # Status bar
        self.status_frame = ctk.CTkFrame(self, fg_color="#111111", height=30, corner_radius=0)
        self.status_frame.grid(row=1, column=0, columnspan=2, sticky="ew")

        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Initializing Core...",
            font=("JetBrains Mono", 10),
            text_color="#555555"
        )
        self.status_label.pack(side="left", padx=20, pady=5)

        self.mode_label = ctk.CTkLabel(
            self.status_frame,
            text="Mode: standard",
            font=("JetBrains Mono", 10),
            text_color="#444444"
        )
        self.mode_label.pack(side="right", padx=20, pady=5)

    def _bind_events(self):
        """Bind keyboard and window events"""
        self.entry.bind("<Return>", lambda e: self._submit())
        self.entry.bind("<Up>", self._history_prev)
        self.entry.bind("<Down>", self._history_next)
        self.mode_var.trace_add("write", self._on_mode_change)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_mode_change(self, *args):
        """Update mode label when mode changes"""
        mode = self.mode_var.get()
        self.mode_label.configure(text=f"Mode: {mode}")

    def _history_prev(self, event):
        """Navigate to previous command in history"""
        if self.command_history and self.history_index < len(self.command_history) - 1:
            self.history_index += 1
            self.entry.delete(0, "end")
            self.entry.insert(0, self.command_history[-(self.history_index + 1)])

    def _history_next(self, event):
        """Navigate to next command in history"""
        if self.history_index > 0:
            self.history_index -= 1
            self.entry.delete(0, "end")
            self.entry.insert(0, self.command_history[-(self.history_index + 1)])
        elif self.history_index == 0:
            self.history_index = -1
            self.entry.delete(0, "end")

    def update_log(self, message: str, tag: str = "normal"):
        """Add timestamped message to the log"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.display.insert("end", f"[{timestamp}] {message}\n")
            self.display.see("end")
        except Exception as e:
            print(f"Log error: {e}")

    def _set_status(self, text: str, color: str = "#555555"):
        """Update status bar"""
        try:
            self.status_label.configure(text=text, text_color=color)
        except:
            pass

    def _set_executing(self, executing: bool):
        """Update UI state during execution"""
        self.is_executing = executing
        state = "disabled" if executing else "normal"

        try:
            self.entry.configure(state=state)
            self.exec_btn.configure(
                state=state,
                text="..." if executing else "EXECUTE"
            )
            self.mode_menu.configure(state=state)
        except:
            pass

    def _init_core(self):
        """Initialize the Raec core in background"""
        try:
            self._set_status("Loading core components...", "#888888")
            self.core = Raec()

            # Enable UI
            self.after(0, lambda: self.entry.configure(
                state="normal",
                placeholder_text="Enter task or command..."
            ))
            self.after(0, lambda: self.exec_btn.configure(state="normal"))
            self.after(0, lambda: self.refresh_btn.configure(state="normal"))

            self._set_status("CORE ONLINE", "#00AA00")
            self.update_log("SYSTEM >> Core initialized. All subsystems operational.")

            # Initial stats refresh
            self._refresh_stats()

        except Exception as e:
            error_trace = traceback.format_exc()
            self.update_log(f"CRITICAL >> Core initialization failed: {e}")
            self._set_status("CORE OFFLINE", "#FF0000")
            print(f"--- Init Error ---\n{error_trace}")

    def _submit(self):
        """Handle command submission"""
        if self.is_executing or not self.core:
            return

        cmd = self.entry.get().strip()
        if not cmd:
            return

        # Add to history
        self.command_history.append(cmd)
        self.history_index = -1

        # Clear input
        self.entry.delete(0, "end")

        # Log command
        mode = self.mode_var.get()
        self.update_log(f"USER [{mode}] >> {cmd}")

        # Execute in background
        threading.Thread(
            target=self._execute,
            args=(cmd, mode),
            daemon=True
        ).start()

    def _execute(self, cmd: str, mode: str):
        """Execute command in background thread"""
        self._set_executing(True)
        self._set_status(f"Executing ({mode})...", "#FFAA00")

        try:
            result = self.core.process_input(cmd, mode=mode)
            self.update_log(f"RAEC >> {result}")
            self._set_status("Ready", "#00AA00")

            # Auto-refresh stats after execution
            self._refresh_stats()

        except Exception as e:
            error_msg = str(e)
            self.update_log(f"ERROR >> {error_msg}")
            self._set_status("Execution failed", "#FF5555")
            print(f"Execution error: {traceback.format_exc()}")

        finally:
            self._set_executing(False)

    def _refresh_stats(self):
        """Refresh the stats panel"""
        if not self.core:
            return

        try:
            # Capture stdout to prevent console spam
            import io
            from contextlib import redirect_stdout

            f = io.StringIO()
            with redirect_stdout(f):
                stats = self.core.analyze_performance()

            self.stats_panel.update_stats(stats)
        except Exception as e:
            print(f"Stats refresh error: {e}")

    def _clear_log(self):
        """Clear the output log"""
        try:
            self.display.delete("1.0", "end")
            self.update_log("SYSTEM >> Log cleared")
        except:
            pass

    def _on_close(self):
        """Handle window close - clean shutdown"""
        try:
            if self.core:
                self.update_log("SYSTEM >> Shutting down...")
                self.core.close()
        except Exception as e:
            print(f"Shutdown error: {e}")
        finally:
            self.destroy()


if __name__ == "__main__":
    try:
        app = RaecGUI()
        app.mainloop()
    except Exception as e:
        print(f"GUI CRASH: {e}")
        traceback.print_exc()
