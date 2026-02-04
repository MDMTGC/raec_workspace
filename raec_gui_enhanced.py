"""
RAEC Enhanced GUI with Real-Time Monitoring Dashboard

Features:
- Real-time performance metrics
- Task history
- System health monitoring
- Interactive controls
- Dark theme
"""
import customtkinter as ctk
import threading
import traceback
import sys
import os
import time
from datetime import datetime
from collections import deque

# CRITICAL: Setup path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from main_optimized import Raec
except ImportError:
    print("ERROR: Could not import optimized Raec. Falling back to standard version.")
    from main import Raec

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class MetricsPanel(ctk.CTkFrame):
    """Real-time metrics display panel"""
    
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        # Title
        title = ctk.CTkLabel(self, text="📊 SYSTEM METRICS", font=("JetBrains Mono", 16, "bold"))
        title.pack(pady=10)
        
        # Metrics display
        self.metrics_text = ctk.CTkTextbox(self, height=200, font=("JetBrains Mono", 11))
        self.metrics_text.pack(fill="both", expand=True, padx=10, pady=10)
        self.metrics_text.configure(state="disabled")
    
    def update_metrics(self, metrics):
        """Update metrics display"""
        self.metrics_text.configure(state="normal")
        self.metrics_text.delete("1.0", "end")
        
        # Format metrics
        lines = []
        lines.append(f"⚡ REAL-TIME PERFORMANCE\n")
        lines.append(f"{'='*40}\n\n")
        
        if 'realtime' in metrics:
            rt = metrics['realtime']
            lines.append(f"Uptime: {rt.get('uptime_seconds', 0):.1f}s\n")
            
            tasks = rt.get('tasks', {})
            lines.append(f"Tasks: {tasks.get('total', 0)} executed\n")
            lines.append(f"  ✓ Success: {tasks.get('successful', 0)}\n")
            lines.append(f"  ✗ Failed: {tasks.get('failed', 0)}\n")
            lines.append(f"  Success Rate: {tasks.get('success_rate', 0):.1%}\n\n")
            
            timing = rt.get('timing', {})
            lines.append(f"Avg Execution: {timing.get('avg_execution_time', 0):.2f}s\n\n")
            
            skills = rt.get('skills', {})
            lines.append(f"Skills:\n")
            lines.append(f"  Extracted: {skills.get('extracted', 0)}\n")
            lines.append(f"  Reused: {skills.get('reused', 0)}\n\n")
            
            tools = rt.get('tools', {})
            lines.append(f"Tools Used: {tools.get('used', 0)}\n")
        
        self.metrics_text.insert("1.0", "".join(lines))
        self.metrics_text.configure(state="disabled")


class TaskHistoryPanel(ctk.CTkFrame):
    """Task execution history panel"""
    
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        # Title
        title = ctk.CTkLabel(self, text="📜 TASK HISTORY", font=("JetBrains Mono", 16, "bold"))
        title.pack(pady=10)
        
        # History display
        self.history_text = ctk.CTkTextbox(self, height=200, font=("JetBrains Mono", 10))
        self.history_text.pack(fill="both", expand=True, padx=10, pady=10)
        self.history_text.configure(state="disabled")
        
        self.task_history = deque(maxlen=20)
    
    def add_task(self, task, mode, success, duration):
        """Add task to history"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        status = "✓" if success else "✗"
        
        self.task_history.append({
            'time': timestamp,
            'task': task[:40],
            'mode': mode,
            'status': status,
            'duration': duration
        })
        
        self._update_display()
    
    def _update_display(self):
        """Update history display"""
        self.history_text.configure(state="normal")
        self.history_text.delete("1.0", "end")
        
        lines = []
        for entry in reversed(self.task_history):
            lines.append(f"[{entry['time']}] {entry['status']} {entry['task']}\n")
            lines.append(f"  Mode: {entry['mode']} | Time: {entry['duration']:.2f}s\n\n")
        
        self.history_text.insert("1.0", "".join(lines))
        self.history_text.configure(state="disabled")


class RaecEnhancedGUI(ctk.CTk):
    """Enhanced Raec GUI with monitoring dashboard"""
    
    def __init__(self):
        super().__init__()
        
        self.title("RAEC | ENHANCED CONTROL INTERFACE")
        self.geometry("1400x900")
        
        # Configure grid
        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Left panel - Main interface
        self.left_panel = ctk.CTkFrame(self)
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        self.left_panel.grid_rowconfigure(1, weight=1)
        self.left_panel.grid_columnconfigure(0, weight=1)
        
        # Right panel - Metrics and monitoring
        self.right_panel = ctk.CTkFrame(self)
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        self.right_panel.grid_rowconfigure(0, weight=1)
        self.right_panel.grid_rowconfigure(1, weight=1)
        
        # === LEFT PANEL COMPONENTS ===
        
        # Header
        header = ctk.CTkLabel(
            self.left_panel,
            text="RAEC CONTROL INTERFACE",
            font=("JetBrains Mono", 20, "bold")
        )
        header.grid(row=0, column=0, pady=20)
        
        # Output display
        self.display = ctk.CTkTextbox(
            self.left_panel,
            font=("JetBrains Mono", 12),
            wrap="word"
        )
        self.display.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 10))
        
        # Control frame
        control_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        control_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 10))
        control_frame.grid_columnconfigure(0, weight=1)
        
        # Mode selector
        mode_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        mode_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        ctk.CTkLabel(mode_frame, text="Execution Mode:", font=("JetBrains Mono", 12)).pack(side="left", padx=(0, 10))
        
        self.mode_var = ctk.StringVar(value="standard")
        mode_options = ["standard", "collaborative", "incremental"]
        
        for mode in mode_options:
            ctk.CTkRadioButton(
                mode_frame,
                text=mode.capitalize(),
                variable=self.mode_var,
                value=mode,
                font=("JetBrains Mono", 11)
            ).pack(side="left", padx=5)
        
        # Input area
        input_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        input_frame.grid(row=1, column=0, sticky="ew")
        input_frame.grid_columnconfigure(0, weight=1)
        
        self.entry = ctk.CTkEntry(
            input_frame,
            placeholder_text="Initializing system...",
            height=40,
            state="disabled",
            font=("JetBrains Mono", 12)
        )
        self.entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.entry.bind("<Return>", lambda e: self.submit())
        
        self.btn = ctk.CTkButton(
            input_frame,
            text="EXECUTE",
            command=self.submit,
            state="disabled",
            height=40,
            font=("JetBrains Mono", 13, "bold")
        )
        self.btn.grid(row=0, column=1)
        
        # Status bar
        self.status_bar = ctk.CTkLabel(
            self.left_panel,
            text="⚙️  Initializing Core Systems...",
            font=("JetBrains Mono", 11),
            anchor="w"
        )
        self.status_bar.grid(row=3, column=0, sticky="ew", padx=20, pady=(0, 10))
        
        # === RIGHT PANEL COMPONENTS ===
        
        # Metrics panel
        self.metrics_panel = MetricsPanel(self.right_panel)
        self.metrics_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Task history panel
        self.history_panel = TaskHistoryPanel(self.right_panel)
        self.history_panel.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # === INITIALIZATION ===
        
        self.core = None
        self.update_log("SYSTEM >> Booting sequence initiated...")
        self.update_log("SYSTEM >> Enhanced monitoring enabled...")
        
        # Start core initialization in background
        threading.Thread(target=self.init_core, daemon=True).start()
        
        # Start metrics update loop
        self.after(2000, self.update_metrics_loop)
    
    def update_log(self, message):
        """Add message to log"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.display.insert("end", f"[{timestamp}] {message}\n")
            self.display.see("end")
        except:
            pass
    
    def init_core(self):
        """Initialize Raec core system"""
        try:
            self.update_log("SYSTEM >> Initializing Raec core...")
            self.core = Raec(enable_network=False)  # Network disabled by default
            
            self.status_bar.configure(text="✅ CORE ONLINE | Ready for Tasks")
            
            if self.entry.winfo_exists():
                self.entry.configure(state="normal", placeholder_text="Enter task...")
                self.btn.configure(state="normal")
            
            self.update_log("SYSTEM >> Core initialized successfully")
            self.update_log("SYSTEM >> Tool-enabled planner active")
            self.update_log("SYSTEM >> All subsystems operational")
            
        except Exception as e:
            error_trace = traceback.format_exc()
            self.update_log(f"CRITICAL ERROR >> Core initialization failed")
            self.update_log(f"ERROR >> {str(e)}")
            
            try:
                self.status_bar.configure(text="❌ CORE OFFLINE | Initialization Failed")
            except:
                pass
            
            print(f"--- Initialization Traceback ---\n{error_trace}")
    
    def submit(self):
        """Submit task for execution"""
        task = self.entry.get().strip()
        if not task or not self.core:
            return
        
        mode = self.mode_var.get()
        
        self.update_log(f"USER >> {task}")
        self.update_log(f"SYSTEM >> Executing in {mode} mode...")
        self.entry.delete(0, "end")
        
        # Execute in background thread
        threading.Thread(
            target=self.execute_task,
            args=(task, mode),
            daemon=True
        ).start()
    
    def execute_task(self, task, mode):
        """Execute task and update UI"""
        start_time = time.time()
        
        try:
            # Execute via core
            result = self.core.process_input(task, mode=mode)
            duration = time.time() - start_time
            
            # Determine success
            success = "error" not in result.lower()
            
            # Update log
            self.update_log(f"RAEC >> {result}")
            self.update_log(f"SYSTEM >> Execution complete ({duration:.2f}s)")
            
            # Update history
            self.history_panel.add_task(task, mode, success, duration)
            
        except Exception as e:
            duration = time.time() - start_time
            self.update_log(f"ERROR >> Execution failed: {str(e)}")
            self.history_panel.add_task(task, mode, False, duration)
    
    def update_metrics_loop(self):
        """Periodically update metrics display"""
        if self.core and hasattr(self.core, 'get_performance_metrics'):
            try:
                metrics = self.core.analyze_performance()
                self.metrics_panel.update_metrics(metrics)
            except:
                pass
        
        # Schedule next update
        self.after(2000, self.update_metrics_loop)


def main():
    """Launch enhanced GUI"""
    try:
        app = RaecEnhancedGUI()
        app.mainloop()
    except Exception as e:
        print(f"GUI CRASH: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
