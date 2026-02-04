import customtkinter as ctk
import threading
import traceback
import sys
import os
from datetime import datetime

# CRITICAL: Setup path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from main import Raec
except ImportError as e:
    print(f"BOOT ERROR: {e}")
    sys.exit(1)

ctk.set_appearance_mode("dark")

class RaecGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("RAEC | UNIFIED CONTROL INTERFACE")
        self.geometry("1000x750")
        self.configure(fg_color="#0A0A0A")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Output Log
        self.display = ctk.CTkTextbox(self, fg_color="#0D0D0D", text_color="#B0B0B0", font=("JetBrains Mono", 13))
        self.display.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        # Status Bar
        self.status_bar = ctk.CTkLabel(self, text="Initializing Core...", fg_color="#1A1A1A", text_color="#555555")
        self.status_bar.grid(row=2, column=0, sticky="ew")

        # Input Area
        self.input_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.input_frame.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="ew")

        self.entry = ctk.CTkEntry(self.input_frame, placeholder_text="Please wait for initialization...", height=40, state="disabled")
        self.entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.entry.bind("<Return>", lambda e: self.submit())

        self.btn = ctk.CTkButton(self.input_frame, text="EXECUTE", command=self.submit, state="disabled", fg_color="#1A1A1A")
        self.btn.pack(side="right")

        self.core = None
        self.update_log("SYSTEM >> Booting sequence initiated...")
        
        threading.Thread(target=self.init_core, daemon=True).start()

    def update_log(self, message):
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.display.insert("end", f"[{timestamp}] {message}\n")
            self.display.see("end")
        except:
            pass

    def init_core(self):
        try:
            self.core = Raec()
            self.status_bar.configure(text="CORE ONLINE", text_color="#00FF00")
            if self.entry.winfo_exists():
                self.entry.configure(state="normal", placeholder_text="Enter Instruction...")
                self.btn.configure(state="normal")
            self.update_log("SYSTEM >> Logic Core online. Direct Link Established.")
        except Exception as e:
            error_trace = traceback.format_exc()
            self.update_log(f"CRITICAL ERROR >> Core failed to load.\n{str(e)}")
            try:
                self.status_bar.configure(text="CORE OFFLINE", text_color="#FF0000")
            except: pass
            print(f"--- Initialization Traceback ---\n{error_trace}")

    def submit(self):
        cmd = self.entry.get()
        if not cmd: return
        self.update_log(f"USER >> {cmd}")
        self.entry.delete(0, "end")
        threading.Thread(target=self.execute, args=(cmd,), daemon=True).start()

    def execute(self, cmd):
        try:
            # DIRECT EXECUTION - NO MASK
            # We simply call the core and log the result.
            raw_logic = self.core.process_input(cmd)
            self.update_log(f"RAEC >> {raw_logic}")
        except Exception as e:
            self.update_log(f"EXECUTION ERROR: {str(e)}")

if __name__ == "__main__":
    try:
        app = RaecGUI()
        app.mainloop()
    except Exception as e:
        print(f"GUI CRASH: {e}")