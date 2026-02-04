import os
import subprocess
import sys

def fix_setup():
    # The correct official tag for "32B Qwen Distill"
    CORRECT_MODEL = "deepseek-r1:32b"
    
    print(f"--- RAEC 32B SETUP ({CORRECT_MODEL}) ---")
    print(f"1. Pulling {CORRECT_MODEL} from library...")
    print("   (This is a 20GB download. Please wait...)")
    
    try:
        # PULL THE MODEL
        # We use check=True to make sure it actually finishes downloading
        subprocess.run(["ollama", "pull", CORRECT_MODEL], check=True, shell=True)
        print("\n✅ Download Complete.")
        
    except subprocess.CalledProcessError:
        print("\n❌ DOWNLOAD FAILED.")
        print("Check your internet connection or disk space.")
        input("Press Enter to exit...")
        return

    # BUILD RAEC
    print(f"\n2. Rebuilding 'raec' using {CORRECT_MODEL}...")
    
    modelfile = f"""FROM {CORRECT_MODEL}

SYSTEM "You are Raec, an intelligent technical assistant. You specialize in Python execution, system architecture, and logical planning. Your responses are concise, accurate, and professional."

PARAMETER temperature 0.1
PARAMETER num_ctx 8192
"""

    try:
        with open("Modelfile", "wb") as f:
            f.write(modelfile.encode('utf-8'))
            
        subprocess.run(["ollama", "rm", "raec"], shell=True, stderr=subprocess.DEVNULL)
        subprocess.run(["ollama", "create", "raec", "-f", "Modelfile"], check=True, shell=True)
        
        print("\n✅ SUCCESS: Raec is now running on 32B.")
        
    except Exception as e:
        print(f"\n❌ BUILD ERROR: {e}")
    finally:
        if os.path.exists("Modelfile"):
            os.remove("Modelfile")
        
    print("\n" + "="*40)
    input("PRESS ENTER TO FINISH...")

if __name__ == "__main__":
    fix_setup()