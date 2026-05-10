import subprocess
import webbrowser
import time
import sys
import os

def start():
    # URL for the local app
    url = "http://127.0.0.1:8000"
    
    print("Starting AutoClip Studio...")
    
    try:
        # Change to the 'app' directory so imports work correctly
        base_dir = os.path.dirname(os.path.abspath(__file__))
        app_dir = os.path.join(base_dir, "app")
        os.chdir(app_dir)
        
        # Use sys.executable to ensure we use the same python interpreter
        # Now we run 'main:app' because we are in the 'app' folder
        cmd = [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"]
        
        print(f"Opening web interface at {url}")
        
        # Use a simple timer to open the browser
        def open_browser():
            time.sleep(2)
            webbrowser.open(url)
        
        import threading
        threading.Thread(target=open_browser, daemon=True).start()
        
        # Start uvicorn
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\nAutoClip Studio stopped.")
    except Exception as e:
        print(f"Error starting server: {e}")

if __name__ == "__main__":
    start()
