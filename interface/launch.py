"""Launch script for the Streamlit interface."""

import os
import subprocess
import sys

def main():
    """Launch the Streamlit app."""
    print("Launching Prompt Optimizer Streamlit interface...")
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, "app.py")
    
    # Launch Streamlit
    cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
