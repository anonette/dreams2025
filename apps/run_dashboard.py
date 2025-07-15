#!/usr/bin/env python3
"""
Dream Analysis Dashboard Launcher
Simple script to launch the Streamlit dashboard
"""

import subprocess
import sys
import os

def main():
    print("🌙 Starting Dream Analysis Dashboard...")
    print("=" * 50)
    
    # Ensure we're in the right directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_dream_analyzer.py",
            "--server.port=8501",
            "--server.headless=false",
            "--server.address=localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")

if __name__ == "__main__":
    main() 