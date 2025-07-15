#!/usr/bin/env python3
"""
Launch script for the optimized Streamlit Dream Analysis app
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("🌙 Starting Dream Analysis Dashboard - Optimized V2...")
    print("📂 Working with data from: logs_optimized_v2/")
    
    # Check if logs_optimized_v2 directory exists
    logs_dir = Path("logs_optimized_v2")
    if not logs_dir.exists():
        print(f"❌ Error: Directory {logs_dir} not found!")
        print("Please ensure the logs_optimized_v2 directory exists with dream data.")
        return
    
    # Check for dream data
    languages_found = []
    for lang_dir in logs_dir.iterdir():
        if lang_dir.is_dir() and not lang_dir.name.startswith('.'):
            gpt4o_dir = lang_dir / "gpt-4o"
            if gpt4o_dir.exists():
                for session_dir in gpt4o_dir.iterdir():
                    if session_dir.is_dir():
                        dreams_file = session_dir / "dreams.csv"
                        if dreams_file.exists():
                            languages_found.append(lang_dir.name)
                            break
    
    if not languages_found:
        print("❌ No dream data found in logs_optimized_v2!")
        print("Expected structure:")
        print("logs_optimized_v2/")
        print("├── english/gpt-4o/session_*/dreams.csv")
        print("├── basque/gpt-4o/session_*/dreams.csv")
        print("└── ...")
        return
    
    print(f"✅ Found dream data for: {', '.join(languages_found)}")
    print()
    print("🚀 Starting Streamlit app...")
    print("📱 The app will open in your default web browser")
    print("🔗 URL: http://localhost:8501")
    print()
    print("📝 Note: Close with Ctrl+C")
    print("=" * 50)
    
    try:
        # Run the streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_dream_analyzer_optimized.py"
        ])
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped by user")
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")

if __name__ == "__main__":
    main() 