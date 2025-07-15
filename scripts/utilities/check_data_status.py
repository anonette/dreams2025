#!/usr/bin/env python3
"""
Quick Data Status Checker
Shows what logs and translations are available
"""

import os
from pathlib import Path
import pandas as pd
import json

def check_data_status():
    print("🔍 Dream Data Status Report")
    print("=" * 50)
    
    # Check available log directories
    print("\n📁 Available Log Directories:")
    for dir_name in ['logs', 'logs_optimized_v2']:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"  ✅ {dir_name}/")
            
            # Check languages in this directory
            languages = []
            for lang_dir in dir_path.iterdir():
                if lang_dir.is_dir() and not lang_dir.name.startswith('.'):
                    gpt4o_dir = lang_dir / "gpt-4o"
                    if gpt4o_dir.exists():
                        session_count = len([d for d in gpt4o_dir.iterdir() if d.is_dir()])
                        if session_count > 0:
                            languages.append(f"{lang_dir.name} ({session_count} sessions)")
            
            if languages:
                print(f"    Languages: {', '.join(languages)}")
            else:
                print(f"    No language data found")
        else:
            print(f"  ❌ {dir_name}/ (not found)")
    
    # Check translations directory
    print("\n📝 Translations Directory:")
    translations_dir = Path("translations")
    if translations_dir.exists():
        print(f"  ✅ translations/")
        
        # Count translation files
        json_files = list(translations_dir.glob("*.json"))
        csv_files = list(translations_dir.glob("*.csv"))
        
        print(f"    JSON files: {len(json_files)}")
        print(f"    CSV files: {len(csv_files)}")
        
        # Group by session
        sessions = {}
        for file in json_files:
            if "_translations_" in file.name:
                parts = file.name.split("_translations_")
                if len(parts) == 2:
                    lang = parts[0]
                    session = parts[1].replace(".json", "")
                    if session not in sessions:
                        sessions[session] = []
                    sessions[session].append(lang)
        
        print(f"\n    Translation Sessions:")
        for session, languages in sessions.items():
            print(f"      {session}: {', '.join(languages)}")
    else:
        print(f"  ❌ translations/ (not found)")
    
    # Find latest session across all directories
    print("\n🕐 Latest Sessions:")
    for dir_name in ['logs', 'logs_optimized_v2']:
        dir_path = Path(dir_name)
        if dir_path.exists():
            all_sessions = []
            for lang_dir in dir_path.iterdir():
                if lang_dir.is_dir():
                    gpt4o_dir = lang_dir / "gpt-4o"
                    if gpt4o_dir.exists():
                        for session_dir in gpt4o_dir.iterdir():
                            if session_dir.is_dir() and session_dir.name.startswith('session_'):
                                all_sessions.append(session_dir.name)
            
            unique_sessions = sorted(list(set(all_sessions)))
            if unique_sessions:
                latest = unique_sessions[-1]
                print(f"  {dir_name}: {latest}")
                
                # Check if translations exist for this session
                if translations_dir.exists():
                    trans_files = list(translations_dir.glob(f"*_translations_{latest}.json"))
                    if trans_files:
                        print(f"    ✅ Translations available: {len(trans_files)} languages")
                    else:
                        print(f"    ⚠️ No translations for this session")
    
    # Recommendations
    print("\n💡 Recommendations:")
    
    # Check if logs_optimized_v2 has more recent data
    logs_latest = None
    opt_latest = None
    
    for dir_name, var_name in [('logs', 'logs_latest'), ('logs_optimized_v2', 'opt_latest')]:
        dir_path = Path(dir_name)
        if dir_path.exists():
            sessions = []
            for lang_dir in dir_path.iterdir():
                if lang_dir.is_dir():
                    gpt4o_dir = lang_dir / "gpt-4o"
                    if gpt4o_dir.exists():
                        for session_dir in gpt4o_dir.iterdir():
                            if session_dir.is_dir():
                                sessions.append(session_dir.name)
            if sessions:
                latest = sorted(list(set(sessions)))[-1]
                if var_name == 'logs_latest':
                    logs_latest = latest
                else:
                    opt_latest = latest
    
    if opt_latest and logs_latest:
        if opt_latest > logs_latest:
            print(f"  🎯 Use logs_optimized_v2 - has newer data ({opt_latest})")
        elif logs_latest > opt_latest:
            print(f"  🎯 Use logs - has newer data ({logs_latest})")
        else:
            print(f"  🎯 Both directories have same latest session ({opt_latest})")
    elif opt_latest:
        print(f"  🎯 Use logs_optimized_v2 - only directory with data")
    elif logs_latest:
        print(f"  🎯 Use logs - only directory with data")
    
    # Check for missing translations
    for dir_name in ['logs', 'logs_optimized_v2']:
        dir_path = Path(dir_name)
        if dir_path.exists():
            sessions = []
            for lang_dir in dir_path.iterdir():
                if lang_dir.is_dir():
                    gpt4o_dir = lang_dir / "gpt-4o"
                    if gpt4o_dir.exists():
                        for session_dir in gpt4o_dir.iterdir():
                            if session_dir.is_dir():
                                sessions.append(session_dir.name)
            
            if sessions:
                latest = sorted(list(set(sessions)))[-1]
                
                # Count languages with dreams
                lang_count = 0
                for lang_dir in dir_path.iterdir():
                    if lang_dir.is_dir() and not lang_dir.name.startswith('.'):
                        dreams_file = lang_dir / "gpt-4o" / latest / "dreams.csv"
                        if dreams_file.exists():
                            lang_count += 1
                
                # Count translations
                trans_count = 0
                if translations_dir.exists():
                    trans_files = list(translations_dir.glob(f"*_translations_{latest}.json"))
                    trans_count = len(trans_files)
                
                if trans_count < lang_count:
                    print(f"  ⚠️ {dir_name}: Missing translations ({trans_count}/{lang_count} languages translated)")

if __name__ == "__main__":
    check_data_status() 