#!/usr/bin/env python3
"""
Cleanup script to remove empty BATCH900 session directories
"""

import os
import pandas as pd

def cleanup_empty_batch_sessions():
    """Remove empty BATCH900 session directories"""
    
    print("🧹 CLEANING UP EMPTY BATCH900 SESSIONS")
    print("=" * 50)
    
    languages = ['english', 'basque', 'hebrew', 'serbian', 'slovenian']
    removed_count = 0
    
    for lang in languages:
        lang_dir = os.path.join('logs', lang, 'gpt-4o')
        if not os.path.exists(lang_dir):
            continue
            
        print(f"\n🔍 Checking {lang.title()}...")
        
        for item in os.listdir(lang_dir):
            if 'BATCH900_' in item and item.startswith('session_'):
                session_path = os.path.join(lang_dir, item)
                
                if os.path.isdir(session_path):
                    dreams_file = os.path.join(session_path, 'dreams.csv')
                    
                    # Check if dreams file exists and has content
                    should_remove = False
                    
                    if not os.path.exists(dreams_file):
                        print(f"  🗑️ No dreams.csv: {item}")
                        should_remove = True
                    else:
                        try:
                            df = pd.read_csv(dreams_file)
                            successful_dreams = len(df[df['status'] == 'success'])
                            
                            if successful_dreams == 0:
                                print(f"  🗑️ Empty dreams file: {item} (0 dreams)")
                                should_remove = True
                            else:
                                print(f"  ✅ Valid session: {item} ({successful_dreams} dreams)")
                                
                        except Exception as e:
                            print(f"  🗑️ Corrupted file: {item} (error: {e})")
                            should_remove = True
                    
                    if should_remove:
                        try:
                            # Remove the entire session directory
                            import shutil
                            shutil.rmtree(session_path)
                            print(f"    ✅ Removed: {session_path}")
                            removed_count += 1
                        except Exception as e:
                            print(f"    ❌ Failed to remove: {e}")
    
    print(f"\n🎉 Cleanup complete!")
    print(f"📊 Removed {removed_count} empty session directories")
    
    if removed_count > 0:
        print(f"\n💡 Now run the batch generation system again.")
        print(f"   It should correctly detect actual progress and not skip batches.")

if __name__ == "__main__":
    cleanup_empty_batch_sessions() 