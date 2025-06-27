"""
Check progress of dream generation sessions.
"""

import json
import os
import glob
from datetime import datetime

def check_progress():
    """Check progress of all dream generation sessions."""
    logs_dir = 'logs'
    
    if not os.path.exists(logs_dir):
        print("No logs directory found.")
        return
    
    # Find all checkpoint files
    checkpoint_files = glob.glob(f"{logs_dir}/checkpoint_*.json")
    
    if not checkpoint_files:
        print("No checkpoint files found.")
        return
    
    print("=== DREAM GENERATION PROGRESS ===\n")
    
    for checkpoint_file in sorted(checkpoint_files):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            session_id = checkpoint_data.get('session_id', 'unknown')
            timestamp = checkpoint_data.get('timestamp', 'unknown')
            api_calls = checkpoint_data.get('api_calls_data', [])
            dreams = checkpoint_data.get('dreams_data', [])
            batch_tracker = checkpoint_data.get('batch_tracker', {})
            
            # Calculate statistics
            successful_dreams = [call for call in api_calls if call.get('status') == 'success']
            failed_calls = [call for call in api_calls if call.get('status') != 'success']
            
            # Group by language
            language_stats = {}
            for call in api_calls:
                language = call.get('language', 'unknown')
                if language not in language_stats:
                    language_stats[language] = {'successful': 0, 'failed': 0, 'total': 0}
                
                language_stats[language]['total'] += 1
                if call.get('status') == 'success':
                    language_stats[language]['successful'] += 1
                else:
                    language_stats[language]['failed'] += 1
            
            print(f"Session: {session_id}")
            print(f"Started: {timestamp}")
            print(f"Total API calls: {len(api_calls)}")
            print(f"Successful dreams: {len(successful_dreams)}")
            print(f"Failed calls: {len(failed_calls)}")
            print(f"Batches completed: {len(batch_tracker)}")
            
            if language_stats:
                print("Progress by language:")
                for lang, stats in language_stats.items():
                    progress_pct = (stats['successful'] / stats['total'] * 100) if stats['total'] > 0 else 0
                    print(f"  {lang}: {stats['successful']}/{stats['total']} ({progress_pct:.1f}%)")
            
            print("-" * 50)
            
        except Exception as e:
            print(f"Error reading {checkpoint_file}: {e}")
            print("-" * 50)

def check_session_progress(session_id):
    """Check progress for a specific session."""
    checkpoint_file = f"logs/checkpoint_{session_id}.json"
    
    if not os.path.exists(checkpoint_file):
        print(f"No checkpoint found for session {session_id}")
        return
    
    try:
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
        
        api_calls = checkpoint_data.get('api_calls_data', [])
        dreams = checkpoint_data.get('dreams_data', [])
        
        print(f"=== SESSION {session_id} PROGRESS ===\n")
        
        # Group by language
        language_stats = {}
        for call in api_calls:
            language = call.get('language', 'unknown')
            if language not in language_stats:
                language_stats[language] = {'successful': 0, 'failed': 0, 'total': 0}
            
            language_stats[language]['total'] += 1
            if call.get('status') == 'success':
                language_stats[language]['successful'] += 1
            else:
                language_stats[language]['failed'] += 1
        
        for lang, stats in language_stats.items():
            progress_pct = (stats['successful'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"{lang}: {stats['successful']}/{stats['total']} dreams ({progress_pct:.1f}%)")
            
            if stats['failed'] > 0:
                print(f"  Failed calls: {stats['failed']}")
        
    except Exception as e:
        print(f"Error reading checkpoint: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Check specific session
        session_id = sys.argv[1]
        check_session_progress(session_id)
    else:
        # Check all sessions
        check_progress() 