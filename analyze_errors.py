"""
Analyze error logs and rejected dreams for debugging and quality improvement.
"""

import json
import os
import glob
from collections import Counter, defaultdict
from datetime import datetime

def analyze_session_errors(session_id):
    """Analyze errors and rejections for a specific session."""
    logs_dir = 'logs'
    
    error_log_file = f"{logs_dir}/error_log_{session_id}.jsonl"
    rejected_dreams_file = f"{logs_dir}/rejected_dreams_{session_id}.jsonl"
    
    print(f"=== ERROR ANALYSIS FOR SESSION {session_id} ===\n")
    
    # Analyze API errors
    if os.path.exists(error_log_file):
        errors = []
        with open(error_log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    errors.append(json.loads(line))
        
        print(f"API ERRORS: {len(errors)} total\n")
        
        if errors:
            # Error types
            error_types = Counter([error.get('error_type', 'unknown') for error in errors])
            print("Error Types:")
            for error_type, count in error_types.most_common():
                print(f"  {error_type}: {count}")
            
            # Errors by language
            language_errors = Counter([error.get('language', 'unknown') for error in errors])
            print("\nErrors by Language:")
            for language, count in language_errors.most_common():
                print(f"  {language}: {count}")
            
            # Sample error messages
            print("\nSample Error Messages:")
            for i, error in enumerate(errors[:5]):
                print(f"  {i+1}. {error.get('error_type', 'unknown')}: {error.get('error_message', 'no message')[:100]}...")
            
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more errors")
    else:
        print("No API error log found.")
    
    print("\n" + "="*50 + "\n")
    
    # Analyze rejected dreams
    if os.path.exists(rejected_dreams_file):
        rejected_dreams = []
        with open(rejected_dreams_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    rejected_dreams.append(json.loads(line))
        
        print(f"REJECTED DREAMS: {len(rejected_dreams)} total\n")
        
        if rejected_dreams:
            # Rejections by language
            language_rejections = Counter([dream.get('language', 'unknown') for dream in rejected_dreams])
            print("Rejections by Language:")
            for language, count in language_rejections.most_common():
                print(f"  {language}: {count}")
            
            # Rejection reasons
            rejection_reasons = Counter([dream.get('rejection_reason', 'unknown') for dream in rejected_dreams])
            print("\nRejection Reasons:")
            for reason, count in rejection_reasons.most_common():
                print(f"  {reason}: {count}")
            
            # Sample rejected dreams
            print("\nSample Rejected Dreams:")
            for i, dream in enumerate(rejected_dreams[:3]):
                original = dream.get('original_dream', 'no content')
                print(f"  {i+1}. {dream.get('language', 'unknown')} - {original[:100]}...")
                print(f"     Reason: {dream.get('rejection_reason', 'unknown')}")
            
            if len(rejected_dreams) > 3:
                print(f"  ... and {len(rejected_dreams) - 3} more rejected dreams")
    else:
        print("No rejected dreams log found.")

def analyze_all_sessions():
    """Analyze errors across all sessions."""
    logs_dir = 'logs'
    
    if not os.path.exists(logs_dir):
        print("No logs directory found.")
        return
    
    # Find all error and rejection log files
    error_files = glob.glob(f"{logs_dir}/error_log_*.jsonl")
    rejection_files = glob.glob(f"{logs_dir}/rejected_dreams_*.jsonl")
    
    print("=== CROSS-SESSION ERROR ANALYSIS ===\n")
    
    # Aggregate all errors
    all_errors = []
    for error_file in error_files:
        with open(error_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    all_errors.append(json.loads(line))
    
    # Aggregate all rejections
    all_rejections = []
    for rejection_file in rejection_files:
        with open(rejection_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    all_rejections.append(json.loads(line))
    
    print(f"TOTAL ERRORS ACROSS ALL SESSIONS: {len(all_errors)}")
    print(f"TOTAL REJECTIONS ACROSS ALL SESSIONS: {len(all_rejections)}\n")
    
    if all_errors:
        # Overall error types
        error_types = Counter([error.get('error_type', 'unknown') for error in all_errors])
        print("Most Common Error Types:")
        for error_type, count in error_types.most_common(5):
            print(f"  {error_type}: {count}")
        
        # Errors by language
        language_errors = Counter([error.get('language', 'unknown') for error in all_errors])
        print("\nErrors by Language:")
        for language, count in language_errors.most_common():
            print(f"  {language}: {count}")
    
    if all_rejections:
        # Rejections by language
        language_rejections = Counter([dream.get('language', 'unknown') for dream in all_rejections])
        print("\nRejections by Language:")
        for language, count in language_rejections.most_common():
            print(f"  {language}: {count}")
        
        # Rejection reasons
        rejection_reasons = Counter([dream.get('rejection_reason', 'unknown') for dream in all_rejections])
        print("\nRejection Reasons:")
        for reason, count in rejection_reasons.most_common():
            print(f"  {reason}: {count}")

def export_error_summary(session_id=None):
    """Export error summary to CSV for further analysis."""
    import pandas as pd
    
    logs_dir = 'logs'
    
    if session_id:
        error_file = f"{logs_dir}/error_log_{session_id}.jsonl"
        rejection_file = f"{logs_dir}/rejected_dreams_{session_id}.jsonl"
        output_prefix = f"error_analysis_{session_id}"
    else:
        # Aggregate all sessions
        error_files = glob.glob(f"{logs_dir}/error_log_*.jsonl")
        rejection_files = glob.glob(f"{logs_dir}/rejected_dreams_*.jsonl")
        output_prefix = "error_analysis_all_sessions"
        
        # Combine all error files
        all_errors = []
        for error_file in error_files:
            with open(error_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        all_errors.append(json.loads(line))
        
        # Combine all rejection files
        all_rejections = []
        for rejection_file in rejection_files:
            with open(rejection_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        all_rejections.append(json.loads(line))
        
        # Save combined data
        if all_errors:
            df_errors = pd.DataFrame(all_errors)
            df_errors.to_csv(f"{logs_dir}/{output_prefix}_errors.csv", index=False, encoding='utf-8')
            print(f"Exported {len(all_errors)} errors to {logs_dir}/{output_prefix}_errors.csv")
        
        if all_rejections:
            df_rejections = pd.DataFrame(all_rejections)
            df_rejections.to_csv(f"{logs_dir}/{output_prefix}_rejections.csv", index=False, encoding='utf-8')
            print(f"Exported {len(all_rejections)} rejections to {logs_dir}/{output_prefix}_rejections.csv")
        
        return
    
    # Single session export
    if os.path.exists(error_file):
        errors = []
        with open(error_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    errors.append(json.loads(line))
        
        if errors:
            df_errors = pd.DataFrame(errors)
            df_errors.to_csv(f"{logs_dir}/{output_prefix}_errors.csv", index=False, encoding='utf-8')
            print(f"Exported {len(errors)} errors to {logs_dir}/{output_prefix}_errors.csv")
    
    if os.path.exists(rejection_file):
        rejections = []
        with open(rejection_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    rejections.append(json.loads(line))
        
        if rejections:
            df_rejections = pd.DataFrame(rejections)
            df_rejections.to_csv(f"{logs_dir}/{output_prefix}_rejections.csv", index=False, encoding='utf-8')
            print(f"Exported {len(rejections)} rejections to {logs_dir}/{output_prefix}_rejections.csv")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            analyze_all_sessions()
        elif sys.argv[1] == "--export":
            session_id = sys.argv[2] if len(sys.argv) > 2 else None
            export_error_summary(session_id)
        else:
            session_id = sys.argv[1]
            analyze_session_errors(session_id)
    else:
        print("Usage:")
        print("  python analyze_errors.py <session_id>     - Analyze specific session")
        print("  python analyze_errors.py --all            - Analyze all sessions")
        print("  python analyze_errors.py --export [session_id] - Export to CSV") 