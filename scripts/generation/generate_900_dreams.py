#!/usr/bin/env python3
"""
Generate 900 dreams per language using 9 sessions of 100 dreams each.
Provides robust generation with automatic recovery and progress tracking.
"""

import asyncio
import os
import time
import json
import subprocess
from datetime import datetime
from pathlib import Path
import sys

class DreamGenerationManager:
    """Manages the generation of 900 dreams per language in 9 sessions."""
    
    def __init__(self):
        self.languages = ['english', 'basque', 'serbian', 'hebrew', 'slovenian']
        self.dreams_per_session = 100
        self.sessions_per_language = 9
        self.total_dreams_per_language = 900
        self.batch_size = 50  # 100 dreams = 2 batches of 50
        self.session_delay = 300  # 5 minutes between sessions
        self.logs_dir = Path('logs')
        self.progress_file = 'generation_progress_900.json'
        
    def load_progress(self):
        """Load generation progress from file."""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_progress(self, progress):
        """Save generation progress to file."""
        progress['last_updated'] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def count_existing_dreams(self, language):
        """Count existing dreams for a language across all sessions."""
        lang_dir = self.logs_dir / language / 'gpt-4o'
        total_dreams = 0
        
        if not lang_dir.exists():
            return 0
            
        for session_dir in lang_dir.iterdir():
            if session_dir.is_dir() and session_dir.name.startswith('session_'):
                dreams_file = session_dir / 'dreams.csv'
                if dreams_file.exists():
                    try:
                        import pandas as pd
                        df = pd.read_csv(dreams_file)
                        successful_dreams = len(df[df['status'] == 'success'])
                        total_dreams += successful_dreams
                    except Exception as e:
                        print(f"Error reading {dreams_file}: {e}")
        
        return total_dreams
    
    async def generate_session(self, language, session_num, total_sessions):
        """Generate one session of 100 dreams for a language."""
        print(f"\n🎯 Starting Session {session_num}/{total_sessions} for {language.upper()}")
        print(f"   Target: 100 dreams (2 batches of 50)")
        print(f"   Batch size: {self.batch_size}")
        
        start_time = time.time()
        
        # Run the batch generator
        cmd = [
            sys.executable, 'batch_dream_generator.py',
            '--language', language,
            '--dreams-per-language', str(self.dreams_per_session),
            '--batch-size', str(self.batch_size),
            '--force-restart'  # Each session is independent
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            duration = time.time() - start_time
            
            if process.returncode == 0:
                print(f"✅ Session {session_num} completed successfully in {duration/60:.1f} minutes")
                return True, stdout.decode(), stderr.decode()
            else:
                print(f"❌ Session {session_num} failed with return code {process.returncode}")
                print(f"Error: {stderr.decode()}")
                return False, stdout.decode(), stderr.decode()
                
        except Exception as e:
            print(f"❌ Session {session_num} failed with exception: {e}")
            return False, "", str(e)
    
    async def generate_language(self, language):
        """Generate 900 dreams for a single language."""
        print(f"\n🌍 STARTING {language.upper()} - Target: 900 dreams in 9 sessions")
        
        progress = self.load_progress()
        if language not in progress:
            progress[language] = {
                'completed_sessions': 0,
                'total_dreams': 0,
                'failed_sessions': [],
                'session_results': []
            }
        
        # Check existing dreams
        existing_dreams = self.count_existing_dreams(language)
        progress[language]['total_dreams'] = existing_dreams
        
        if existing_dreams >= self.total_dreams_per_language:
            print(f"✅ {language} already has {existing_dreams} dreams (target: {self.total_dreams_per_language})")
            return True
        
        print(f"📊 Current progress: {existing_dreams}/{self.total_dreams_per_language} dreams")
        
        completed_sessions = progress[language]['completed_sessions']
        
        for session_num in range(completed_sessions + 1, self.sessions_per_language + 1):
            success, stdout, stderr = await self.generate_session(language, session_num, self.sessions_per_language)
            
            # Record session result
            session_result = {
                'session_num': session_num,
                'success': success,
                'timestamp': datetime.now().isoformat(),
                'stdout_length': len(stdout),
                'stderr_length': len(stderr)
            }
            progress[language]['session_results'].append(session_result)
            
            if success:
                progress[language]['completed_sessions'] = session_num
                # Recount dreams to get accurate total
                progress[language]['total_dreams'] = self.count_existing_dreams(language)
                print(f"📈 Total dreams for {language}: {progress[language]['total_dreams']}")
            else:
                progress[language]['failed_sessions'].append(session_num)
                print(f"⚠️  Session {session_num} failed, will retry later")
            
            # Save progress after each session
            self.save_progress(progress)
            
            # Delay between sessions (except after the last one)
            if session_num < self.sessions_per_language:
                print(f"⏱️  Waiting {self.session_delay/60:.1f} minutes before next session...")
                await asyncio.sleep(self.session_delay)
        
        # Final count
        final_dreams = self.count_existing_dreams(language)
        progress[language]['total_dreams'] = final_dreams
        
        if final_dreams >= self.total_dreams_per_language:
            print(f"🎉 {language} COMPLETED: {final_dreams} dreams generated!")
            return True
        else:
            print(f"⚠️  {language} INCOMPLETE: {final_dreams}/{self.total_dreams_per_language} dreams")
            return False
    
    async def generate_all_languages(self):
        """Generate 900 dreams for all languages."""
        print("🚀 STARTING 900-DREAM GENERATION")
        print(f"📋 Languages: {', '.join(self.languages)}")
        print(f"🎯 Target: {self.total_dreams_per_language} dreams per language")
        print(f"📦 Strategy: {self.sessions_per_language} sessions of {self.dreams_per_session} dreams each")
        print(f"⚙️  Batch size: {self.batch_size} dreams per batch")
        print(f"⏱️  Delay between sessions: {self.session_delay/60:.1f} minutes")
        
        start_time = time.time()
        results = {}
        
        for language in self.languages:
            success = await self.generate_language(language)
            results[language] = success
        
        total_time = time.time() - start_time
        
        # Final report
        print("\n" + "="*60)
        print("🎯 FINAL GENERATION REPORT")
        print("="*60)
        
        total_dreams = 0
        for language in self.languages:
            dream_count = self.count_existing_dreams(language)
            total_dreams += dream_count
            status = "✅ COMPLETE" if results[language] else "⚠️  INCOMPLETE"
            print(f"{language.upper():>10}: {dream_count:>4} dreams {status}")
        
        target_total = len(self.languages) * self.total_dreams_per_language
        print("-" * 60)
        print(f"{'TOTAL':>10}: {total_dreams:>4}/{target_total} dreams")
        print(f"{'TIME':>10}: {total_time/3600:.1f} hours")
        print(f"{'SUCCESS':>10}: {sum(results.values())}/{len(self.languages)} languages")
        
        if total_dreams >= target_total:
            print("\n🎉 MISSION ACCOMPLISHED! All 4,500 dreams generated!")
        else:
            print(f"\n⚠️  INCOMPLETE: {target_total - total_dreams} dreams remaining")
            
        return results
    
    def generate_analysis_script(self):
        """Generate a script to analyze all the collected data."""
        analysis_script = '''#!/usr/bin/env python3
"""
Analyze 900 dreams per language across multiple sessions.
Combines data from all sessions for comprehensive analysis.
"""

import pandas as pd
import json
from pathlib import Path
import glob

def combine_all_sessions():
    """Combine data from all sessions into comprehensive datasets."""
    logs_dir = Path('logs')
    all_dreams = []
    all_api_calls = []
    
    print("🔍 Scanning for dream sessions...")
    
    languages = ['english', 'basque', 'serbian', 'hebrew', 'slovenian']
    
    for language in languages:
        lang_dir = logs_dir / language / 'gpt-4o'
        if not lang_dir.exists():
            continue
            
        session_count = 0
        lang_dreams = 0
        
        for session_dir in lang_dir.iterdir():
            if session_dir.is_dir() and session_dir.name.startswith('session_'):
                session_count += 1
                
                # Load dreams
                dreams_file = session_dir / 'dreams.csv'
                if dreams_file.exists():
                    df = pd.read_csv(dreams_file)
                    successful_dreams = df[df['status'] == 'success']
                    all_dreams.append(successful_dreams)
                    lang_dreams += len(successful_dreams)
                
                # Load API calls
                api_file = session_dir / 'api_calls.csv'
                if api_file.exists():
                    df = pd.read_csv(api_file)
                    all_api_calls.append(df)
        
        print(f"  {language}: {session_count} sessions, {lang_dreams} dreams")
    
    # Combine all data
    if all_dreams:
        combined_dreams = pd.concat(all_dreams, ignore_index=True)
        combined_dreams.to_csv('combined_900_dreams.csv', index=False, encoding='utf-8')
        print(f"💾 Saved {len(combined_dreams)} dreams to combined_900_dreams.csv")
    
    if all_api_calls:
        combined_api_calls = pd.concat(all_api_calls, ignore_index=True)
        combined_api_calls.to_csv('combined_900_api_calls.csv', index=False, encoding='utf-8')
        print(f"💾 Saved {len(combined_api_calls)} API calls to combined_900_api_calls.csv")
    
    return combined_dreams if all_dreams else None, combined_api_calls if all_api_calls else None

def analyze_900_dreams():
    """Analyze the 900-dream dataset."""
    dreams_df, api_calls_df = combine_all_sessions()
    
    if dreams_df is None:
        print("❌ No dream data found!")
        return
    
    print("\\n📊 ANALYSIS OF 900-DREAM DATASET")
    print("="*50)
    
    # Language breakdown
    lang_counts = dreams_df['language'].value_counts()
    print("\\nDreams per language:")
    for lang, count in lang_counts.items():
        print(f"  {lang}: {count}")
    
    # Success rates from API calls
    if api_calls_df is not None:
        success_rates = api_calls_df.groupby('language')['status'].apply(
            lambda x: (x == 'success').mean() * 100
        ).round(1)
        print("\\nSuccess rates:")
        for lang, rate in success_rates.items():
            print(f"  {lang}: {rate}%")
    
    # Dream length analysis
    dreams_df['dream_length'] = dreams_df['dream'].str.len()
    length_stats = dreams_df.groupby('language')['dream_length'].agg(['mean', 'std', 'min', 'max']).round(0)
    print("\\nDream length statistics (characters):")
    print(length_stats)
    
    # Time span analysis
    if 'timestamp' in dreams_df.columns:
        dreams_df['timestamp'] = pd.to_datetime(dreams_df['timestamp'])
        time_span = dreams_df['timestamp'].max() - dreams_df['timestamp'].min()
        print(f"\\nGeneration time span: {time_span}")
    
    print(f"\\n✅ Analysis complete! Total dreams analyzed: {len(dreams_df)}")

if __name__ == "__main__":
    analyze_900_dreams()
'''
        
        with open('analyze_900_dreams.py', 'w') as f:
            f.write(analysis_script)
        
        print("📝 Created analyze_900_dreams.py script for data analysis")

async def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate 900 dreams per language')
    parser.add_argument('--language', type=str, help='Generate for specific language only')
    parser.add_argument('--sessions-per-language', type=int, default=9, help='Number of sessions per language')
    parser.add_argument('--dreams-per-session', type=int, default=100, help='Dreams per session')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size within each session')
    parser.add_argument('--session-delay', type=int, default=300, help='Seconds between sessions')
    parser.add_argument('--check-progress', action='store_true', help='Check current progress only')
    
    args = parser.parse_args()
    
    manager = DreamGenerationManager()
    
    # Override defaults with command line args
    manager.sessions_per_language = args.sessions_per_language
    manager.dreams_per_session = args.dreams_per_session
    manager.total_dreams_per_language = args.sessions_per_language * args.dreams_per_session
    manager.batch_size = args.batch_size
    manager.session_delay = args.session_delay
    
    if args.check_progress:
        # Check current progress
        print("📊 CURRENT PROGRESS REPORT")
        print("="*40)
        total_dreams = 0
        for language in manager.languages:
            dream_count = manager.count_existing_dreams(language)
            total_dreams += dream_count
            percentage = (dream_count / manager.total_dreams_per_language) * 100
            print(f"{language.upper():>10}: {dream_count:>4}/{manager.total_dreams_per_language} ({percentage:>5.1f}%)")
        
        target_total = len(manager.languages) * manager.total_dreams_per_language
        overall_percentage = (total_dreams / target_total) * 100
        print("-" * 40)
        print(f"{'TOTAL':>10}: {total_dreams:>4}/{target_total} ({overall_percentage:>5.1f}%)")
        return
    
    if args.language:
        # Generate for specific language only
        if args.language not in manager.languages:
            print(f"❌ Invalid language: {args.language}")
            print(f"Available languages: {', '.join(manager.languages)}")
            return
        
        manager.languages = [args.language]
        print(f"🎯 Generating {manager.total_dreams_per_language} dreams for {args.language} only")
    
    # Generate analysis script
    manager.generate_analysis_script()
    
    # Start generation
    results = await manager.generate_all_languages()
    
    print("\n🎯 Generation complete! Use these commands to analyze your data:")
    print("   python analyze_900_dreams.py                    # Quick analysis")
    print("   python analyze_multilingual_data.py             # Comprehensive analysis")
    print("   python statistical_analysis.py --session-id LATEST  # Statistical modeling")

if __name__ == "__main__":
    asyncio.run(main()) 