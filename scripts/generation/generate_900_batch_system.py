#!/usr/bin/env python3
"""
900-Dream Batch Generation System
Generate exactly 900 additional dreams per language in batches of 100.

Target: 4,500 total dreams (900 per language × 5 languages)
Batches: 9 batches of 100 dreams per language
Final Total: 1,000 dreams per language (100 existing + 900 new)
"""

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Import the successful configuration
from optimized_batch_v2 import OptimizedBatchV2, OptimizedV2Config

class Batch900Generator:
    """Generate 900 additional dreams per language in controlled batches"""
    
    def __init__(self, api_keys: Dict[str, str]):
        self.api_keys = api_keys
        
        # Configuration
        self.dreams_per_batch = 100
        self.total_batches = 9  # 900 ÷ 100 = 9 batches
        self.dreams_per_language = 900  # Additional dreams per language
        self.total_new_dreams = 4500  # 900 × 5 languages
        
        # Session management
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"BATCH900_{timestamp}"
        
        # Languages
        self.languages = ['english', 'basque', 'hebrew', 'serbian', 'slovenian']
        
        # Progress tracking
        self.completed_batches = {}
        for lang in self.languages:
            self.completed_batches[lang] = 0
        
        # Results tracking
        self.batch_results = {}
        self.total_start_time = None
        self._resumed_session = False
        
        print(f"🚀 900-DREAM BATCH GENERATION SYSTEM")
        print(f"=" * 60)
        print(f"📊 Target: 900 additional dreams per language")
        print(f"🎯 Total new dreams: {self.total_new_dreams:,}")
        print(f"🔄 Batches: {self.total_batches} batches of {self.dreams_per_batch} dreams each")
        print(f"📝 Session: {self.session_id}")
        print(f"⚙️  Using proven no-system-prompt configuration")
        print(f"🏁 Final total: 1,000 dreams per language")
        
        # Check for existing progress and offer resumption
        self.check_existing_progress()
        print()
    
    def get_current_progress(self) -> Dict:
        """Get current progress across all languages"""
        total_completed_batches = sum(self.completed_batches.values())
        total_target_batches = len(self.languages) * self.total_batches  # 5 × 9 = 45
        
        progress_percent = (total_completed_batches / total_target_batches) * 100
        
        return {
            'completed_batches': total_completed_batches,
            'target_batches': total_target_batches,
            'progress_percent': progress_percent,
            'by_language': self.completed_batches.copy()
        }
    
    def display_progress(self):
        """Display current progress"""
        progress = self.get_current_progress()
        
        print(f"\n📊 CURRENT PROGRESS")
        print(f"=" * 40)
        print(f"Overall: {progress['completed_batches']}/{progress['target_batches']} batches ({progress['progress_percent']:.1f}%)")
        print()
        
        for lang in self.languages:
            completed = progress['by_language'][lang]
            print(f"  {lang.title():>10}: {completed}/{self.total_batches} batches ({completed * self.dreams_per_batch}/900 dreams)")
        print()
    
    def check_existing_progress(self):
        """Check for existing batch directories and count completed dreams"""
        print(f"\n🔍 Checking for existing progress...")
        
        # Look for existing BATCH900 sessions in main logs directory
        existing_sessions = []
        if os.path.exists('logs'):
            # Scan each language directory for BATCH900 sessions
            for lang in self.languages:
                lang_dir = os.path.join('logs', lang, 'gpt-4o')
                if os.path.exists(lang_dir):
                    for item in os.listdir(lang_dir):
                        if 'BATCH900_' in item:
                            # Extract the base session ID (without _B01, _B02, etc.)
                            base_session = item.split('_B')[0] if '_B' in item else item
                            session_key = base_session.replace('session_', '')
                            
                            # Only process each unique session once
                            if not any(s['session_id'] == session_key for s in existing_sessions):
                                progress = self.scan_logs_directory_progress(session_key)
                                if progress['total_dreams'] > 0:
                                    existing_sessions.append({
                                        'session_id': session_key,
                                        'path': f'logs/',
                                        'progress': progress
                                    })
        
        if existing_sessions:
            print(f"🔄 Found {len(existing_sessions)} previous session(s) with progress:")
            
            # Sort by most recent
            existing_sessions.sort(key=lambda x: x['session_id'], reverse=True)
            
            for i, session in enumerate(existing_sessions[:3], 1):  # Show top 3
                prog = session['progress']
                print(f"  {i}. {session['session_id']}")
                print(f"     Progress: {prog['total_dreams']}/4,500 dreams ({prog['completion_percent']:.1f}%)")
                print(f"     Status: {prog['status']}")
                for lang, count in prog['by_language'].items():
                    print(f"       {lang.title()}: {count}/900 dreams")
                print()
            
            print(f"💡 Resume Options:")
            print(f"  • Press Enter to start new session")
            print(f"  • Type '1', '2', etc. to resume a session")
            print(f"  • Type 'q' to quit")
            
            choice = input("Enter your choice: ").strip()
            
            if choice.isdigit() and 1 <= int(choice) <= len(existing_sessions):
                selected_session = existing_sessions[int(choice) - 1]
                self.resume_from_session(selected_session)
                return True
            elif choice.lower() == 'q':
                print("👋 Goodbye!")
                exit(0)
        
        print(f"▶️  Starting new session: {self.session_id}")
        return False
    
    def scan_session_progress(self, session_path: str, session_id: str) -> Dict:
        """Scan a session directory and count completed dreams"""
        progress = {
            'total_dreams': 0,
            'by_language': {},
            'completion_percent': 0,
            'status': 'Unknown'
        }
        
        try:
            # Initialize language counts
            for lang in self.languages:
                progress['by_language'][lang] = 0
            
            # Scan batch directories
            for batch_num in range(1, 10):  # batches 1-9
                batch_dir = os.path.join(session_path, f'batch_{batch_num:02d}')
                if os.path.exists(batch_dir):
                    # Count dreams in each language for this batch
                    for lang in self.languages:
                        lang_dir = os.path.join(batch_dir, lang, 'gpt-4o')
                        if os.path.exists(lang_dir):
                            # Look for session subdirectories
                            for subdir in os.listdir(lang_dir):
                                if subdir.startswith('session_'):
                                    dreams_file = os.path.join(lang_dir, subdir, 'dreams.csv')
                                    if os.path.exists(dreams_file):
                                        try:
                                            import pandas as pd
                                            df = pd.read_csv(dreams_file)
                                            successful_dreams = len(df[df['status'] == 'success'])
                                            progress['by_language'][lang] += successful_dreams
                                        except:
                                            pass
            
            # Calculate totals
            progress['total_dreams'] = sum(progress['by_language'].values())
            progress['completion_percent'] = (progress['total_dreams'] / 4500) * 100
            
            # Determine status
            if progress['total_dreams'] == 0:
                progress['status'] = 'No progress'
            elif progress['total_dreams'] >= 4500:
                progress['status'] = '✅ Complete'
            else:
                progress['status'] = '🔄 In progress'
                
        except Exception as e:
            print(f"⚠️  Error scanning {session_id}: {e}")
        
        return progress
    
    def scan_logs_directory_progress(self, session_id: str) -> Dict:
        """Scan the main logs directory for BATCH900 progress"""
        progress = {
            'total_dreams': 0,
            'by_language': {},
            'completion_percent': 0,
            'status': 'Unknown'
        }
        
        try:
            # Initialize language counts
            for lang in self.languages:
                progress['by_language'][lang] = 0
            
            # Scan each language's sessions
            for lang in self.languages:
                lang_dir = os.path.join('logs', lang, 'gpt-4o')
                if os.path.exists(lang_dir):
                    # Look for all sessions with this BATCH900 session ID
                    for session_dir in os.listdir(lang_dir):
                        # Match new format: session_BATCH900_20250707_HHMMSS_ENGLISH_B01
                        if session_id in session_dir and session_dir.startswith('session_') and f"_{lang.upper()}_B" in session_dir:
                            dreams_file = os.path.join(lang_dir, session_dir, 'dreams.csv')
                            if os.path.exists(dreams_file):
                                try:
                                    import pandas as pd
                                    df = pd.read_csv(dreams_file)
                                    successful_dreams = len(df[df['status'] == 'success'])
                                    if successful_dreams > 0:  # Only count sessions with actual dreams
                                        progress['by_language'][lang] += successful_dreams
                                        print(f"    📁 Found {successful_dreams} dreams in {session_dir}")
                                    else:
                                        print(f"    🗑️ Empty session: {session_dir}")
                                except Exception as e:
                                    print(f"    ⚠️ Error reading {session_dir}: {e}")
            
            # Calculate totals
            progress['total_dreams'] = sum(progress['by_language'].values())
            progress['completion_percent'] = (progress['total_dreams'] / 4500) * 100
            
            # Determine status
            if progress['total_dreams'] == 0:
                progress['status'] = 'No progress'
            elif progress['total_dreams'] >= 4500:
                progress['status'] = '✅ Complete'
            else:
                progress['status'] = '🔄 In progress'
                
        except Exception as e:
            print(f"⚠️  Error scanning logs for {session_id}: {e}")
        
        return progress
    
    def resume_from_session(self, session_info: Dict):
        """Resume from an existing session"""
        self.session_id = session_info['session_id']
        progress = session_info['progress']
        
        print(f"\n🔄 RESUMING SESSION: {self.session_id}")
        print(f"📊 Current progress: {progress['total_dreams']}/4,500 dreams ({progress['completion_percent']:.1f}%)")
        
        # Set up progress tracking based on existing dreams
        for lang in self.languages:
            existing_dreams = progress['by_language'][lang]
            # ONLY count complete batches of 100 dreams
            completed_batches = existing_dreams // 100  # Integer division
            self.completed_batches[lang] = completed_batches
            
            print(f"📊 {lang.title()}: {existing_dreams} existing dreams = {completed_batches} complete batches")
            
            if existing_dreams % 100 > 0:  # Partial batch exists
                remaining_in_partial = existing_dreams % 100
                print(f"⚠️  {lang.title()}: Found partial batch with {remaining_in_partial} dreams")
                print(f"    Will continue from batch {completed_batches + 1}")
            elif existing_dreams == 0:
                print(f"▶️  {lang.title()}: No dreams found, starting from batch 1")
            else:
                print(f"✅ {lang.title()}: All batches 1-{completed_batches} are complete")
        
        print(f"✅ Session resumed! Will continue from where it left off.")
        
        # Mark as resumed session
        self._resumed_session = True
        
        # Display current status
        self.display_progress()
    
    async def run_single_batch(self, language: str, batch_number: int) -> Dict:
        """Run a single batch of 100 dreams for one language"""
        
        print(f"\n🔄 BATCH {batch_number}/9 - {language.upper()}")
        print(f"Target: {self.dreams_per_batch} dreams")
        
        batch_start_time = time.time()
        
        # Create batch generator with specific config
        batch_generator = OptimizedBatchV2(self.api_keys)
        
        # Configure for this specific batch
        batch_generator.config.dreams_per_language = self.dreams_per_batch
        batch_generator.config.total_target_dreams = self.dreams_per_batch
        
        # Save to main logs directory (same as existing dreams)
        batch_generator.base_logs_dir = 'logs'
        
        # CRITICAL: Set starting dream number to continue from existing dreams
        # Your existing dreams: 1-100, so batch 1 should start at 101, batch 2 at 201, etc.
        existing_dreams_base = 100  # Your baseline dreams per language
        start_dream_number = existing_dreams_base + ((batch_number - 1) * self.dreams_per_batch) + 1
        
        # Override the dream numbering in the generator
        batch_generator._starting_dream_number = start_dream_number
        
        print(f"    🔢 Dream numbering: {start_dream_number} to {start_dream_number + self.dreams_per_batch - 1}")
        
        # CRITICAL: Use unique session ID per language to avoid conflicts
        batch_generator.session_id = f"{self.session_id}_{language.upper()}_B{batch_number:02d}"
        print(f"    🆔 Session ID: {batch_generator.session_id}")
        
        try:
            # Generate just this language for this batch
            result = await batch_generator.generate_language_batch(language)
            
            batch_duration = time.time() - batch_start_time
            
            # Update progress
            self.completed_batches[language] += 1
            
            # Store results
            batch_key = f"{language}_batch_{batch_number}"
            self.batch_results[batch_key] = {
                'language': language,
                'batch_number': batch_number,
                'successful_dreams': result['successful_dreams'],
                'failed_dreams': result['failed_dreams'], 
                'success_rate': result['success_rate'],
                'duration_minutes': batch_duration / 60,
                'avg_chars': result['avg_chars'],
                'avg_words': result['avg_words']
            }
            
            print(f"  ✅ Batch {batch_number} complete!")
            print(f"  📊 Success: {result['successful_dreams']}/{self.dreams_per_batch} dreams ({result['success_rate']:.1f}%)")
            print(f"  ⏱️  Duration: {batch_duration/60:.1f} minutes")
            print(f"  📏 Average: {result['avg_chars']:.0f} chars, {result['avg_words']:.0f} words")
            
            return result
            
        except Exception as e:
            print(f"  ❌ Batch {batch_number} failed: {e}")
            return None
    
    async def run_batch_round(self, batch_number: int):
        """Run one batch round across all languages"""
        
        print(f"\n🎬 STARTING BATCH ROUND {batch_number}/9")
        print(f"=" * 50)
        print(f"Generating {self.dreams_per_batch} dreams per language...")
        
        round_start_time = time.time()
        
        # Run all languages for this batch
        tasks = []
        for language in self.languages:
            # Only generate if this language hasn't completed this batch
            if self.completed_batches[language] < batch_number:
                print(f"  📋 {language.title()}: Needs batch {batch_number} (currently {self.completed_batches[language]}/9)")
                task = self.run_single_batch(language, batch_number)
                tasks.append(task)
            else:
                print(f"  ✅ {language.title()}: Batch {batch_number} already complete")
        
        if tasks:
            # Run languages sequentially to avoid race conditions and duplicate numbering
            results = []
            for task in tasks:
                try:
                    result = await task
                    results.append(result)
                except Exception as e:
                    print(f"❌ Task failed: {e}")
                    results.append(e)
            
            round_duration = time.time() - round_start_time
            
            # Summary for this round
            successful_batches = len([r for r in results if r and not isinstance(r, Exception)])
            total_dreams_this_round = successful_batches * self.dreams_per_batch
            
            print(f"\n🎉 BATCH ROUND {batch_number} COMPLETE!")
            print(f"  ✅ Successful batches: {successful_batches}/{len(tasks)}")
            print(f"  📊 Dreams generated: {total_dreams_this_round}")
            print(f"  ⏱️  Round duration: {round_duration/60:.1f} minutes")
            
            # Display current progress
            self.display_progress()
        else:
            print(f"  ✅ Batch round {batch_number} already complete!")
    
    async def run_complete_900_generation(self):
        """Run the complete 900-dream generation process"""
        
        print(f"\n🎬 STARTING 900-DREAM GENERATION")
        print(f"Target: {self.total_new_dreams:,} dreams across {len(self.languages)} languages")
        
        self.total_start_time = time.time()
        
        try:
            # Run 9 batch rounds
            for batch_num in range(1, self.total_batches + 1):
                await self.run_batch_round(batch_num)
                
                # Small break between rounds
                if batch_num < self.total_batches:
                    print(f"\n💤 Brief pause before next batch...")
                    await asyncio.sleep(5)
            
            # Generate final summary
            await self.generate_completion_report()
            
        except KeyboardInterrupt:
            print(f"\n🛑 Generation interrupted by user")
            print(f"💾 Progress has been saved automatically")
            await self.generate_progress_report()
        except Exception as e:
            print(f"❌ Error during generation: {e}")
            await self.generate_progress_report()
    
    async def generate_completion_report(self):
        """Generate comprehensive completion report"""
        
        total_duration = time.time() - self.total_start_time
        total_completed_batches = sum(self.completed_batches.values())
        
        report_file = f"batch900_completion_report_{self.session_id}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# 900-Dream Batch Generation Report\n\n")
            f.write(f"**Session**: {self.session_id}\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total Duration**: {total_duration/60:.1f} minutes ({total_duration/3600:.1f} hours)\n\n")
            
            f.write(f"## Summary\n\n")
            f.write(f"- **Target**: 900 dreams per language (4,500 total)\n")
            f.write(f"- **Batch Strategy**: 9 batches of 100 dreams each\n")
            f.write(f"- **Completed Batches**: {total_completed_batches}/45\n")
            f.write(f"- **Success Rate**: Based on proven no-system-prompt config\n\n")
            
            f.write(f"## Language Progress\n\n")
            f.write(f"| Language | Batches | Dreams | Status |\n")
            f.write(f"|----------|---------|--------|--------|\n")
            
            for lang in self.languages:
                batches = self.completed_batches[lang]
                dreams = batches * self.dreams_per_batch
                status = "✅ Complete" if batches == 9 else f"🔄 {batches}/9"
                f.write(f"| {lang.title()} | {batches}/9 | {dreams}/900 | {status} |\n")
            
            f.write(f"\n## Batch Results\n\n")
            for batch_key, result in self.batch_results.items():
                f.write(f"### {result['language'].title()} - Batch {result['batch_number']}\n")
                f.write(f"- Dreams: {result['successful_dreams']}/{self.dreams_per_batch}\n")
                f.write(f"- Success Rate: {result['success_rate']:.1f}%\n")
                f.write(f"- Duration: {result['duration_minutes']:.1f} minutes\n")
                f.write(f"- Average: {result['avg_words']:.0f} words, {result['avg_chars']:.0f} chars\n\n")
            
            f.write(f"## Final Dataset\n\n")
            f.write(f"After completion, you will have:\n")
            f.write(f"- **1,000 dreams per language** (100 existing + 900 new)\n")
            f.write(f"- **5,000 total dreams** across 5 languages\n")
            f.write(f"- **Research-grade dataset** suitable for publication\n")
            f.write(f"- **No system prompts** - pure comparison data\n")
        
        print(f"\n📄 Completion report saved: {report_file}")
        
        # Display final summary
        print(f"\n🎉 900-DREAM GENERATION COMPLETE!")
        print(f"✅ Total batches completed: {total_completed_batches}/45")
        print(f"⏱️  Total time: {total_duration/60:.1f} minutes")
        print(f"🏁 Final dataset: 1,000 dreams per language (5,000 total)")
    
    async def generate_progress_report(self):
        """Generate progress report for interrupted sessions"""
        
        progress = self.get_current_progress()
        report_file = f"batch900_progress_report_{self.session_id}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# 900-Dream Generation Progress Report\n\n")
            f.write(f"**Session**: {self.session_id}\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Status**: Interrupted/In Progress\n\n")
            
            f.write(f"## Current Progress\n\n")
            f.write(f"- **Completed Batches**: {progress['completed_batches']}/45 ({progress['progress_percent']:.1f}%)\n")
            f.write(f"- **Dreams Generated**: ~{progress['completed_batches'] * 100:,}\n\n")
            
            f.write(f"## Language Status\n\n")
            for lang in self.languages:
                batches = progress['by_language'][lang]
                dreams = batches * 100
                f.write(f"- **{lang.title()}**: {batches}/9 batches ({dreams}/900 dreams)\n")
            
            f.write(f"\n## Resume Instructions\n\n")
            f.write(f"To resume this session, the system will automatically detect incomplete batches\n")
            f.write(f"and continue from where it left off.\n")
        
        print(f"📄 Progress report saved: {report_file}")

async def main():
    """Main interface for 900-dream batch generation"""
    
    print(f"🌙 900-DREAM BATCH GENERATION SYSTEM")
    print(f"Based on your successful OptimizedBatchV2 configuration")
    print()
    
    # Load API keys from environment variables (same as working scripts)
    import os
    api_keys = {
        'openai': os.getenv('OPENAI_API_KEY'),
        'anthropic': os.getenv('ANTHROPIC_API_KEY'),
        'google': os.getenv('GOOGLE_API_KEY')
    }
    
    # Check if we have at least OpenAI key
    if not api_keys['openai']:
        print(f"❌ OpenAI API key not found!")
        print(f"Please set your environment variable:")
        print(f"  export OPENAI_API_KEY='your-api-key-here'")
        print(f"Or in PowerShell:")
        print(f"  $env:OPENAI_API_KEY='your-api-key-here'")
        return
    
    print(f"✅ Found OpenAI API key: {'*' * (len(api_keys['openai']) - 4)}{api_keys['openai'][-4:]}")
    
    print(f"📋 GENERATION PLAN:")
    print(f"  • 900 additional dreams per language")
    print(f"  • 4,500 total new dreams")
    print(f"  • 9 batches of 100 dreams each")
    print(f"  • Final total: 1,000 dreams per language")
    print(f"  • No system prompts (proven configuration)")
    print(f"  • Estimated time: ~750 minutes (12.5 hours)")
    print()
    
    # Create the generator (this will check for existing progress)
    generator = Batch900Generator(api_keys)
    
    # If we're resuming, skip confirmation
    if hasattr(generator, '_resumed_session') and generator._resumed_session:
        print(f"🚀 Continuing from resumed session...")
        await generator.run_complete_900_generation()
    else:
        # New session - ask for confirmation
        confirm = input(f"🚀 Start 900-dream generation? (y/n): ").strip().lower()
        
        if confirm == 'y':
            await generator.run_complete_900_generation()
        else:
            print(f"❌ Generation cancelled")

if __name__ == "__main__":
    asyncio.run(main()) 