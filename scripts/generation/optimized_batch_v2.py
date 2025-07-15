#!/usr/bin/env python3
"""
Optimized Dream Batch Generator v2.0
Pure Immediate Dream Scenario with Enhanced Parameters

This is a standalone batch generator that operates independently from previous systems.
Uses refined prompts, optimized parameters, and pure immediate dream scenario.

Version: 2.0
Session Prefix: OPT_V2_
Date: 2025-01-06
"""

import asyncio
import json
import csv
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass

# Import the optimized configuration
from optimized_dream_languages import LANGUAGE_CONFIG, get_optimized_config
from src.models.llm_interface import LLMInterface, GenerationConfig

@dataclass
class OptimizedV2Config:
    """Version 2.0 optimized configuration"""
    version: str = "2.0"
    system_name: str = "OptimizedBatchV2"
    session_prefix: str = "OPT_V2_"
    temperature: float = 1.1
    top_p: float = 0.98
    presence_penalty: float = 0.1
    frequency_penalty: float = 0.0
    batch_size: int = 50
    dreams_per_language: int = 100
    total_target_dreams: int = 500
    use_system_prompt: bool = False
    use_markers: bool = False
    use_variants: bool = False
    scenario_type: str = "Pure Immediate Dream Writing"

class OptimizedBatchV2:
    """Optimized Dream Batch Generator Version 2.0"""
    
    def __init__(self, api_keys: Dict[str, str], model: str = 'gpt-4o'):
        self.config = OptimizedV2Config()
        self.llm_interface = LLMInterface(api_keys)
        self.model = model
        
        # Create unique session ID with version prefix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"{self.config.session_prefix}{timestamp}"
        
        # Create separate logs directory for V2
        self.base_logs_dir = 'logs_optimized_v2'
        os.makedirs(self.base_logs_dir, exist_ok=True)
        
        # Data structures
        self.api_calls_data = {}
        self.dreams_data = {}
        self.batch_tracker = {}
        self.generation_stats = {}
        
        # Timing tracking
        self.call_times = []
        self.start_time = time.time()
        
        # Checkpoint and resumption
        self.checkpoint_file = f"{self.base_logs_dir}/checkpoint_{self.session_id}.json"
        self.resume_session_id = None
        
        print(f"🚀 Optimized Dream Batch Generator v{self.config.version}")
        print(f"📊 Session: {self.session_id}")
        print(f"🎯 Target: {self.config.total_target_dreams} dreams ({self.config.dreams_per_language} per language)")
        print(f"⚙️  Config: temp={self.config.temperature}, top_p={self.config.top_p}")
        print(f"📝 Scenario: {self.config.scenario_type}")
        print(f"📁 Logs: {self.base_logs_dir}/")
        
        # Check for existing sessions to resume
        self.check_for_resumable_sessions()
    
    def check_for_resumable_sessions(self):
        """Check for existing incomplete sessions that can be resumed"""
        if not os.path.exists(self.base_logs_dir):
            return
            
        # Look for existing session directories
        session_dirs = []
        for item in os.listdir(self.base_logs_dir):
            if item.startswith('OPT_V2_') and os.path.isdir(os.path.join(self.base_logs_dir, item)):
                session_dirs.append(item)
        
        if not session_dirs:
            return
            
        # Find the most recent session
        session_dirs.sort(reverse=True)
        
        # Check if any sessions have incomplete progress
        resumable_sessions = []
        for session_dir in session_dirs[:3]:  # Check last 3 sessions
            session_path = os.path.join(self.base_logs_dir, session_dir)
            progress = self.get_session_progress(session_path)
            if progress and progress['incomplete']:
                resumable_sessions.append({
                    'session_id': session_dir.replace('session_', ''),
                    'path': session_path,
                    'progress': progress
                })
        
        if resumable_sessions:
            print(f"\n🔄 Found {len(resumable_sessions)} resumable session(s):")
            for i, session in enumerate(resumable_sessions, 1):
                prog = session['progress']
                print(f"  {i}. Session {session['session_id']}")
                print(f"     Progress: {prog['completed_dreams']}/{prog['total_target']} dreams")
                print(f"     Languages: {prog['languages_status']}")
            
            print(f"\n💡 Options:")
            print(f"  • Press Enter to start new session")
            print(f"  • Type '1', '2', etc. to resume a session")
            print(f"  • Type 'q' to quit")
            
            choice = input("Enter your choice: ").strip()
            
            if choice.isdigit() and 1 <= int(choice) <= len(resumable_sessions):
                selected_session = resumable_sessions[int(choice) - 1]
                self.resume_session(selected_session)
                return
            elif choice.lower() == 'q':
                print("👋 Goodbye!")
                exit(0)
        
        print(f"\n▶️  Starting new session: {self.session_id}")
    
    def get_session_progress(self, session_path: str) -> Optional[Dict]:
        """Get progress information for a session"""
        try:
            # Check if this is a proper session directory
            if not os.path.exists(session_path):
                return None
                
            # Look for language subdirectories
            total_dreams = 0
            languages_status = {}
            
            for language in LANGUAGE_CONFIG.keys():
                lang_path = os.path.join(session_path, language, 'gpt-4o')
                if os.path.exists(lang_path):
                    # Find the session subdirectory
                    session_subdirs = [d for d in os.listdir(lang_path) if d.startswith('session_')]
                    if session_subdirs:
                        session_subdir = session_subdirs[0]  # Take the first one
                        dreams_file = os.path.join(lang_path, session_subdir, 'dreams.csv')
                        if os.path.exists(dreams_file):
                            try:
                                df = pd.read_csv(dreams_file)
                                successful_dreams = len(df[df['status'] == 'success'])
                                total_dreams += successful_dreams
                                languages_status[language] = f"{successful_dreams}/{self.config.dreams_per_language}"
                            except:
                                languages_status[language] = "0/100"
                        else:
                            languages_status[language] = "0/100"
                else:
                    languages_status[language] = "0/100"
            
            # Determine if incomplete
            incomplete = total_dreams < self.config.total_target_dreams
            
            return {
                'completed_dreams': total_dreams,
                'total_target': self.config.total_target_dreams,
                'languages_status': languages_status,
                'incomplete': incomplete
            }
        except Exception as e:
            print(f"⚠️  Error checking session progress: {e}")
            return None
    
    def resume_session(self, session_info: Dict):
        """Resume an existing session"""
        self.resume_session_id = session_info['session_id']
        self.session_id = self.resume_session_id
        
        print(f"\n🔄 Resuming session: {self.session_id}")
        print(f"📊 Current progress: {session_info['progress']['completed_dreams']}/{session_info['progress']['total_target']} dreams")
        
        # Load existing data
        self.load_existing_session_data()
        
        print(f"✅ Session {self.session_id} loaded successfully")
        print(f"📈 Resuming from where we left off...")
    
    def load_existing_session_data(self):
        """Load existing session data for resumption"""
        for language in LANGUAGE_CONFIG.keys():
            lang_path = os.path.join(self.base_logs_dir, language, 'gpt-4o')
            if os.path.exists(lang_path):
                session_subdirs = [d for d in os.listdir(lang_path) if d.startswith('session_')]
                if session_subdirs:
                    session_subdir = session_subdirs[0]
                    session_path = os.path.join(lang_path, session_subdir)
                    
                    # Load API calls
                    api_calls_file = os.path.join(session_path, 'api_calls.csv')
                    if os.path.exists(api_calls_file):
                        try:
                            df = pd.read_csv(api_calls_file)
                            self.api_calls_data[language] = df.to_dict('records')
                        except:
                            self.api_calls_data[language] = []
                    
                    # Load dreams
                    dreams_file = os.path.join(session_path, 'dreams.csv')
                    if os.path.exists(dreams_file):
                        try:
                            df = pd.read_csv(dreams_file)
                            self.dreams_data[language] = df.to_dict('records')
                        except:
                            self.dreams_data[language] = []
                    
                    # Initialize generation stats
                    if language not in self.generation_stats:
                        self.generation_stats[language] = {
                            'successful': len([d for d in self.dreams_data.get(language, []) if d.get('status') == 'success']),
                            'failed': len([d for d in self.dreams_data.get(language, []) if d.get('status') != 'success']),
                            'total_chars': 0,
                            'total_words': 0,
                            'avg_duration': 0,
                            'durations': []
                        }

    def setup_language_directory(self, language: str):
        """Setup directory structure for a language"""
        lang_dir = os.path.join(
            self.base_logs_dir,
            language,
            self.model,
            f"session_{self.session_id}"
        )
        os.makedirs(lang_dir, exist_ok=True)
        
        # Initialize data structures
        if language not in self.api_calls_data:
            self.api_calls_data[language] = []
        if language not in self.dreams_data:
            self.dreams_data[language] = []
        if language not in self.generation_stats:
            self.generation_stats[language] = {
                'successful': 0,
                'failed': 0,
                'total_chars': 0,
                'total_words': 0,
                'avg_duration': 0,
                'durations': []
            }
        
        return lang_dir
    
    async def generate_single_dream(self, language: str, dream_number: int) -> Dict[str, Any]:
        """Generate a single optimized dream"""
        
        config = LANGUAGE_CONFIG[language]
        prompt = config['prompt']  # Fixed: use config['prompt'] instead of OPTIMIZED_CONFIG
        
        # Generate unique IDs
        call_id = str(uuid.uuid4())
        batch_id = f"v2_batch_{language}_{datetime.now().strftime('%H%M%S')}"
        user_id = f"v2_user_{uuid.uuid4().hex[:8]}_{np.random.randint(10000, 99999)}"
        prompt_id = f"v2_{hash(prompt) % 100000000:08x}"
        
        start_time = time.time()
        
        # Generation config
        gen_config = GenerationConfig(
            model=self.model,
            temperature=self.config.temperature,
            max_tokens=1000,
            top_p=self.config.top_p,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty
        )
        
        try:
            # Generate dream (no system prompt for pure scenario)
            dream_content = await self.llm_interface.generate_dream(prompt, gen_config, None)
            end_time = time.time()
            duration = end_time - start_time
            status = 'success'
            
            # Track timing
            self.call_times.append(end_time)
            self.generation_stats[language]['durations'].append(duration)
            
            # Update stats
            self.generation_stats[language]['successful'] += 1
            self.generation_stats[language]['total_chars'] += len(dream_content)
            self.generation_stats[language]['total_words'] += len(dream_content.split())
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            status = 'error'
            dream_content = f"Generation failed: {str(e)}"
            self.generation_stats[language]['failed'] += 1
            
            print(f"  ❌ Dream {dream_number} failed: {e}")
        
        # Create comprehensive API call record
        api_call = {
            'call_id': call_id,
            'batch_id': batch_id,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'language': language,
            'language_code': config['code'],
            'script': config['script'],
            'model': self.model,
            'temperature': self.config.temperature,
            'top_p': self.config.top_p,
            'presence_penalty': self.config.presence_penalty,
            'frequency_penalty': self.config.frequency_penalty,
            'base_prompt': prompt,
            'modified_prompt': prompt,
            'system_message': None,  # Pure scenario
            'prompt_id': prompt_id,
            'marker_info': 'none',
            'used_invisible_markers': False,
            'dream_number': dream_number,
            'batch_size': self.config.batch_size,
            'dream': dream_content,
            'status': status,
            'duration_seconds': round(duration, 3),
            'temporal_delay_seconds': 0.1,
            'start_time': datetime.fromtimestamp(start_time).isoformat(),
            'end_time': datetime.fromtimestamp(end_time).isoformat(),
            'session_id': self.session_id,
            'version': self.config.version,
            'scenario_type': self.config.scenario_type
        }
        
        # Create dream record
        dream_record = {
            'call_id': call_id,
            'batch_id': batch_id,
            'user_id': user_id,
            'language': language,
            'language_code': config['code'],
            'script': config['script'],
            'model': self.model,
            'temperature': self.config.temperature,
            'dream': dream_content,
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'dream_number': dream_number,
            'prompt_id': prompt_id,
            'marker_info': 'none',
            'temporal_delay': 0.1,
            'version': self.config.version,
            'char_count': len(dream_content),
            'word_count': len(dream_content.split())
        }
        
        # Store records
        self.api_calls_data[language].append(api_call)
        self.dreams_data[language].append(dream_record)
        
        # Progress indicator
        char_count = len(dream_content)
        word_count = len(dream_content.split())
        print(f"  ✅ Dream {dream_number}: {status} ({duration:.2f}s) - {char_count} chars, {word_count} words")
        
        return {
            'call_id': call_id,
            'language': language,
            'status': status,
            'duration': duration,
            'char_count': char_count,
            'word_count': word_count
        }
    
    def get_language_progress(self, language: str) -> Dict:
        """Get progress for a specific language"""
        existing_dreams = self.dreams_data.get(language, [])
        successful_dreams = len([d for d in existing_dreams if d.get('status') == 'success'])
        
        return {
            'completed': successful_dreams,
            'target': self.config.dreams_per_language,
            'remaining': max(0, self.config.dreams_per_language - successful_dreams),
            'progress_percent': (successful_dreams / self.config.dreams_per_language) * 100
        }
    
    async def generate_language_batch(self, language: str) -> Dict[str, Any]:
        """Generate complete batch for one language with resumption support"""
        
        # Check current progress
        progress = self.get_language_progress(language)
        
        print(f"\n🌍 Starting {language.upper()} generation")
        print(f"📝 Prompt: {LANGUAGE_CONFIG[language]['prompt']}")
        print(f"📊 Current progress: {progress['completed']}/{progress['target']} dreams ({progress['progress_percent']:.1f}%)")
        
        if progress['remaining'] == 0:
            print(f"✅ {language.upper()} already complete!")
            return {
                'language': language,
                'successful_dreams': progress['completed'],
                'failed_dreams': 0,
                'success_rate': 100.0,
                'avg_chars': 0,
                'avg_words': 0,
                'batch_duration': 0,
                'avg_dream_duration': 0,
                'resumed': True
            }
        
        print(f"🎯 Need {progress['remaining']} more dreams")
        
        # Setup directory
        lang_dir = self.setup_language_directory(language)
        
        # Generate remaining dreams
        batch_start = time.time()
        results = []
        
        # Check if we have a starting dream number override (for batch continuation)
        if hasattr(self, '_starting_dream_number'):
            # For fresh batch: start_number = override (e.g., 101)
            # For resumed batch: start_number = override + completed (e.g., 101 + 50 = 151)
            start_number = self._starting_dream_number + progress['completed']
            end_number = start_number + progress['remaining']
            print(f"🔢 Using override start number: {self._starting_dream_number}")
            print(f"🔢 Adjusted for completed dreams: {start_number} (continuing after {progress['completed']} existing)")
        else:
            start_number = progress['completed'] + 1
            end_number = start_number + progress['remaining']
        
        for dream_num in range(start_number, end_number):
            try:
                result = await self.generate_single_dream(language, dream_num)
                results.append(result)
                
                # Small delay between requests
                await asyncio.sleep(0.1)
                
                # Progress updates every 25 dreams
                if (dream_num - start_number + 1) % 25 == 0:
                    successful = len([r for r in results if r['status'] == 'success'])
                    remaining_count = end_number - dream_num - 1
                    print(f"  📊 Progress: {dream_num - start_number + 1}/{progress['remaining']} ({successful} successful, {remaining_count} remaining)")
                    
                    # Save intermediate progress
                    await self.save_language_data(language, lang_dir)
                    
            except KeyboardInterrupt:
                print(f"\n🛑 Interrupted during {language} generation")
                print(f"💾 Saving progress...")
                await self.save_language_data(language, lang_dir)
                raise
            except Exception as e:
                print(f"❌ Error in {language} generation: {e}")
                continue
        
        batch_duration = time.time() - batch_start
        
        # Calculate final stats
        successful_dreams = len([r for r in results if r['status'] == 'success'])
        failed_dreams = len([r for r in results if r['status'] != 'success'])
        
        if successful_dreams > 0:
            avg_chars = np.mean([r['char_count'] for r in results if r['status'] == 'success'])
            avg_words = np.mean([r['word_count'] for r in results if r['status'] == 'success'])
        else:
            avg_chars = 0
            avg_words = 0
            
        success_rate = (successful_dreams / len(results)) * 100 if results else 0
        
        # Update generation stats
        stats = self.generation_stats[language]
        stats['avg_duration'] = np.mean(stats['durations']) if stats['durations'] else 0
        
        total_successful = progress['completed'] + successful_dreams
        print(f"  🎉 {language.upper()} Complete!")
        print(f"    ✅ Total successful: {total_successful}/{self.config.dreams_per_language}")
        print(f"    📏 Recent batch average: {avg_chars:.0f} chars, {avg_words:.0f} words")
        print(f"    ⏱️  Duration: {batch_duration:.1f}s (avg {stats['avg_duration']:.2f}s per dream)")
        
        # Save final language data
        await self.save_language_data(language, lang_dir)
        
        return {
            'language': language,
            'successful_dreams': total_successful,
            'failed_dreams': failed_dreams,
            'success_rate': (total_successful / self.config.dreams_per_language) * 100,
            'avg_chars': avg_chars,
            'avg_words': avg_words,
            'batch_duration': batch_duration,
            'avg_dream_duration': stats['avg_duration'],
            'resumed': progress['completed'] > 0
        }
    
    async def save_language_data(self, language: str, lang_dir: str):
        """Save all data for a language"""
        
        # Save API calls CSV
        api_calls_file = os.path.join(lang_dir, "api_calls.csv")
        if self.api_calls_data[language]:
            api_df = pd.DataFrame(self.api_calls_data[language])
            api_df.to_csv(api_calls_file, index=False, encoding='utf-8')
        
        # Save dreams CSV
        dreams_file = os.path.join(lang_dir, "dreams.csv")
        if self.dreams_data[language]:
            dreams_df = pd.DataFrame(self.dreams_data[language])
            dreams_df.to_csv(dreams_file, index=False, encoding='utf-8')
        
        # Create session data JSON
        successful_calls = len([call for call in self.api_calls_data[language] if call['status'] == 'success'])
        failed_calls = len([call for call in self.api_calls_data[language] if call['status'] != 'success'])
        total_calls = len(self.api_calls_data[language])
        
        # Calculate temporal statistics
        if len(self.call_times) > 1:
            intervals = [self.call_times[i] - self.call_times[i-1] for i in range(1, len(self.call_times))]
            temporal_stats = {
                'intervals': intervals,
                'mean_interval': np.mean(intervals),
                'std_interval': np.std(intervals),
                'min_interval': np.min(intervals),
                'max_interval': np.max(intervals),
                'total_span_hours': (self.call_times[-1] - self.call_times[0]) / 3600
            }
        else:
            temporal_stats = {
                'intervals': [],
                'mean_interval': 0,
                'std_interval': 0,
                'min_interval': 0,
                'max_interval': 0,
                'total_span_hours': 0
            }
        
        session_data = {
            'metadata': {
                'version': self.config.version,
                'system_name': self.config.system_name,
                'language': language,
                'language_code': LANGUAGE_CONFIG[language]['code'],
                'script': LANGUAGE_CONFIG[language]['script'],
                'model': self.model,
                'session_id': self.session_id,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'successful_calls': successful_calls,
                'failed_calls': failed_calls,
                'total_calls': total_calls,
                'scenario_type': self.config.scenario_type,
                'sampling_config': {
                    'temperature': self.config.temperature,
                    'top_p': self.config.top_p,
                    'presence_penalty': self.config.presence_penalty,
                    'frequency_penalty': self.config.frequency_penalty,
                    'batch_size': self.config.batch_size,
                    'use_system_prompt': self.config.use_system_prompt,
                    'use_markers': self.config.use_markers,
                    'use_variants': self.config.use_variants
                }
            },
            'temporal_statistics': temporal_stats,
            'generation_statistics': self.generation_stats[language],
            'optimization_info': {
                'version': self.config.version,
                'improvements': [
                    'Enhanced temperature (1.1) for increased creativity',
                    'Wider top_p (0.98) for richer vocabulary access',
                    'Pure immediate scenario (no system prompt)',
                    'No artificial markers or variants',
                    'Refined idiomatic translations'
                ],
                'expected_improvements': [
                    '+134% average dream length',
                    '+87% vocabulary richness',
                    '100% AI disclaimer elimination',
                    'Enhanced cultural authenticity'
                ]
            },
            'api_calls': self.api_calls_data[language],
            'dreams': self.dreams_data[language]
        }
        
        session_file = os.path.join(lang_dir, "session_data.json")
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
        
        print(f"    💾 Saved {language} data: {total_calls} API calls, {len(self.dreams_data[language])} dreams")
    
    async def save_global_summary(self, language_results: List[Dict]):
        """Save global session summary"""
        
        # Combine all data
        all_api_calls = []
        all_dreams = []
        
        for language_calls in self.api_calls_data.values():
            all_api_calls.extend(language_calls)
        
        for language_dreams in self.dreams_data.values():
            all_dreams.extend(language_dreams)
        
        # Save combined CSV files
        if all_api_calls:
            all_calls_file = f"{self.base_logs_dir}/all_api_calls_{self.session_id}.csv"
            pd.DataFrame(all_api_calls).to_csv(all_calls_file, index=False, encoding='utf-8')
        
        if all_dreams:
            all_dreams_file = f"{self.base_logs_dir}/all_dreams_{self.session_id}.csv"
            pd.DataFrame(all_dreams).to_csv(all_dreams_file, index=False, encoding='utf-8')
        
        # Calculate global statistics
        total_successful = sum(result['successful_dreams'] for result in language_results)
        total_generated = sum(result['successful_dreams'] + result['failed_dreams'] for result in language_results)
        global_success_rate = (total_successful / total_generated) * 100 if total_generated > 0 else 0
        
        avg_chars_global = np.mean([result['avg_chars'] for result in language_results if 'avg_chars' in result])
        avg_words_global = np.mean([result['avg_words'] for result in language_results if 'avg_words' in result])
        
        total_duration = time.time() - self.start_time
        
        # Create comprehensive summary
        summary = {
            'session_info': {
                'session_id': self.session_id,
                'version': self.config.version,
                'system_name': self.config.system_name,
                'generated_at': datetime.now().isoformat(),
                'total_duration_minutes': total_duration / 60,
                'scenario_type': self.config.scenario_type
            },
            'targets': {
                'dreams_per_language': self.config.dreams_per_language,
                'total_languages': len(LANGUAGE_CONFIG),
                'total_target_dreams': self.config.total_target_dreams
            },
            'results': {
                'total_dreams_generated': total_generated,
                'total_successful_dreams': total_successful,
                'global_success_rate': global_success_rate,
                'avg_chars_per_dream': avg_chars_global,
                'avg_words_per_dream': avg_words_global
            },
            'configuration': {
                'temperature': self.config.temperature,
                'top_p': self.config.top_p,
                'presence_penalty': self.config.presence_penalty,
                'frequency_penalty': self.config.frequency_penalty,
                'use_system_prompt': self.config.use_system_prompt,
                'use_markers': self.config.use_markers,
                'use_variants': self.config.use_variants
            },
            'language_results': {
                result['language']: {
                    'successful_dreams': result['successful_dreams'],
                    'success_rate': result['success_rate'],
                    'avg_chars': result['avg_chars'],
                    'avg_words': result['avg_words'],
                    'batch_duration': result['batch_duration']
                }
                for result in language_results
            },
            'optimization_notes': {
                'version': f"Optimized Batch Generator v{self.config.version}",
                'key_improvements': [
                    'Pure immediate dream scenario',
                    'Enhanced temperature and top_p parameters',
                    'Refined idiomatic translations',
                    'No artificial system prompts or markers'
                ],
                'compatibility': 'Fully compatible with existing analysis infrastructure'
            }
        }
        
        summary_file = f"{self.base_logs_dir}/session_summary_{self.session_id}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Global summary saved: {summary_file}")
        return summary
    
    async def generate_all_languages(self) -> Dict[str, Any]:
        """Generate optimized dreams for all languages with interruption handling"""
        
        print(f"🚀 OPTIMIZED BATCH GENERATION v{self.config.version}")
        print(f"🎯 Generating {self.config.total_target_dreams} dreams across {len(LANGUAGE_CONFIG)} languages")
        print(f"📁 Logs directory: {self.base_logs_dir}/")
        print(f"{'='*80}")
        
        language_results = []
        
        # Check overall progress
        total_completed = 0
        for language in LANGUAGE_CONFIG.keys():
            progress = self.get_language_progress(language)
            total_completed += progress['completed']
        
        print(f"📊 Overall progress: {total_completed}/{self.config.total_target_dreams} dreams")
        
        if total_completed >= self.config.total_target_dreams:
            print(f"🎉 All languages already complete!")
            return await self.generate_completion_summary()
        
        try:
            for i, language in enumerate(LANGUAGE_CONFIG.keys(), 1):
                print(f"\n🌍 Language {i}/{len(LANGUAGE_CONFIG)}: {language.upper()}")
                print(f"{'='*60}")
                
                try:
                    result = await self.generate_language_batch(language)
                    language_results.append(result)
                    
                    # Save progress after each language
                    await self.save_global_summary(language_results)
                    
                    if result.get('resumed'):
                        print(f"✅ {language.upper()} completed (resumed from previous session)")
                    else:
                        print(f"✅ {language.upper()} completed successfully")
                        
                except KeyboardInterrupt:
                    print(f"\n🛑 Generation interrupted during {language}")
                    print(f"💾 Saving current progress...")
                    
                    # Save what we have so far
                    if language_results:
                        await self.save_global_summary(language_results)
                    
                    print(f"📊 Progress saved. You can resume later by running the script again.")
                    print(f"💡 Session ID: {self.session_id}")
                    return {
                        'session_id': self.session_id,
                        'status': 'interrupted',
                        'completed_languages': len(language_results),
                        'total_languages': len(LANGUAGE_CONFIG),
                        'language_results': language_results
                    }
                    
                except Exception as e:
                    print(f"❌ Error in {language}: {e}")
                    print(f"⏭️  Continuing with next language...")
                    continue
        
        except KeyboardInterrupt:
            print(f"\n🛑 Generation interrupted")
            print(f"💾 Saving progress...")
            await self.save_global_summary(language_results)
            return {
                'session_id': self.session_id,
                'status': 'interrupted',
                'completed_languages': len(language_results),
                'total_languages': len(LANGUAGE_CONFIG),
                'language_results': language_results
            }
        
        # Generate final summary
        return await self.generate_completion_summary(language_results)
    
    async def generate_completion_summary(self, language_results: List[Dict] = None) -> Dict[str, Any]:
        """Generate completion summary with all results"""
        
        if language_results is None:
            # Generate results from existing data
            language_results = []
            for language in LANGUAGE_CONFIG.keys():
                progress = self.get_language_progress(language)
                language_results.append({
                    'language': language,
                    'successful_dreams': progress['completed'],
                    'failed_dreams': 0,
                    'success_rate': 100.0 if progress['completed'] >= progress['target'] else 0.0,
                    'avg_chars': 0,
                    'avg_words': 0,
                    'batch_duration': 0,
                    'avg_dream_duration': 0,
                    'resumed': True
                })
        
        # Save final global summary
        summary = await self.save_global_summary(language_results)
        
        # Final report
        print(f"\n{'='*80}")
        print(f"🎉 OPTIMIZED BATCH v{self.config.version} COMPLETE!")
        print(f"{'='*80}")
        print(f"📊 Session: {self.session_id}")
        print(f"🎯 Dreams Generated: {summary['results']['total_successful_dreams']}/{summary['targets']['total_target_dreams']}")
        print(f"✅ Success Rate: {summary['results']['global_success_rate']:.1f}%")
        print(f"📏 Average Quality: {summary['results']['avg_chars_per_dream']:.0f} chars, {summary['results']['avg_words_per_dream']:.0f} words")
        print(f"⏱️  Total Duration: {summary['session_info']['total_duration_minutes']:.1f} minutes")
        print(f"📁 Data saved to: {self.base_logs_dir}/")
        
        # Language breakdown
        print(f"\n📊 Language Breakdown:")
        for result in language_results:
            status = "✅ Complete" if result['successful_dreams'] >= self.config.dreams_per_language else "⏳ Incomplete"
            resumed = " (Resumed)" if result.get('resumed') else ""
            print(f"  {result['language'].upper()}: {result['successful_dreams']}/{self.config.dreams_per_language} dreams {status}{resumed}")
        
        print(f"\n🔍 Next Steps:")
        print(f"1. Analyze results: python analyze_optimized_v2.py {self.session_id}")
        print(f"2. Run thematic analysis on this enhanced data")
        print(f"3. Compare with previous batch results")
        print(f"4. Integrate with existing research pipeline")
        
        return summary

async def main():
    """Main execution function for Optimized Batch v2.0"""
    
    # Check API keys
    import os
    api_keys = {
        'openai': os.getenv('OPENAI_API_KEY')
    }
    
    if not api_keys['openai']:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    print("🔑 API key detected")
    
    # Create and run generator
    generator = OptimizedBatchV2(api_keys, model='gpt-4o')
    
    # Confirm before starting
    print(f"\n📋 Configuration Summary:")
    print(f"   • Version: {generator.config.version}")
    print(f"   • Dreams per language: {generator.config.dreams_per_language}")
    print(f"   • Total dreams: {generator.config.total_target_dreams}")
    print(f"   • Temperature: {generator.config.temperature}")
    print(f"   • Top-p: {generator.config.top_p}")
    print(f"   • Scenario: {generator.config.scenario_type}")
    print(f"   • Session: {generator.session_id}")
    
    input("\nPress Enter to start generation (Ctrl+C to cancel)...")
    
    # Run generation
    try:
        summary = await generator.generate_all_languages()
        
        print(f"\n🎊 SUCCESS! Optimized Batch v{generator.config.version} completed successfully!")
        print(f"📁 All data saved to: {generator.base_logs_dir}/")
        print(f"🆔 Session ID: {generator.session_id}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Generation interrupted by user")
        print("Partial data may be saved in the logs directory")
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        print("Check the logs directory for any partial data")

if __name__ == "__main__":
    asyncio.run(main()) 