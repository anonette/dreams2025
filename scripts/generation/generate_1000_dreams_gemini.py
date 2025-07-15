#!/usr/bin/env python3
"""
Generate 1000 Dreams per Language using Google Gemini 2.5 Flash
Uses Google's AI Platform API (via an OpenAI-compatible interface)
Generates in batches of 100 dreams for all 5 languages (5000 total dreams)

Languages: English, Basque, Serbian, Hebrew, Slovenian
Total Target: 5000 dreams (1000 per language)
Batch Size: 100 dreams per batch (10 batches per language)
Model: Google Gemini 2.5 Flash via Direct API
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
from openai import RateLimitError
from tqdm.asyncio import tqdm, trange

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, will use system environment variables
    pass

# Import the optimized configuration
from optimized_dream_languages import LANGUAGE_CONFIG, get_optimized_config
from src.models.llm_interface import LLMInterface, GenerationConfig

@dataclass
class GeminiV2Config:
    """Configuration for Gemini 2.5 Flash batch generation"""
    version: str = "Gemini_2.5_Flash"
    system_name: str = "GeminiBatchV2"
    session_prefix: str = "GEMINI_"
    model: str = "gemini-2.5-flash"  # Direct Google API model name
    temperature: float = 1.1
    top_p: float = 0.98
    batch_size: int = 100  # 100 dreams per batch
    batches_per_language: int = 10  # 10 batches to reach 1000 dreams
    dreams_per_language: int = 1000
    total_target_dreams: int = 5000  # 1000 × 5 languages
    use_system_prompt: bool = False
    scenario_type: str = "Pure Immediate Dream Writing with Gemini 2.5 Flash Direct API"

class GeminiDreamGenerator:
    """Dream generator using Google Gemini 2.5 Flash via its direct API"""
    
    def __init__(self, api_keys: Dict[str, str]):
        self.config = GeminiV2Config()
        
        # Ensure Google AI API key is available
        if 'gemini' not in api_keys or not api_keys['gemini']:
            raise ValueError("Google Gemini API key is required in api_keys['gemini']")
        
        # Setup modified API keys for Google's OpenAI-compatible endpoint
        modified_api_keys = api_keys.copy()
        modified_api_keys['openai'] = api_keys['gemini']  # Use Gemini key as OpenAI key
        
        self.llm_interface = LLMInterface(modified_api_keys)
        # Override OpenAI client to use Google's endpoint
        self.llm_interface.setup_google_gemini_client()
        
        # Create logs directory for Gemini 2.5 Flash
        self.base_logs_dir = 'logs_gemini_2_5_flash'
        os.makedirs(self.base_logs_dir, exist_ok=True)
        
        # Find existing session or create new one
        self.session_id = self.find_or_create_session()
        
        # Data structures
        self.api_calls_data = {}
        self.dreams_data = {}
        self.batch_tracker = {}
        self.generation_stats = {}
        
        # Initialize data structures for each language
        self.languages = list(LANGUAGE_CONFIG.keys())
        for language in self.languages:
            self.api_calls_data[language] = []
            self.dreams_data[language] = []
            self.generation_stats[language] = {
                'successful_dreams': 0,
                'failed_dreams': 0,
                'total_chars': 0,
                'total_words': 0,
                'total_duration': 0,
                'batches_completed': 0
            }
        
        # Timing tracking
        self.call_times = []
        self.start_time = time.time()
        
        # Checkpoint system
        self.checkpoint_file = f"{self.base_logs_dir}/checkpoint_{self.session_id}.json"
        
        print(f"🚀 Gemini 2.5 Flash Dream Generator v{self.config.version}")
        print(f"📊 Session: {self.session_id}")
        print(f"🎯 Target: {self.config.total_target_dreams} dreams ({self.config.dreams_per_language} per language)")
        print(f"🤖 Model: {self.config.model}")
        print(f"⚙️  Config: temp={self.config.temperature}, top_p={self.config.top_p}")
        print(f"📝 Scenario: {self.config.scenario_type}")
        print(f"📁 Logs: {self.base_logs_dir}/")
        print(f"🔢 Batches: {self.config.batches_per_language} batches of {self.config.batch_size} dreams per language")
        
        # Load existing progress
        self.load_existing_progress()
    
    def find_or_create_session(self) -> str:
        """Find existing incomplete session or create new one"""
        
        # Check for existing checkpoint files
        checkpoint_pattern = f"{self.base_logs_dir}/checkpoint_{self.config.session_prefix}*.json"
        import glob
        existing_checkpoints = glob.glob(checkpoint_pattern)
        
        if existing_checkpoints:
            # Find the most recent checkpoint
            most_recent_checkpoint = max(existing_checkpoints, key=os.path.getmtime)
            
            try:
                with open(most_recent_checkpoint, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                
                session_id = checkpoint_data['session_id']
                
                # Check if this session is incomplete
                total_completed = sum(
                    checkpoint_data['progress'][lang]['successful_dreams'] 
                    for lang in checkpoint_data['progress']
                )
                
                if total_completed < self.config.total_target_dreams:
                    print(f"🔄 Found incomplete session: {session_id}")
                    print(f"📊 Resuming from {total_completed}/{self.config.total_target_dreams} dreams")
                    return session_id
                else:
                    print(f"✅ Previous session {session_id} already completed ({total_completed} dreams)")
                    
            except (json.JSONDecodeError, KeyError) as e:
                print(f"⚠️  Could not read checkpoint file {most_recent_checkpoint}: {e}")
        
        # Check for existing session directories
        for language in LANGUAGE_CONFIG.keys():
            lang_path = os.path.join(self.base_logs_dir, language, 'gemini-2-5-flash')
            if os.path.exists(lang_path):
                session_subdirs = [d for d in os.listdir(lang_path) if d.startswith('session_')]
                if session_subdirs:
                    # Get the most recent session
                    most_recent_session = max(
                        session_subdirs,
                        key=lambda x: os.path.getmtime(os.path.join(lang_path, x))
                    )
                    
                    # Extract session ID from directory name
                    session_id = most_recent_session.replace('session_', '')
                    
                    # Check if this session has incomplete data
                    dreams_file = os.path.join(lang_path, most_recent_session, 'dreams.csv')
                    if os.path.exists(dreams_file):
                        try:
                            df = pd.read_csv(dreams_file)
                            successful_dreams = len(df[df['status'] == 'success'])
                            
                            if successful_dreams < self.config.dreams_per_language:
                                print(f"🔄 Found incomplete session from directory: {session_id}")
                                print(f"📊 {language} has {successful_dreams}/{self.config.dreams_per_language} dreams")
                                return session_id
                                
                        except Exception as e:
                            print(f"⚠️  Could not read dreams file for {language}: {e}")
        
        # Create new session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_session_id = f"{self.config.session_prefix}{timestamp}"
        print(f"🆕 Creating new session: {new_session_id}")
        return new_session_id
    
    def load_existing_progress(self):
        """Load existing session data for resumption"""
        for language in self.languages:
            lang_path = os.path.join(self.base_logs_dir, language, 'gemini-2-5-flash')
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
                            
                            # Update stats
                            successful_dreams = len([d for d in self.dreams_data[language] if d.get('status') == 'success'])
                            self.generation_stats[language]['successful_dreams'] = successful_dreams
                            self.generation_stats[language]['batches_completed'] = successful_dreams // self.config.batch_size
                        except:
                            self.dreams_data[language] = []
    
    def setup_language_directory(self, language: str) -> str:
        """Setup directory structure for a language"""
        lang_dir = os.path.join(self.base_logs_dir, language, 'gemini-2-5-flash', f'session_{self.session_id}')
        os.makedirs(lang_dir, exist_ok=True)
        return lang_dir
    
    async def generate_single_dream(self, language: str, dream_number: int, batch_number: int) -> Dict[str, Any]:
        """Generate a single dream using Google Gemini"""
        
        config = LANGUAGE_CONFIG[language]
        prompt = config['prompt']
        
        # Generate unique IDs
        call_id = str(uuid.uuid4())
        batch_id = f"gemini_batch_{language}_{batch_number}_{datetime.now().strftime('%H%M%S')}"
        user_id = f"gemini_user_{uuid.uuid4().hex[:8]}_{np.random.randint(10000, 99999)}"
        prompt_id = f"gemini_{hash(prompt) % 100000000:08x}"
        
        start_time = time.time()
        
        # Generation config for Gemini
        gen_config = GenerationConfig(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=4096,  # Further increased to prevent truncation
            top_p=self.config.top_p
        )
        
        try:
            # Generate dream using Google's API (no system message for pure scenario)
            dream_content = await self.llm_interface.generate_dream(prompt, gen_config, None)
            end_time = time.time()
            duration = end_time - start_time
            status = 'success'
            
            # Track timing
            self.call_times.append(end_time)
            
        except RateLimitError as e:
            end_time = time.time()
            duration = end_time - start_time
            status = 'error'
            dream_content = f"Error: Rate limit exceeded. {str(e)}"
            print(f"🛑 Rate limit hit for {language} dream {dream_number}. The API is busy. This dream will be marked as failed.")
        
        except Exception as e: # Catch other exceptions
            end_time = time.time()
            duration = end_time - start_time
            status = 'error'
            dream_content = f"Error: {str(e)}"
            print(f"❌ Error generating {language} dream {dream_number}: {e}")
        
        # Create comprehensive API call record
        api_call = {
            'call_id': call_id,
            'batch_id': batch_id,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'language': language,
            'language_code': config['code'],
            'script': config['script'],
            'model': self.config.model,
            'temperature': self.config.temperature,
            'top_p': self.config.top_p,
            'base_prompt': prompt,
            'modified_prompt': prompt,
            'system_message': None,  # Pure scenario
            'prompt_id': prompt_id,
            'marker_info': 'none',
            'used_invisible_markers': False,
            'dream_number': dream_number,
            'batch_number': batch_number,
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
            'model': self.config.model,
            'temperature': self.config.temperature,
            'dream': dream_content,
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'dream_number': dream_number,
            'batch_number': batch_number,
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
        
        # Update stats
        if status == 'success':
            self.generation_stats[language]['successful_dreams'] += 1
            self.generation_stats[language]['total_chars'] += len(dream_content)
            self.generation_stats[language]['total_words'] += len(dream_content.split())
        else:
            self.generation_stats[language]['failed_dreams'] += 1
        
        self.generation_stats[language]['total_duration'] += duration
        
        # Progress indicator
        char_count = len(dream_content)
        word_count = len(dream_content.split())
        # print(f"  ✅ Dream {dream_number}: {status} ({duration:.2f}s) - {char_count} chars, {word_count} words") # Removed for cleaner tqdm output
        
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
        successful_dreams = self.generation_stats[language]['successful_dreams']
        
        return {
            'completed': successful_dreams,
            'target': self.config.dreams_per_language,
            'remaining': max(0, self.config.dreams_per_language - successful_dreams),
            'progress_percent': (successful_dreams / self.config.dreams_per_language) * 100,
            'batches_completed': successful_dreams // self.config.batch_size,
            'batches_remaining': max(0, self.config.batches_per_language - (successful_dreams // self.config.batch_size))
        }
    
    async def generate_language_batch(self, language: str, batch_number: int) -> Dict[str, Any]:
        """Generate one batch (100 dreams) for a specific language"""
        
        # Setup directory
        lang_dir = self.setup_language_directory(language)
        
        # Generate batch
        batch_start = time.time()
        results = []
        
        # Calculate dream numbers for this batch
        start_dream = batch_number * self.config.batch_size + 1
        end_dream = start_dream + self.config.batch_size
        
        # Use trange for a per-batch progress bar
        for dream_num in trange(start_dream, end_dream, desc=f"  {language.title():<10s} Batch {batch_number + 1:2d}", unit=" dream", leave=False):
            result = await self.generate_single_dream(language, dream_num, batch_number + 1)
            results.append(result)
            
            # Add delay to respect Gemini 2.5 Flash rate limits and server load
            # 10 seconds to reduce server pressure during high load periods
            await asyncio.sleep(10)
        
        batch_duration = time.time() - batch_start
        
        # Update batch completion
        self.generation_stats[language]['batches_completed'] = batch_number + 1
        
        # Save progress after each batch
        await self.save_language_data(language, lang_dir)
        
        # Calculate batch stats
        successful_dreams = len([r for r in results if r['status'] == 'success'])
        failed_dreams = len([r for r in results if r['status'] != 'success'])
        success_rate = (successful_dreams / len(results)) * 100 if results else 0
        
        avg_chars = np.mean([r['char_count'] for r in results if r['status'] == 'success']) if successful_dreams > 0 else 0
        avg_words = np.mean([len(str(r.get('dream', '')).split()) for r in results if r['status'] == 'success']) if successful_dreams > 0 else 0
        avg_duration = np.mean([r['duration'] for r in results]) if results else 0
        
        print(f"📊 Batch {batch_number + 1} complete:")
        print(f"  ✅ Success: {successful_dreams}/{len(results)} ({success_rate:.1f}%)")
        print(f"  📈 Avg length: {avg_chars:.0f} chars, {avg_words:.0f} words")
        print(f"  ⏱️  Duration: {batch_duration:.1f}s (avg {avg_duration:.2f}s per dream)")
        
        return {
            'language': language,
            'batch_number': batch_number + 1,
            'successful_dreams': successful_dreams,
            'failed_dreams': failed_dreams,
            'success_rate': success_rate,
            'avg_chars': avg_chars,
            'avg_words': avg_words,
            'batch_duration': batch_duration,
            'avg_dream_duration': avg_duration
        }
    
    async def save_language_data(self, language: str, lang_dir: str):
        """Save data for a specific language"""
        
        # Save API calls (both CSV and JSON)
        if self.api_calls_data[language]:
            api_calls_df = pd.DataFrame(self.api_calls_data[language])
            api_calls_csv = os.path.join(lang_dir, 'api_calls.csv')
            api_calls_json = os.path.join(lang_dir, 'api_calls.json')
            api_calls_df.to_csv(api_calls_csv, index=False, encoding='utf-8')
            with open(api_calls_json, 'w', encoding='utf-8') as f:
                json.dump(self.api_calls_data[language], f, indent=2, ensure_ascii=False)
        
        # Save dreams (both CSV and JSON)
        if self.dreams_data[language]:
            dreams_df = pd.DataFrame(self.dreams_data[language])
            dreams_csv = os.path.join(lang_dir, 'dreams.csv')
            dreams_json = os.path.join(lang_dir, 'dreams.json')
            dreams_df.to_csv(dreams_csv, index=False, encoding='utf-8')
            with open(dreams_json, 'w', encoding='utf-8') as f:
                json.dump(self.dreams_data[language], f, indent=2, ensure_ascii=False)
        
        # Save session data
        successful_calls = len([d for d in self.api_calls_data[language] if d.get('status') == 'success'])
        failed_calls = len([d for d in self.api_calls_data[language] if d.get('status') != 'success'])
        total_calls = len(self.api_calls_data[language])
        
        session_data = {
            'metadata': {
                'version': self.config.version,
                'system_name': self.config.system_name,
                'language': language,
                'language_code': LANGUAGE_CONFIG[language]['code'],
                'script': LANGUAGE_CONFIG[language]['script'],
                'model': self.config.model,
                'session_id': self.session_id,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'successful_calls': successful_calls,
                'failed_calls': failed_calls,
                'total_calls': total_calls,
                'scenario_type': self.config.scenario_type,
                'sampling_config': {
                    'temperature': self.config.temperature,
                    'top_p': self.config.top_p,
                    'batch_size': self.config.batch_size,
                    'use_system_prompt': self.config.use_system_prompt
                }
            },
            'generation_statistics': self.generation_stats[language]
        }
        
        session_file = os.path.join(lang_dir, 'session_data.json')
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
    
    async def generate_all_languages(self) -> Dict[str, Any]:
        """Generate 1000 dreams for each language in batches of 100"""
        
        print(f"🚀 GEMINI 2.5 FLASH BATCH GENERATION")
        print(f"🎯 Generating {self.config.total_target_dreams} dreams across {len(self.languages)} languages")
        print(f"📁 Logs directory: {self.base_logs_dir}/")
        print(f"{'='*80}")
        
        # Check overall progress
        total_completed = sum(self.generation_stats[lang]['successful_dreams'] for lang in self.languages)
        print(f"📊 Overall progress: {total_completed}/{self.config.total_target_dreams} dreams")
        
        # Calculate total batches to run for the progress bar
        total_batches_to_run = 0
        for language in self.languages:
            progress = self.get_language_progress(language)
            total_batches_to_run += progress['batches_remaining']
        
        all_results = []
        
        try:
            with tqdm(total=total_batches_to_run, desc="Overall Progress", unit="batch") as pbar:
                for language in self.languages:
                    progress = self.get_language_progress(language)
                    
                    if progress['remaining'] == 0:
                        continue
                    
                    # Generate remaining batches
                    for batch_num in range(progress['batches_completed'], self.config.batches_per_language):
                        batch_result = await self.generate_language_batch(language, batch_num)
                        all_results.append(batch_result)
                        pbar.update(1)
                        
                        # Save checkpoint after each batch
                        await self.save_checkpoint()
        
        except KeyboardInterrupt:
            print(f"\n🛑 Generation interrupted by user")
            await self.save_checkpoint()
            return await self.generate_completion_summary()
        
        # Generate final summary
        return await self.generate_completion_summary()
    
    async def save_checkpoint(self):
        """Save checkpoint for resumption"""
        checkpoint_data = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'version': self.config.version,
                'model': self.config.model,
                'dreams_per_language': self.config.dreams_per_language,
                'batch_size': self.config.batch_size
            },
            'progress': {
                lang: {
                    'successful_dreams': self.generation_stats[lang]['successful_dreams'],
                    'batches_completed': self.generation_stats[lang]['batches_completed']
                }
                for lang in self.languages
            }
        }
        
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
    
    async def generate_completion_summary(self) -> Dict[str, Any]:
        """Generate final completion summary"""
        
        # Calculate final statistics
        total_successful = sum(self.generation_stats[lang]['successful_dreams'] for lang in self.languages)
        total_failed = sum(self.generation_stats[lang]['failed_dreams'] for lang in self.languages)
        total_calls = total_successful + total_failed
        
        global_success_rate = (total_successful / total_calls * 100) if total_calls > 0 else 0
        
        # Generate summary
        summary = {
            'session_id': self.session_id,
            'model': self.config.model,
            'total_successful_dreams': total_successful,
            'total_failed_dreams': total_failed,
            'total_calls': total_calls,
            'global_success_rate': global_success_rate,
            'target_dreams': self.config.total_target_dreams,
            'completion_percentage': (total_successful / self.config.total_target_dreams) * 100,
            'language_results': {
                lang: {
                    'successful_dreams': self.generation_stats[lang]['successful_dreams'],
                    'failed_dreams': self.generation_stats[lang]['failed_dreams'],
                    'success_rate': (self.generation_stats[lang]['successful_dreams'] / 
                                   (self.generation_stats[lang]['successful_dreams'] + self.generation_stats[lang]['failed_dreams']) * 100)
                                   if (self.generation_stats[lang]['successful_dreams'] + self.generation_stats[lang]['failed_dreams']) > 0 else 0,
                    'avg_chars': self.generation_stats[lang]['total_chars'] / max(1, self.generation_stats[lang]['successful_dreams']),
                    'avg_words': self.generation_stats[lang]['total_words'] / max(1, self.generation_stats[lang]['successful_dreams']),
                    'batches_completed': self.generation_stats[lang]['batches_completed']
                }
                for lang in self.languages
            }
        }
        
        # Save global files with the summary to avoid recursion
        await self.save_global_files(summary)
        
        return summary
    
    async def save_global_files(self, summary: Dict[str, Any] = None):
        """Save global consolidated files"""
        
        # Consolidate all dreams
        all_dreams = []
        all_api_calls = []
        
        for language in self.languages:
            all_dreams.extend(self.dreams_data[language])
            all_api_calls.extend(self.api_calls_data[language])
        
        # Save global dreams file (both CSV and JSON)
        if all_dreams:
            dreams_df = pd.DataFrame(all_dreams)
            dreams_csv = os.path.join(self.base_logs_dir, f'all_dreams_{self.session_id}.csv')
            dreams_json = os.path.join(self.base_logs_dir, f'all_dreams_{self.session_id}.json')
            dreams_df.to_csv(dreams_csv, index=False, encoding='utf-8')
            with open(dreams_json, 'w', encoding='utf-8') as f:
                json.dump(all_dreams, f, indent=2, ensure_ascii=False)
        
        # Save global API calls file (both CSV and JSON)
        if all_api_calls:
            api_calls_df = pd.DataFrame(all_api_calls)
            api_calls_csv = os.path.join(self.base_logs_dir, f'all_api_calls_{self.session_id}.csv')
            api_calls_json = os.path.join(self.base_logs_dir, f'all_api_calls_{self.session_id}.json')
            api_calls_df.to_csv(api_calls_csv, index=False, encoding='utf-8')
            with open(api_calls_json, 'w', encoding='utf-8') as f:
                json.dump(all_api_calls, f, indent=2, ensure_ascii=False)
        
        # Save session summary (only if provided to avoid recursion)
        if summary:
            summary_file = os.path.join(self.base_logs_dir, f'session_summary_{self.session_id}.json')
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

async def main():
    """Main execution function for Gemini 2.5 Flash batch generation"""
    
    # Check API keys
    gemini_key = os.getenv('GEMINI_API_KEY')
    
    if not gemini_key:
        print("❌ Error: GEMINI_API_KEY environment variable not set")
        print("Please set your Google Gemini API key:")
        print("export GEMINI_API_KEY='your-gemini-api-key-here'")
        print()
        print("You can get an API key from: https://ai.google.dev/")
        return
    
    print("🔑 Google Gemini API key detected")
    
    # Prepare API keys
    api_keys = {
        'gemini': gemini_key
    }
    
    # Create and run generator
    try:
        generator = GeminiDreamGenerator(api_keys)
        
        # Confirm before starting
        print(f"\n📋 Configuration Summary:")
        print(f"   • Model: {generator.config.model}")
        print(f"   • Dreams per language: {generator.config.dreams_per_language}")
        print(f"   • Total dreams: {generator.config.total_target_dreams}")
        print(f"   • Batch size: {generator.config.batch_size}")
        print(f"   • Batches per language: {generator.config.batches_per_language}")
        print(f"   • Temperature: {generator.config.temperature}")
        print(f"   • Top-p: {generator.config.top_p}")
        print(f"   • Session: {generator.session_id}")
        
        input("\nPress Enter to start generation (Ctrl+C to cancel)...")
        
        # Run generation
        summary = await generator.generate_all_languages()
        
        print(f"\n🎊 SUCCESS! Gemini 2.5 Flash batch generation completed!")
        print(f"✅ Generated {summary['total_successful_dreams']}/{summary['target_dreams']} dreams")
        print(f"📊 Overall success rate: {summary['global_success_rate']:.1f}%")
        print(f"📁 All data saved to: {generator.base_logs_dir}/")
        print(f"🆔 Session ID: {generator.session_id}")
        
        print(f"\n📊 Language Breakdown:")
        for lang, stats in summary['language_results'].items():
            print(f"  {lang.title():>10}: {stats['successful_dreams']:>4} dreams ({stats['success_rate']:.1f}% success)")
        
    except KeyboardInterrupt:
        print("\n⚠️  Generation interrupted by user")
        print("Progress has been saved and can be resumed by running this script again")
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        print("Check the logs directory for any partial data")

if __name__ == "__main__":
    asyncio.run(main())
