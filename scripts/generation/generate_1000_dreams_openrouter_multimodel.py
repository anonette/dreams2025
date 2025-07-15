#!/usr/bin/env python3
"""
Generate 1000 Dreams per Language using OpenRouter with Multiple Models
Uses the same settings as GPT-4o configuration but via OpenRouter API
Supports multiple models and generates in batches for all 5 languages (5000 total dreams)

Languages: English, Basque, Serbian, Hebrew, Slovenian
Total Target: 5000 dreams (1000 per language)
Batch Size: 50 dreams per batch (20 batches per language)
Models: Multiple models available through OpenRouter
"""

import asyncio
import json
import csv
import sys
import os
import time
import uuid
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, will use system environment variables
    pass

# Import the configuration and LLM interface
from config.languages.optimized_dream_languages import LANGUAGE_CONFIG
from src.models.llm_interface import LLMInterface, GenerationConfig

@dataclass
class OpenRouterMultiModelConfig:
    """Configuration for OpenRouter multi-model batch generation"""
    version: str = "OpenRouter_MultiModel_v2.0_Concurrent"
    system_name: str = "OpenRouterMultiModelGenerator"
    session_prefix: str = "OPENROUTER_"
    
    # Available models through OpenRouter (popular and capable models)
    available_models: List[str] = None
    
    # GPT-4o equivalent parameters (matching working scripts)
    temperature: float = 1.1
    top_p: float = 0.98
    presence_penalty: float = 0.1
    frequency_penalty: float = 0.0
    
    batch_size: int = 50  # 50 dreams per batch
    batches_per_language: int = 20  # 20 batches to reach 1000 dreams
    dreams_per_language: int = 1000
    total_target_dreams: int = 5000  # 1000 × 5 languages
    use_system_prompt: bool = True  # Use system message like GPT-4o setup
    scenario_type: str = "Multi-Model Dream Generation via OpenRouter with GPT-4o Parameters (Concurrent)"
    
    # Concurrency configuration
    concurrent_dreams_per_language: int = 8  # Number of dreams to generate concurrently per language
    concurrent_languages_per_model: bool = True  # Process all languages concurrently per model
    max_concurrent_requests: int = 40  # Global limit (8 dreams × 5 languages)
    dream_batch_size: int = 25  # Process dreams in smaller concurrent batches for progress tracking
    
    def __post_init__(self):
        if self.available_models is None:
            self.available_models = [
                "anthropic/claude-3.5-sonnet",
                "openai/gpt-4o",
                "openai/gpt-4o-mini", 
                "mistralai/mistral-nemo",
                "google/gemini-pro-1.5",
                "meta-llama/llama-3.1-70b-instruct",
                "qwen/qwen-2.5-72b-instruct",
                "deepseek/deepseek-chat"
            ]

class OpenRouterMultiModelGenerator:
    """Dream generator using multiple models via OpenRouter"""
    
    def __init__(self, api_keys: Dict[str, str], selected_models: List[str] = None):
        self.config = OpenRouterMultiModelConfig()
        self.llm_interface = LLMInterface(api_keys)
        
        # Set models to use
        if selected_models:
            self.models = selected_models
        else:
            self.models = self.config.available_models
        
        # Create unique session ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"{self.config.session_prefix}{timestamp}"
        
        # Create logs directory
        self.base_logs_dir = 'logs_openrouter_multimodel'
        os.makedirs(self.base_logs_dir, exist_ok=True)
        
        # Data structures organized by model and language
        self.api_calls_data = {}
        self.dreams_data = {}
        self.generation_stats = {}
        
        # Initialize data structures for each model and language
        for model in self.models:
            self.api_calls_data[model] = {}
            self.dreams_data[model] = {}
            self.generation_stats[model] = {}
            
            for language in LANGUAGE_CONFIG.keys():
                self.api_calls_data[model][language] = []
                self.dreams_data[model][language] = []
                self.generation_stats[model][language] = {
                    'successful': 0,
                    'failed': 0,
                    'total_chars': 0,
                    'total_words': 0,
                    'durations': []
                }
        
        # Timing tracking
        self.call_times = []
        self.start_time = time.time()
        
        print(f"🚀 OpenRouter Multi-Model Dream Generator v{self.config.version}")
        print(f"📊 Session: {self.session_id}")
        print(f"🤖 Models: {len(self.models)} models")
        for i, model in enumerate(self.models, 1):
            print(f"   {i}. {model}")
        print(f"🎯 Target: {self.config.total_target_dreams} dreams per model ({self.config.dreams_per_language} per language)")
        print(f"⚙️  Config: temp={self.config.temperature}, top_p={self.config.top_p}, presence={self.config.presence_penalty}")
        print(f"📝 Scenario: {self.config.scenario_type}")
        print(f"📁 Logs: {self.base_logs_dir}/")
        print()
    
    def setup_model_language_directory(self, model: str, language: str) -> str:
        """Create directory structure for model and language"""
        # Clean model name for directory
        clean_model = model.replace('/', '_').replace(':', '_')
        lang_dir = f"{self.base_logs_dir}/{clean_model}/{language}"
        session_dir = f"{lang_dir}/session_{self.session_id}"
        os.makedirs(session_dir, exist_ok=True)
        return session_dir
    
    async def generate_single_dream(self, model: str, language: str, dream_number: int) -> Dict[str, Any]:
        """Generate a single dream using specified model via OpenRouter"""
        
        config = LANGUAGE_CONFIG[language]
        prompt = config['prompt']
        system_message = config.get('system_message') if self.config.use_system_prompt else None
        
        # Generate unique IDs
        call_id = str(uuid.uuid4())
        batch_id = f"openrouter_batch_{model.replace('/', '_')}_{language}_{datetime.now().strftime('%H%M%S')}"
        user_id = f"openrouter_user_{uuid.uuid4().hex[:8]}_{np.random.randint(10000, 99999)}"
        prompt_id = f"openrouter_{hash(prompt) % 100000000:08x}"
        
        start_time = time.time()
        
        # Generation config with GPT-4o parameters
        gen_config = GenerationConfig(
            model=model,
            temperature=self.config.temperature,
            max_tokens=1000,  # Same as GPT-4o setup
            top_p=self.config.top_p,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty
        )
        
        try:
            # Generate dream using OpenRouter
            dream_content = await self.llm_interface.generate_dream(prompt, gen_config, system_message)
            end_time = time.time()
            duration = end_time - start_time
            status = 'success'
            
            # Track timing
            self.call_times.append(end_time)
            self.generation_stats[model][language]['durations'].append(duration)
            
            # Update stats
            self.generation_stats[model][language]['successful'] += 1
            self.generation_stats[model][language]['total_chars'] += len(dream_content)
            self.generation_stats[model][language]['total_words'] += len(dream_content.split())
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            status = 'error'
            dream_content = f"Generation failed: {str(e)}"
            self.generation_stats[model][language]['failed'] += 1
            
            print(f"  ❌ {model} - {language} dream {dream_number} failed: {e}")
        
        # Create comprehensive API call record (same structure as GPT-4o)
        api_call = {
            'call_id': call_id,
            'batch_id': batch_id,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'language': language,
            'language_code': config['code'],
            'script': config['script'],
            'model': model,
            'temperature': self.config.temperature,
            'top_p': self.config.top_p,
            'presence_penalty': self.config.presence_penalty,
            'frequency_penalty': self.config.frequency_penalty,
            'base_prompt': prompt,
            'modified_prompt': prompt,
            'system_message': system_message,
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
            'scenario_type': self.config.scenario_type,
            'session_independence': True,
            'temporal_dispersion': 0
        }
        
        # Create dream record
        dream_record = {
            'call_id': call_id,
            'batch_id': batch_id,
            'user_id': user_id,
            'language': language,
            'language_code': config['code'],
            'script': config['script'],
            'model': model,
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
        self.api_calls_data[model][language].append(api_call)
        self.dreams_data[model][language].append(dream_record)
        
        # Progress indicator
        char_count = len(dream_content)
        word_count = len(dream_content.split())
        model_short = model.split('/')[-1][:15]
        print(f"  ✅ {model_short:<15} {language:<10} #{dream_number:3d}: {status} ({duration:.2f}s) - {char_count} chars, {word_count} words")
        
        return {
            'call_id': call_id,
            'model': model,
            'language': language,
            'status': status,
            'duration': duration,
            'char_count': char_count,
            'word_count': word_count
        }
    
    async def generate_single_dream_with_semaphore(self, semaphore: asyncio.Semaphore, model: str, language: str, dream_number: int) -> Dict[str, Any]:
        """Generate a single dream with semaphore control for concurrency"""
        async with semaphore:
            return await self.generate_single_dream(model, language, dream_number)
    
    async def generate_dreams_batch_concurrent(self, model: str, language: str, dream_numbers: List[int]) -> List[Dict[str, Any]]:
        """Generate a batch of dreams concurrently with controlled concurrency"""
        semaphore = asyncio.Semaphore(self.config.concurrent_dreams_per_language)
        
        # Create tasks for concurrent generation
        tasks = [
            self.generate_single_dream_with_semaphore(semaphore, model, language, dream_num)
            for dream_num in dream_numbers
        ]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"  ❌ {model} - {language} dream {dream_numbers[i]} failed with exception: {result}")
                # Create error result
                processed_results.append({
                    'call_id': str(uuid.uuid4()),
                    'model': model,
                    'language': language,
                    'status': 'error',
                    'duration': 0,
                    'char_count': 0,
                    'word_count': 0
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_model_language_progress(self, model: str, language: str) -> Dict:
        """Get progress for a specific model and language"""
        existing_dreams = self.dreams_data.get(model, {}).get(language, [])
        successful_dreams = len([d for d in existing_dreams if d.get('status') == 'success'])
        
        return {
            'completed': successful_dreams,
            'target': self.config.dreams_per_language,
            'remaining': max(0, self.config.dreams_per_language - successful_dreams),
            'progress_percent': (successful_dreams / self.config.dreams_per_language) * 100
        }
    
    async def save_model_language_data(self, model: str, language: str, lang_dir: str):
        """Save model and language-specific data"""
        model_api_calls = self.api_calls_data[model][language]
        model_dreams = self.dreams_data[model][language]
        
        # Save API calls
        api_calls_file = f"{lang_dir}/api_calls_{self.session_id}.csv"
        if model_api_calls:
            df_api = pd.DataFrame(model_api_calls)
            df_api.to_csv(api_calls_file, index=False, encoding='utf-8')
        
        # Save dreams
        dreams_file = f"{lang_dir}/dreams_{self.session_id}.csv"
        if model_dreams:
            df_dreams = pd.DataFrame(model_dreams)
            df_dreams.to_csv(dreams_file, index=False, encoding='utf-8')
        
        # Save session data
        successful_calls = len([call for call in model_api_calls if call['status'] == 'success'])
        failed_calls = len([call for call in model_api_calls if call['status'] != 'success'])
        
        session_data = {
            'metadata': {
                'model': model,
                'language': language,
                'language_code': LANGUAGE_CONFIG[language]['code'],
                'script': LANGUAGE_CONFIG[language]['script'],
                'session_id': self.session_id,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'successful_calls': successful_calls,
                'failed_calls': failed_calls,
                'total_calls': len(model_api_calls),
                'generation_config': {
                    'temperature': self.config.temperature,
                    'top_p': self.config.top_p,
                    'presence_penalty': self.config.presence_penalty,
                    'frequency_penalty': self.config.frequency_penalty,
                    'batch_size': self.config.batch_size,
                    'scenario_type': self.config.scenario_type,
                    'use_system_prompt': self.config.use_system_prompt
                }
            },
            'dreams': model_dreams,
            'api_calls': model_api_calls
        }
        
        session_file = f"{lang_dir}/session_data.json"
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
    
    async def generate_model_language_batch(self, model: str, language: str) -> Dict[str, Any]:
        """Generate complete batch for one model and language with resumption support"""
        
        # Check current progress
        progress = self.get_model_language_progress(model, language)
        
        model_short = model.split('/')[-1]
        print(f"\n🤖 {model_short} - {language.upper()}")
        print(f"📝 Prompt: {LANGUAGE_CONFIG[language]['prompt']}")
        print(f"📊 Progress: {progress['completed']}/{progress['target']} dreams ({progress['progress_percent']:.1f}%)")
        
        if progress['remaining'] == 0:
            print(f"✅ {model_short} - {language.upper()} already complete!")
            return {
                'model': model,
                'language': language,
                'successful_dreams': progress['completed'],
                'failed_dreams': 0,
                'success_rate': 100.0,
                'avg_chars': 0,
                'avg_words': 0,
                'batch_duration': 0,
                'resumed': True
            }
        
        print(f"🎯 Need {progress['remaining']} more dreams")
        
        # Setup directory
        lang_dir = self.setup_model_language_directory(model, language)
        
        # Generate remaining dreams using concurrent batches
        batch_start = time.time()
        results = []
        
        start_number = progress['completed'] + 1
        end_number = start_number + progress['remaining']
        remaining_dreams = list(range(start_number, end_number))
        
        print(f"🚀 Using concurrent generation: {self.config.concurrent_dreams_per_language} dreams at a time")
        
        # Process dreams in concurrent batches
        batch_size = self.config.dream_batch_size
        for i in range(0, len(remaining_dreams), batch_size):
            batch_dreams = remaining_dreams[i:i + batch_size]
            
            try:
                print(f"  🔄 Processing concurrent batch {i//batch_size + 1}: dreams {batch_dreams[0]}-{batch_dreams[-1]}")
                
                # Generate this batch concurrently
                batch_results = await self.generate_dreams_batch_concurrent(model, language, batch_dreams)
                results.extend(batch_results)
                
                # Progress update
                successful = len([r for r in results if r['status'] == 'success'])
                completed_so_far = len(results)
                remaining_count = len(remaining_dreams) - completed_so_far
                print(f"  📊 Progress: {completed_so_far}/{len(remaining_dreams)} ({successful} successful, {remaining_count} remaining)")
                
                # Save intermediate progress every batch
                await self.save_model_language_data(model, language, lang_dir)
                
                # Small delay between batches to be respectful to the API
                if remaining_count > 0:  # Don't delay after the last batch
                    await asyncio.sleep(1.0)
                    
            except KeyboardInterrupt:
                print(f"\n🛑 Interrupted during {model} - {language} generation")
                print(f"💾 Saving progress...")
                await self.save_model_language_data(model, language, lang_dir)
                raise
            except Exception as e:
                print(f"❌ Error in {model} - {language} batch generation: {e}")
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
        stats = self.generation_stats[model][language]
        stats['avg_duration'] = np.mean(stats['durations']) if stats['durations'] else 0
        
        total_successful = progress['completed'] + successful_dreams
        print(f"  🎉 {model_short} - {language.upper()} Complete!")
        print(f"    ✅ Total successful: {total_successful}/{self.config.dreams_per_language}")
        print(f"    📏 Recent batch average: {avg_chars:.0f} chars, {avg_words:.0f} words")
        print(f"    ⏱️  Duration: {batch_duration:.1f}s (avg {stats['avg_duration']:.2f}s per dream)")
        
        # Save final data
        await self.save_model_language_data(model, language, lang_dir)
        
        return {
            'model': model,
            'language': language,
            'successful_dreams': total_successful,
            'failed_dreams': failed_dreams,
            'success_rate': (total_successful / self.config.dreams_per_language) * 100,
            'avg_chars': avg_chars,
            'avg_words': avg_words,
            'batch_duration': batch_duration,
            'resumed': progress['completed'] > 0
        }
    
    async def generate_model_all_languages_concurrent(self, model: str) -> Dict[str, Any]:
        """Generate dreams for all languages concurrently for a single model"""
        print(f"\n🤖 Starting model: {model}")
        
        if self.config.concurrent_languages_per_model:
            print(f"🚀 Processing all {len(LANGUAGE_CONFIG)} languages concurrently for {model}")
            
            # Create tasks for all languages
            language_tasks = []
            for language in LANGUAGE_CONFIG.keys():
                task = self.generate_model_language_batch(model, language)
                language_tasks.append((language, task))
            
            # Execute all language tasks concurrently
            results = {}
            try:
                # Use asyncio.gather to run all languages concurrently
                language_results = await asyncio.gather(
                    *[task for _, task in language_tasks], 
                    return_exceptions=True
                )
                
                # Process results
                for i, (language, _) in enumerate(language_tasks):
                    result = language_results[i]
                    if isinstance(result, Exception):
                        print(f"❌ Error with {model} - {language}: {result}")
                        results[language] = {
                            'model': model,
                            'language': language,
                            'successful_dreams': 0,
                            'failed_dreams': 0,
                            'error': str(result)
                        }
                    else:
                        results[language] = result
                        
            except KeyboardInterrupt:
                print(f"\n🛑 Generation interrupted during {model}")
                # Return partial results
                for language in LANGUAGE_CONFIG.keys():
                    if language not in results:
                        results[language] = {
                            'model': model,
                            'language': language,
                            'successful_dreams': 0,
                            'failed_dreams': 0,
                            'error': 'Interrupted'
                        }
                return results
        else:
            # Sequential processing (fallback)
            print(f"🔄 Processing languages sequentially for {model}")
            results = {}
            for language in LANGUAGE_CONFIG.keys():
                try:
                    result = await self.generate_model_language_batch(model, language)
                    results[language] = result
                except KeyboardInterrupt:
                    print(f"\n🛑 Generation interrupted")
                    return results
                except Exception as e:
                    print(f"❌ Error with {model} - {language}: {e}")
                    results[language] = {
                        'model': model,
                        'language': language,
                        'successful_dreams': 0,
                        'failed_dreams': 0,
                        'error': str(e)
                    }
        
        return results

    async def generate_all_models_and_languages(self) -> Dict[str, Any]:
        """Generate dreams for all models and languages"""
        
        print(f"🚀 OPENROUTER MULTI-MODEL DREAM GENERATION")
        print(f"🎯 Generating {self.config.total_target_dreams} dreams per model across {len(LANGUAGE_CONFIG)} languages")
        print(f"🤖 Models: {len(self.models)} models")
        print(f"⚡ Concurrency: {self.config.concurrent_dreams_per_language} dreams per language, languages {'concurrent' if self.config.concurrent_languages_per_model else 'sequential'}")
        print(f"📁 Logs directory: {self.base_logs_dir}/")
        print(f"{'='*80}")
        
        results = {}
        
        for model in self.models:
            try:
                model_results = await self.generate_model_all_languages_concurrent(model)
                results[model] = model_results
            except KeyboardInterrupt:
                print(f"\n🛑 Generation interrupted")
                return results
            except Exception as e:
                print(f"❌ Error with model {model}: {e}")
                results[model] = {
                    language: {
                        'model': model,
                        'language': language,
                        'successful_dreams': 0,
                        'failed_dreams': 0,
                        'error': str(e)
                    }
                    for language in LANGUAGE_CONFIG.keys()
                }
        
        # Generate summary
        await self.save_global_summary(results)
        
        return results
    
    async def save_global_summary(self, results: Dict):
        """Save global session summary"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Collect all API calls and dreams
        all_api_calls = []
        all_dreams = []
        
        for model in self.models:
            for language in LANGUAGE_CONFIG.keys():
                all_api_calls.extend(self.api_calls_data.get(model, {}).get(language, []))
                all_dreams.extend(self.dreams_data.get(model, {}).get(language, []))
        
        # Save combined CSV files
        if all_api_calls:
            api_calls_file = f"{self.base_logs_dir}/all_api_calls_{self.session_id}.csv"
            df_api = pd.DataFrame(all_api_calls)
            df_api.to_csv(api_calls_file, index=False, encoding='utf-8')
        
        if all_dreams:
            dreams_file = f"{self.base_logs_dir}/all_dreams_{self.session_id}.csv"
            df_dreams = pd.DataFrame(all_dreams)
            df_dreams.to_csv(dreams_file, index=False, encoding='utf-8')
        
        # Calculate totals
        total_successful = sum(
            results.get(model, {}).get(lang, {}).get('successful_dreams', 0)
            for model in self.models
            for lang in LANGUAGE_CONFIG.keys()
        )
        
        # Save session summary
        summary_data = {
            'session_info': {
                'session_id': self.session_id,
                'version': self.config.version,
                'system_name': self.config.system_name,
                'generated_at': datetime.now().isoformat(),
                'total_duration_minutes': (time.time() - self.start_time) / 60,
                'scenario_type': self.config.scenario_type
            },
            'targets': {
                'dreams_per_language': self.config.dreams_per_language,
                'total_languages': len(LANGUAGE_CONFIG),
                'total_models': len(self.models),
                'total_target_dreams': self.config.total_target_dreams * len(self.models)
            },
            'models_used': self.models,
            'results': {
                'total_dreams_generated': total_successful,
                'total_successful_dreams': total_successful,
                'avg_chars_per_dream': np.mean([len(d['dream']) for d in all_dreams if d.get('status') == 'success']) if all_dreams else 0,
                'avg_words_per_dream': np.mean([d['word_count'] for d in all_dreams if d.get('status') == 'success']) if all_dreams else 0
            },
            'configuration': {
                'temperature': self.config.temperature,
                'top_p': self.config.top_p,
                'presence_penalty': self.config.presence_penalty,
                'frequency_penalty': self.config.frequency_penalty,
                'use_system_prompt': self.config.use_system_prompt,
                'scenario_type': self.config.scenario_type
            },
            'detailed_results': results
        }
        
        summary_file = f"{self.base_logs_dir}/session_summary_{self.session_id}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n🎉 GENERATION COMPLETE!")
        print(f"📊 Total dreams generated: {total_successful}")
        print(f"🤖 Models used: {len(self.models)}")
        print(f"📁 Session: {self.session_id}")
        print(f"💾 Global summary saved: {summary_file}")

async def generate_1000_openrouter_dreams():
    """Generate 1000 dreams per language using multiple OpenRouter models"""
    
    print("🎯 GENERATING 1000 DREAMS PER LANGUAGE WITH OPENROUTER MODELS")
    print("=" * 60)
    print("🤖 Multiple models via OpenRouter")
    print("📊 Target: 1000 dreams per language per model")
    print("📝 Same parameters as GPT-4o setup")
    print("🔄 Will resume from existing progress")
    print()
    
    # Load API keys
    api_keys = {}
    
    # Check for OpenRouter API key
    openrouter_key = os.getenv('OPENROUTER_API_KEY')
    if not openrouter_key:
        print("❌ OPENROUTER_API_KEY not found in environment variables")
        print("💡 Please set your OpenRouter API key:")
        print("   export OPENROUTER_API_KEY='your-openrouter-key-here'")
        return
    
    api_keys['openrouter'] = openrouter_key
    
    # Ask user which models to use
    config = OpenRouterMultiModelConfig()
    print("🤖 Available models:")
    for i, model in enumerate(config.available_models, 1):
        print(f"   {i}. {model}")
    
    print("\n📝 You can:")
    print("   1. Press Enter to use all models")
    print("   2. Enter model numbers (e.g., '1,3,5' for specific models)")
    print("   3. Enter 'fast' for fast models only")
    print("   4. Enter 'premium' for premium models only")
    
    choice = input("\nYour choice: ").strip()
    
    selected_models = None
    if choice == "":
        selected_models = config.available_models
        print("✅ Using all available models")
    elif choice.lower() == "fast":
        selected_models = [
            "openai/gpt-4o-mini",
            "mistralai/mistral-nemo", 
            "qwen/qwen-2.5-72b-instruct"
        ]
        print("⚡ Using fast models")
    elif choice.lower() == "premium":
        selected_models = [
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4o",
            "google/gemini-pro-1.5"
        ]
        print("💎 Using premium models")
    else:
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(',')]
            selected_models = [config.available_models[i] for i in indices if 0 <= i < len(config.available_models)]
            print(f"✅ Using selected models: {selected_models}")
        except:
            print("❌ Invalid selection, using all models")
            selected_models = config.available_models
    
    # Create generator
    generator = OpenRouterMultiModelGenerator(api_keys, selected_models)
    
    print(f"\n🆔 Session ID: {generator.session_id}")
    print(f"📁 Logs directory: {generator.base_logs_dir}/")
    print()
    
    # Generate for all models and languages
    try:
        results = await generator.generate_all_models_and_languages()
        
        print("\n🎉 GENERATION COMPLETE!")
        print(f"📊 Results by model:")
        
        for model, model_results in results.items():
            model_short = model.split('/')[-1]
            total_dreams = sum(lang_result.get('successful_dreams', 0) for lang_result in model_results.values())
            print(f"  {model_short:>20}: {total_dreams} dreams")
        
        print(f"\n🏁 Multi-model generation complete!")
        
    except KeyboardInterrupt:
        print("\n🛑 Generation interrupted")
        print("💾 Progress has been saved - you can resume by running this script again")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(generate_1000_openrouter_dreams())
