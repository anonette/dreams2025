#!/usr/bin/env python3
"""
Generate 1000 Dreams per Language using OpenRouter (Single Model)
Uses the same settings as GPT-4o configuration but via OpenRouter API
Based on the working generate_1000_dreams_mistral.py structure

Languages: English, Basque, Serbian, Hebrew, Slovenian
Total Target: 5000 dreams (1000 per language)
Batch Size: 50 dreams per batch (20 batches per language)
Model: Configurable OpenRouter model
"""

import asyncio
import os
import json
import csv
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, will use system environment variables
    pass

# Import the configuration and LLM interface
from src.config.languages import LANGUAGE_CONFIG
from src.models.llm_interface import LLMInterface, GenerationConfig

@dataclass
class OpenRouterDreamConfig:
    """Configuration for OpenRouter dream generation (single model)"""
    version: str = "1.0"
    system_name: str = "OpenRouterDreamGenerator"
    session_prefix: str = "OPENROUTER_"
    model: str = "anthropic/claude-3.5-sonnet"  # Default model, can be changed
    
    # GPT-4o equivalent parameters (matching working scripts)
    temperature: float = 1.1
    top_p: float = 0.98
    presence_penalty: float = 0.1
    frequency_penalty: float = 0.0
    
    batch_size: int = 50
    dreams_per_language: int = 1000
    total_target_dreams: int = 5000  # 1000 × 5 languages
    use_system_prompt: bool = True  # Use system message like GPT-4o setup
    scenario_type: str = "Dream Generation via OpenRouter with GPT-4o Parameters"

class OpenRouterDreamGenerator:
    """Dream generator using OpenRouter (single model)"""
    
    def __init__(self, api_keys: Dict[str, str], model: str = None):
        self.config = OpenRouterDreamConfig()
        if model:
            self.config.model = model
            
        self.llm_interface = LLMInterface(api_keys)
        
        # Create unique session ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"{self.config.session_prefix}{timestamp}"
        
        # Create logs directory with model name
        model_name = self.config.model.replace('/', '_').replace(':', '_')
        self.base_logs_dir = f'logs_openrouter_{model_name}'
        os.makedirs(self.base_logs_dir, exist_ok=True)
        
        # Data structures
        self.api_calls_data = {}
        self.dreams_data = {}
        self.generation_stats = {}
        
        # Initialize language data structures
        for language in LANGUAGE_CONFIG.keys():
            self.api_calls_data[language] = []
            self.dreams_data[language] = []
            self.generation_stats[language] = {
                'successful': 0,
                'failed': 0,
                'total_chars': 0,
                'total_words': 0,
                'durations': []
            }
        
        # Timing tracking
        self.call_times = []
        self.start_time = time.time()
        
        print(f"🚀 OpenRouter Dream Generator v{self.config.version}")
        print(f"📊 Session: {self.session_id}")
        print(f"🤖 Model: {self.config.model}")
        print(f"🎯 Target: {self.config.total_target_dreams} dreams ({self.config.dreams_per_language} per language)")
        print(f"⚙️  Config: temp={self.config.temperature}, top_p={self.config.top_p}, presence={self.config.presence_penalty}")
        print(f"📝 Scenario: {self.config.scenario_type}")
        print(f"📁 Logs: {self.base_logs_dir}/")
        print()
    
    def setup_language_directory(self, language: str) -> str:
        """Create directory structure for language"""
        model_name = self.config.model.replace('/', '_').replace(':', '_')
        lang_dir = f"{self.base_logs_dir}/{language}/{model_name}"
        session_dir = f"{lang_dir}/session_{self.session_id}"
        os.makedirs(session_dir, exist_ok=True)
        return session_dir
    
    async def generate_single_dream(self, language: str, dream_number: int) -> Dict[str, Any]:
        """Generate a single dream using OpenRouter"""
        
        config = LANGUAGE_CONFIG[language]
        prompt = config['prompt']
        system_message = config.get('system_message') if self.config.use_system_prompt else None
        
        # Generate unique IDs
        call_id = str(uuid.uuid4())
        batch_id = f"openrouter_batch_{language}_{datetime.now().strftime('%H%M%S')}"
        user_id = f"openrouter_user_{uuid.uuid4().hex[:8]}_{np.random.randint(10000, 99999)}"
        prompt_id = f"openrouter_{hash(prompt) % 100000000:08x}"
        
        start_time = time.time()
        
        # Generation config with GPT-4o parameters
        gen_config = GenerationConfig(
            model=self.config.model,
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
        
        # Create comprehensive API call record (same structure as GPT-4o)
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
            'model': self.config.model,
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
    
    async def save_language_data(self, language: str, lang_dir: str):
        """Save language-specific data"""
        language_api_calls = self.api_calls_data[language]
        language_dreams = self.dreams_data[language]
        
        # Save API calls
        api_calls_file = f"{lang_dir}/api_calls_{self.session_id}.csv"
        if language_api_calls:
            df_api = pd.DataFrame(language_api_calls)
            df_api.to_csv(api_calls_file, index=False, encoding='utf-8')
        
        # Save dreams
        dreams_file = f"{lang_dir}/dreams_{self.session_id}.csv"
        if language_dreams:
            df_dreams = pd.DataFrame(language_dreams)
            df_dreams.to_csv(dreams_file, index=False, encoding='utf-8')
        
        # Save session data
        successful_calls = len([call for call in language_api_calls if call['status'] == 'success'])
        failed_calls = len([call for call in language_api_calls if call['status'] != 'success'])
        
        session_data = {
            'metadata': {
                'language': language,
                'language_code': LANGUAGE_CONFIG[language]['code'],
                'script': LANGUAGE_CONFIG[language]['script'],
                'model': self.config.model,
                'session_id': self.session_id,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'successful_calls': successful_calls,
                'failed_calls': failed_calls,
                'total_calls': len(language_api_calls),
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
            'dreams': language_dreams,
            'api_calls': language_api_calls
        }
        
        session_file = f"{lang_dir}/session_data.json"
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 {language} data saved: {len(language_dreams)} dreams, {len(language_api_calls)} API calls")
    
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
                'resumed': True
            }
        
        print(f"🎯 Need {progress['remaining']} more dreams")
        
        # Setup directory
        lang_dir = self.setup_language_directory(language)
        
        # Generate remaining dreams
        batch_start = time.time()
        results = []
        
        start_number = progress['completed'] + 1
        end_number = start_number + progress['remaining']
        
        for dream_num in range(start_number, end_number):
            try:
                result = await self.generate_single_dream(language, dream_num)
                results.append(result)
                
                # Small delay between requests
                await asyncio.sleep(0.1)
                
                # Progress updates every 50 dreams
                if (dream_num - start_number + 1) % 50 == 0:
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
            'resumed': progress['completed'] > 0
        }
    
    async def generate_all_languages(self) -> Dict[str, Any]:
        """Generate dreams for all languages"""
        
        print(f"🚀 OPENROUTER DREAM GENERATION")
        print(f"🎯 Generating {self.config.total_target_dreams} dreams across {len(LANGUAGE_CONFIG)} languages")
        print(f"🤖 Model: {self.config.model}")
        print(f"📁 Logs directory: {self.base_logs_dir}/")
        print(f"{'='*80}")
        
        results = {}
        
        for language in LANGUAGE_CONFIG.keys():
            try:
                result = await self.generate_language_batch(language)
                results[language] = result
            except KeyboardInterrupt:
                print(f"\n🛑 Generation interrupted")
                break
            except Exception as e:
                print(f"❌ Error with {language}: {e}")
                results[language] = {
                    'language': language,
                    'successful_dreams': 0,
                    'failed_dreams': 0,
                    'error': str(e)
                }
        
        # Generate summary
        total_successful = sum(r.get('successful_dreams', 0) for r in results.values())
        
        print(f"\n🎉 GENERATION COMPLETE!")
        print(f"📊 Total dreams generated: {total_successful}")
        print(f"🤖 Model used: {self.config.model}")
        print(f"📁 Session: {self.session_id}")
        
        # Save global summary
        await self.save_global_summary(results, total_successful)
        
        return results
    
    async def save_global_summary(self, results: Dict, total_successful: int):
        """Save global session summary"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Collect all API calls and dreams
        all_api_calls = []
        all_dreams = []
        
        for language in LANGUAGE_CONFIG.keys():
            all_api_calls.extend(self.api_calls_data.get(language, []))
            all_dreams.extend(self.dreams_data.get(language, []))
        
        # Save combined CSV files
        if all_api_calls:
            api_calls_file = f"{self.base_logs_dir}/all_api_calls_{self.session_id}.csv"
            df_api = pd.DataFrame(all_api_calls)
            df_api.to_csv(api_calls_file, index=False, encoding='utf-8')
        
        if all_dreams:
            dreams_file = f"{self.base_logs_dir}/all_dreams_{self.session_id}.csv"
            df_dreams = pd.DataFrame(all_dreams)
            df_dreams.to_csv(dreams_file, index=False, encoding='utf-8')
        
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
                'total_target_dreams': self.config.total_target_dreams
            },
            'model_used': self.config.model,
            'results': {
                'total_dreams_generated': total_successful,
                'total_successful_dreams': total_successful,
                'global_success_rate': (total_successful / self.config.total_target_dreams) * 100,
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
            'language_results': results
        }
        
        summary_file = f"{self.base_logs_dir}/session_summary_{self.session_id}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Global summary saved: {summary_file}")

async def generate_1000_openrouter_dreams():
    """Generate 1000 dreams per language using OpenRouter"""
    
    print("🎯 GENERATING 1000 DREAMS PER LANGUAGE WITH OPENROUTER")
    print("=" * 50)
    print("📊 Target: 1000 dreams per language (5000 total)")
    print("📝 Same parameters as GPT-4o setup")
    print("🔄 Will resume from existing progress")
    print()
    
    # Available models
    available_models = [
        "anthropic/claude-3.5-sonnet",
        "openai/gpt-4o",
        "openai/gpt-4o-mini", 
        "mistralai/mistral-nemo",
        "google/gemini-pro-1.5",
        "meta-llama/llama-3.1-70b-instruct",
        "qwen/qwen-2.5-72b-instruct",
        "deepseek/deepseek-chat"
    ]
    
    print("🤖 Available models:")
    for i, model in enumerate(available_models, 1):
        print(f"   {i}. {model}")
    
    # Get model selection
    while True:
        try:
            choice = input(f"\nSelect model (1-{len(available_models)}): ").strip()
            model_index = int(choice) - 1
            if 0 <= model_index < len(available_models):
                selected_model = available_models[model_index]
                break
            else:
                print(f"❌ Please enter a number between 1 and {len(available_models)}")
        except ValueError:
            print("❌ Please enter a valid number")
    
    print(f"✅ Selected model: {selected_model}")
    
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
    
    # Create generator
    generator = OpenRouterDreamGenerator(api_keys, selected_model)
    
    print(f"\n🆔 Session ID: {generator.session_id}")
    print(f"📁 Logs directory: {generator.base_logs_dir}/")
    print()
    
    # Generate for all languages
    try:
        results = await generator.generate_all_languages()
        
        print("\n🎉 GENERATION COMPLETE!")
        print(f"📊 Results:")
        
        total_new_dreams = 0
        for lang, result in results.items():
            new_dreams = result.get('successful_dreams', 0)
            total_new_dreams += new_dreams
            print(f"  {lang.title():>10}: {new_dreams} dreams")
        
        print(f"\n✅ Total dreams generated: {total_new_dreams}")
        print(f"🤖 Model: {selected_model}")
        print(f"🏁 Generation complete!")
        
    except KeyboardInterrupt:
        print("\n🛑 Generation interrupted")
        print("💾 Progress has been saved - you can resume by running this script again")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(generate_1000_openrouter_dreams())