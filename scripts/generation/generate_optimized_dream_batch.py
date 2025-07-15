#!/usr/bin/env python3
"""
Optimized Dream Batch Generator
Generates 100 dreams per language using the refined optimized configuration.
Integrates seamlessly with existing analysis infrastructure.
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
class OptimizedSamplingConfig:
    """Optimized sampling configuration matching existing system structure"""
    temperature: float = 1.1
    top_p: float = 0.98
    presence_penalty: float = 0.1
    frequency_penalty: float = 0.0
    batch_size: int = 50
    use_prompt_variants: bool = False  # Pure scenario - no variants
    invisible_marker_probability: float = 0.0  # Pure scenario - no markers
    use_temporal_dispersion: bool = False
    temporal_dispersion_hours: int = 2
    min_temporal_dispersion_minutes: int = 30
    min_samples_per_language: int = 100
    session_independence: bool = True

class OptimizedBatchDreamGenerator:
    def __init__(self, api_keys: Dict[str, str], model: str = 'gpt-4o'):
        self.llm_interface = LLMInterface(api_keys)
        self.model = model
        self.sampling_config = OptimizedSamplingConfig()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create structured logs directory matching existing system
        self.base_logs_dir = 'logs'
        self.current_language = None
        os.makedirs(self.base_logs_dir, exist_ok=True)
        
        # Initialize data structures for logging (matching existing format)
        self.api_calls_data = {}  # Will be organized by language
        self.dreams_data = {}     # Will be organized by language
        self.batch_tracker = {}
        
        # Global checkpoint system
        self.checkpoint_file = f"{self.base_logs_dir}/checkpoint_{self.session_id}.json"
        self.load_checkpoint()
        
        # Temporal tracking (matching existing system)
        self.call_times = []
        
        print(f"🚀 Optimized Dream Generator initialized")
        print(f"📊 Configuration: temp={self.sampling_config.temperature}, top_p={self.sampling_config.top_p}")
        print(f"🎯 Target: {self.sampling_config.min_samples_per_language} dreams per language")
        print(f"📁 Session: {self.session_id}")
        
    def setup_language_logging(self, language: str):
        """Setup structured logging directories for a specific language and model (matching existing system)."""
        self.current_language = language
        
        # Create structured directory: logs/language/model/session_id/
        self.language_logs_dir = os.path.join(
            self.base_logs_dir, 
            language, 
            self.model, 
            f"session_{self.session_id}"
        )
        os.makedirs(self.language_logs_dir, exist_ok=True)
        
        # Initialize language-specific data structures
        if language not in self.api_calls_data:
            self.api_calls_data[language] = []
        if language not in self.dreams_data:
            self.dreams_data[language] = []
        
        print(f"📁 Setup logging for {language}: {self.language_logs_dir}")
    
    def load_checkpoint(self):
        """Load existing progress from checkpoint"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.api_calls_data = data.get('api_calls_data', {})
                self.dreams_data = data.get('dreams_data', {})
                self.batch_tracker = data.get('batch_tracker', {})
                print(f"📋 Loaded checkpoint with {sum(len(calls) for calls in self.api_calls_data.values())} existing calls")
            except Exception as e:
                print(f"⚠️  Error loading checkpoint: {e}")
                self.api_calls_data = {}
                self.dreams_data = {}
                self.batch_tracker = {}
        else:
            self.api_calls_data = {}
            self.dreams_data = {}
            self.batch_tracker = {}
    
    def save_checkpoint(self):
        """Save current progress to checkpoint"""
        try:
            checkpoint_data = {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'api_calls_data': self.api_calls_data,
                'dreams_data': self.dreams_data,
                'batch_tracker': self.batch_tracker
            }
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️  Error saving checkpoint: {e}")
    
    def get_progress(self, language: str, target_dreams: int):
        """Get progress for a specific language"""
        existing_dreams = len(self.dreams_data.get(language, []))
        successful_dreams = len([d for d in self.dreams_data.get(language, []) if d.get('status') == 'success'])
        
        return {
            'total_dreams': existing_dreams,
            'successful_dreams': successful_dreams,
            'remaining_dreams': max(0, target_dreams - successful_dreams),
            'progress_percent': (successful_dreams / target_dreams) * 100 if target_dreams > 0 else 0
        }
    
    async def generate_dream_with_optimized_config(self, language: str, batch_id: str, 
                                                 dream_number: int) -> Dict:
        """Generate a single dream using optimized configuration"""
        
        config = LANGUAGE_CONFIG[language]
        optimized_prompt = config['prompt']
        
        # Generate unique identifiers (matching existing system)
        call_id = str(uuid.uuid4())
        user_id = f"{str(uuid.uuid4())}_{np.random.randint(10000, 99999)}"
        prompt_id = f"{hash(optimized_prompt) % 100000000:08x}"
        
        start_time = time.time()
        
        # Prepare generation config with optimized parameters
        generation_config = GenerationConfig(
            model=self.model,
            temperature=self.sampling_config.temperature,
            max_tokens=1000,
            top_p=self.sampling_config.top_p,
            frequency_penalty=self.sampling_config.frequency_penalty,
            presence_penalty=self.sampling_config.presence_penalty
        )
        
        try:
            # No system message for pure immediate scenario
            dream = await self.llm_interface.generate_dream(optimized_prompt, generation_config, None)
            end_time = time.time()
            duration = end_time - start_time
            status = 'success'
            dream_content = dream
            
            # Track timing
            self.call_times.append(end_time)
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            status = 'error'
            dream_content = f"Error: {str(e)}"
            print(f"❌ Error generating {language} dream {dream_number}: {e}")
        
        # Log API call with full metadata (matching existing system exactly)
        api_call_data = {
            'call_id': call_id,
            'batch_id': batch_id,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'language': language,
            'language_code': config['code'],
            'script': config['script'],
            'model': self.model,
            'temperature': self.sampling_config.temperature,
            'top_p': self.sampling_config.top_p,
            'presence_penalty': self.sampling_config.presence_penalty,
            'frequency_penalty': self.sampling_config.frequency_penalty,
            'base_prompt': optimized_prompt,
            'modified_prompt': optimized_prompt,  # No modifications in pure scenario
            'system_message': None,  # No system message in pure scenario
            'prompt_id': prompt_id,
            'marker_info': 'none',  # No markers in pure scenario
            'used_invisible_markers': False,
            'dream_number': dream_number,
            'batch_size': self.sampling_config.batch_size,
            'dream': dream_content,
            'status': status,
            'duration_seconds': round(duration, 3),
            'temporal_delay_seconds': 0.1,  # Minimal delay
            'start_time': datetime.fromtimestamp(start_time).isoformat(),
            'end_time': datetime.fromtimestamp(end_time).isoformat(),
            'session_id': self.session_id,
            'temporal_dispersion': self.sampling_config.temporal_dispersion_hours,
            'session_independence': self.sampling_config.session_independence
        }
        
        self.api_calls_data[language].append(api_call_data)
        
        # Log dream data for analysis (matching existing system exactly)
        dream_data = {
            'call_id': call_id,
            'batch_id': batch_id,
            'user_id': user_id,
            'language': language,
            'language_code': config['code'],
            'script': config['script'],
            'model': self.model,
            'temperature': self.sampling_config.temperature,
            'dream': dream_content,
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'dream_number': dream_number,
            'prompt_id': prompt_id,
            'marker_info': 'none',
            'temporal_delay': 0.1
        }
        self.dreams_data[language].append(dream_data)
        
        print(f"  ✅ Dream {dream_number}: {status} ({duration:.2f}s) - {len(dream_content)} chars")
        
        return {
            'call_id': call_id,
            'batch_id': batch_id,
            'user_id': user_id,
            'language': language,
            'dream': dream_content,
            'status': status,
            'duration': duration,
            'prompt_id': prompt_id,
            'temporal_delay': 0.1
        }
    
    async def generate_batch_for_language(self, language: str, batch_size: int = 50) -> List[Dict]:
        """Generate a batch of dreams for a specific language"""
        
        batch_id = str(uuid.uuid4())
        results = []
        successful_calls = 0
        failed_calls = 0
        
        print(f"🔄 Generating batch of {batch_size} dreams for {language}")
        
        for i in range(batch_size):
            dream_number = len(self.dreams_data.get(language, [])) + 1
            
            result = await self.generate_dream_with_optimized_config(language, batch_id, dream_number)
            results.append(result)
            
            if result['status'] == 'success':
                successful_calls += 1
            else:
                failed_calls += 1
            
            # Small delay between requests
            await asyncio.sleep(0.1)
        
        # Track batch statistics
        self.batch_tracker[batch_id] = {
            'language': language,
            'batch_size': batch_size,
            'successful_calls': successful_calls,
            'failed_calls': failed_calls,
            'start_time': datetime.now().isoformat(),
            'session_id': self.session_id
        }
        
        print(f"  📊 Batch complete: {successful_calls} successful, {failed_calls} failed")
        
        return results
    
    async def generate_dreams_for_language(self, language: str, num_dreams: int = 100) -> Dict:
        """Generate dreams for a language using batched approach"""
        
        if language not in LANGUAGE_CONFIG:
            raise ValueError(f"Language '{language}' not found in configuration")
        
        # Setup structured logging for this language
        self.setup_language_logging(language)
        
        # Check current progress
        progress = self.get_progress(language, num_dreams)
        print(f"📊 Progress for {language}: {progress['successful_dreams']}/{num_dreams} dreams ({progress['progress_percent']:.1f}%)")
        
        if progress['successful_dreams'] >= num_dreams:
            print(f"✅ {language} already complete!")
            return {
                'language': language,
                'status': 'complete',
                'dreams_generated': progress['successful_dreams'],
                'target_dreams': num_dreams
            }
        
        remaining = progress['remaining_dreams']
        print(f"🎯 Need {remaining} more dreams for {language}")
        
        # Generate remaining dreams in batches
        batch_size = min(self.sampling_config.batch_size, remaining)
        batches_needed = (remaining + batch_size - 1) // batch_size
        
        for batch_num in range(batches_needed):
            current_batch_size = min(batch_size, remaining - (batch_num * batch_size))
            
            print(f"\n🔄 Batch {batch_num + 1}/{batches_needed} for {language} ({current_batch_size} dreams)")
            
            await self.generate_batch_for_language(language, current_batch_size)
            
            # Save checkpoint after each batch
            self.save_checkpoint()
            
            # Save language-specific logs after each batch
            await self._save_language_logs(language)
        
        final_progress = self.get_progress(language, num_dreams)
        print(f"🎉 {language} complete: {final_progress['successful_dreams']}/{num_dreams} dreams")
        
        return {
            'language': language,
            'status': 'complete',
            'dreams_generated': final_progress['successful_dreams'],
            'target_dreams': num_dreams,
            'batches_completed': batches_needed
        }
    
    async def _save_language_logs(self, language: str):
        """Save detailed logs for the language in structured directories (matching existing system)"""
        
        # Get data for this language
        language_api_calls = self.api_calls_data.get(language, [])
        language_dreams = self.dreams_data.get(language, [])
        
        if not language_api_calls:
            return
        
        # Save API calls CSV (matching existing column structure exactly)
        api_calls_file = os.path.join(self.language_logs_dir, "api_calls.csv")
        api_calls_df = pd.DataFrame(language_api_calls)
        api_calls_df.to_csv(api_calls_file, index=False, encoding='utf-8')
        
        # Save dreams CSV (matching existing column structure exactly)
        dreams_file = os.path.join(self.language_logs_dir, "dreams.csv")
        dreams_df = pd.DataFrame(language_dreams)
        dreams_df.to_csv(dreams_file, index=False, encoding='utf-8')
        
        # Calculate temporal statistics
        if len(self.call_times) > 1:
            intervals = [self.call_times[i] - self.call_times[i-1] for i in range(1, len(self.call_times))]
            temporal_stats = {
                'intervals': intervals,
                'mean_interval': np.mean(intervals) if intervals else 0,
                'std_interval': np.std(intervals) if intervals else 0,
                'min_interval': np.min(intervals) if intervals else 0,
                'max_interval': np.max(intervals) if intervals else 0,
                'total_span_hours': (self.call_times[-1] - self.call_times[0]) / 3600 if len(self.call_times) > 1 else 0
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
        
        # Create session data JSON (matching existing structure exactly)
        successful_calls = len([call for call in language_api_calls if call['status'] == 'success'])
        failed_calls = len([call for call in language_api_calls if call['status'] != 'success'])
        
        session_data = {
            'metadata': {
                'language': language,
                'language_code': LANGUAGE_CONFIG[language]['code'],
                'script': LANGUAGE_CONFIG[language]['script'],
                'model': self.model,
                'session_id': self.session_id,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'successful_calls': successful_calls,
                'failed_calls': failed_calls,
                'total_calls': len(language_api_calls),
                'sampling_config': {
                    'temperature': self.sampling_config.temperature,
                    'top_p': self.sampling_config.top_p,
                    'presence_penalty': self.sampling_config.presence_penalty,
                    'frequency_penalty': self.sampling_config.frequency_penalty,
                    'batch_size': self.sampling_config.batch_size,
                    'use_prompt_variants': self.sampling_config.use_prompt_variants,
                    'invisible_marker_probability': self.sampling_config.invisible_marker_probability,
                    'use_temporal_dispersion': self.sampling_config.use_temporal_dispersion,
                    'temporal_dispersion_hours': self.sampling_config.temporal_dispersion_hours,
                    'min_temporal_dispersion_minutes': self.sampling_config.min_temporal_dispersion_minutes
                }
            },
            'temporal_statistics': temporal_stats,
            'entropy_statistics': {
                'language': language,
                'total_calls': len(language_api_calls),
                'calls_with_markers': 0,  # No markers in optimized config
                'marker_usage_rate': 0.0,
                'marker_type_distribution': {'none': len(language_api_calls)},
                'unique_prompt_ids': len(set(call['prompt_id'] for call in language_api_calls)),
                'configured_marker_probability': 0.0
            },
            'api_calls': language_api_calls,
            'dreams': language_dreams
        }
        
        session_file = os.path.join(self.language_logs_dir, "session_data.json")
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Saved {language} logs: {len(language_api_calls)} API calls, {len(language_dreams)} dreams")
    
    async def _save_global_logs(self):
        """Save global session logs (matching existing system)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Flatten all API calls from all languages
        all_api_calls = []
        for language_calls in self.api_calls_data.values():
            all_api_calls.extend(language_calls)
        
        # Flatten all dreams from all languages
        all_dreams = []
        for language_dreams in self.dreams_data.values():
            all_dreams.extend(language_dreams)
        
        # Save global API calls CSV
        if all_api_calls:
            all_calls_filename = f"{self.base_logs_dir}/all_api_calls_{self.session_id}.csv"
            all_calls_df = pd.DataFrame(all_api_calls)
            all_calls_df.to_csv(all_calls_filename, index=False, encoding='utf-8')
        
        # Save global dreams CSV
        if all_dreams:
            all_dreams_filename = f"{self.base_logs_dir}/all_dreams_{self.session_id}.csv"
            all_dreams_df = pd.DataFrame(all_dreams)
            all_dreams_df.to_csv(all_dreams_filename, index=False, encoding='utf-8')
        
        # Save batch tracker
        batch_filename = f"{self.base_logs_dir}/batch_tracker_{self.session_id}.json"
        with open(batch_filename, 'w', encoding='utf-8') as f:
            json.dump(self.batch_tracker, f, ensure_ascii=False, indent=2)
        
        # Save session summary
        summary_filename = f"{self.base_logs_dir}/session_summary_{self.session_id}.json"
        summary_data = {
            'session_id': self.session_id,
            'timestamp': timestamp,
            'total_languages': len(self.api_calls_data),
            'total_api_calls': len(all_api_calls),
            'total_dreams': len(all_dreams),
            'successful_dreams': len([d for d in all_dreams if d['status'] == 'success']),
            'languages': list(self.api_calls_data.keys()),
            'sampling_config': {
                'temperature': self.sampling_config.temperature,
                'top_p': self.sampling_config.top_p,
                'presence_penalty': self.sampling_config.presence_penalty,
                'frequency_penalty': self.sampling_config.frequency_penalty,
                'batch_size': self.sampling_config.batch_size,
                'session_independence': self.sampling_config.session_independence
            },
            'optimization_note': 'Generated using optimized configuration: no system prompt, temperature=1.1, top_p=0.98'
        }
        
        with open(summary_filename, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Global logs saved: {len(all_api_calls)} total API calls, {len(all_dreams)} total dreams")
    
    async def generate_all_languages(self, dreams_per_language: int = 100) -> Dict:
        """Generate dreams for all configured languages"""
        results = {}
        total_target = dreams_per_language * len(LANGUAGE_CONFIG)
        
        print(f"🚀 Starting optimized dream generation")
        print(f"🎯 Target: {dreams_per_language} dreams × {len(LANGUAGE_CONFIG)} languages = {total_target} total dreams")
        print(f"⚙️  Config: temp={self.sampling_config.temperature}, top_p={self.sampling_config.top_p}")
        print(f"📝 Scenario: Pure immediate dream scenario (no system prompt, no markers)")
        
        for i, language in enumerate(LANGUAGE_CONFIG.keys(), 1):
            config = LANGUAGE_CONFIG[language]
            print(f"\n{'='*60}")
            print(f"🌍 Language {i}/{len(LANGUAGE_CONFIG)}: {language.upper()}")
            print(f"📝 Prompt: {config['prompt']}")
            print(f"{'='*60}")
            
            result = await self.generate_dreams_for_language(language, dreams_per_language)
            results[language] = result
            
            # Save global logs after each language
            await self._save_global_logs()
        
        # Final summary
        total_generated = sum(result['dreams_generated'] for result in results.values())
        success_rate = (total_generated / total_target) * 100 if total_target > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"🎉 GENERATION COMPLETE")
        print(f"📊 Total dreams: {total_generated}/{total_target} ({success_rate:.1f}%)")
        print(f"📁 Session: {self.session_id}")
        print(f"💾 Logs saved to: {self.base_logs_dir}/")
        print(f"{'='*60}")
        
        # Save final checkpoint
        self.save_checkpoint()
        
        return {
            'session_id': self.session_id,
            'total_dreams_generated': total_generated,
            'total_target_dreams': total_target,
            'success_rate': success_rate,
            'languages': results,
            'sampling_config': {
                'temperature': self.sampling_config.temperature,
                'top_p': self.sampling_config.top_p,
                'presence_penalty': self.sampling_config.presence_penalty,
                'frequency_penalty': self.sampling_config.frequency_penalty
            }
        }

async def main():
    """Main execution function"""
    
    # Check for API keys
    import os
    api_keys = {
        'openai': os.getenv('OPENAI_API_KEY')
    }
    
    if not api_keys['openai']:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return
    
    # Create generator
    generator = OptimizedBatchDreamGenerator(api_keys)
    
    # Generate dreams for all languages
    results = await generator.generate_all_languages(dreams_per_language=100)
    
    # Print final summary
    print(f"\n🎯 MISSION SUMMARY")
    print(f"Session ID: {results['session_id']}")
    print(f"Dreams Generated: {results['total_dreams_generated']}/{results['total_target_dreams']}")
    print(f"Success Rate: {results['success_rate']:.1f}%")
    print(f"\nLanguage Breakdown:")
    for lang, data in results['languages'].items():
        print(f"  {lang}: {data['dreams_generated']} dreams")
    
    print(f"\n📊 Ready for analysis:")
    print(f"  • Thematic Analysis: python dream_thematic_analysis.py")
    print(f"  • Statistical Analysis: python statistical_analysis.py --session-id {results['session_id']}")
    print(f"  • Cultural Analysis: python src/analysis/cultural_analysis.py")
    print(f"  • Typological Analysis: python src/analysis/typological_analyzer.py")

if __name__ == "__main__":
    asyncio.run(main()) 