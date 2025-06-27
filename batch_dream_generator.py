"""
Batch dream generator for cross-linguistic research with rigorous statistical sampling.
Implements multilevel modeling protocols with mixed-effects logistic regression preparation.
Enhanced with temporal clustering controls and prompt entropy measures.

NOTE: Temporal dispersion controls are SUSPENDED by default for faster generation.
Use --enable-temporal-dispersion flag to enable temporal entropy controls.
"""

import asyncio
import json
import csv
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import time
import uuid
import random
import schedule
from dataclasses import dataclass

from src.models.llm_interface import LLMInterface, GenerationConfig
from src.config.languages import LANGUAGE_CONFIG

@dataclass
class SamplingConfig:
    """Configuration for rigorous statistical sampling with temporal clustering controls."""
    batch_size: int = 50
    # Temporal entropy controls (SUSPENDED by default - available as option)
    use_temporal_dispersion: bool = False  # SUSPENDED: Set to False by default
    temporal_dispersion_hours: int = 2
    min_temporal_dispersion_minutes: int = 30  # Minimum time between individual calls
    max_temporal_dispersion_hours: int = 24   # Maximum spread for temporal diversity
    session_independence: bool = True
    min_samples_per_language: int = 500
    user_id_regeneration: bool = True
    presence_penalty: float = 0.1
    frequency_penalty: float = 0.0
    temperature: float = 1.0
    top_p: float = 1.0
    # Prompt entropy controls
    use_prompt_variants: bool = True
    invisible_marker_probability: float = 0.3  # 30% of prompts get markers
    prompt_variant_types: int = 5  # Number of different invisible marker types

class PromptEntropyGenerator:
    """Generates prompt variants with invisible markers for increased entropy."""
    
    def __init__(self, variant_types: int = 5):
        self.variant_types = variant_types
        self.invisible_markers = [
            "<!-- -->",  # HTML comment
            "\u200B",    # Zero-width space
            "\u2060",    # Word joiner (invisible)
            "\uFEFF",    # Zero-width no-break space
            "‌",         # Zero-width non-joiner
        ]
        
    def generate_prompt_variant(self, base_prompt: str, use_markers: bool = True) -> Tuple[str, str, str]:
        """
        Generate a prompt variant with invisible markers and return prompt_id.
        
        Returns:
            tuple: (modified_prompt, prompt_id, marker_type)
        """
        prompt_id = str(uuid.uuid4())[:8]  # Short prompt ID for logging
        
        if not use_markers:
            return base_prompt, prompt_id, "none"
        
        # Select random marker type
        marker_idx = random.randint(0, len(self.invisible_markers) - 1)
        marker = self.invisible_markers[marker_idx]
        marker_type = f"marker_{marker_idx}"
        
        # Apply marker in different positions
        position = random.choice(['prefix', 'suffix', 'middle'])
        
        if position == 'prefix':
            modified_prompt = f"{marker}{base_prompt}"
        elif position == 'suffix':
            modified_prompt = f"{base_prompt}{marker}"
        else:  # middle - insert after first word if possible
            words = base_prompt.split(' ', 1)
            if len(words) > 1:
                modified_prompt = f"{words[0]}{marker} {words[1]}"
            else:
                modified_prompt = f"{base_prompt}{marker}"
                
        return modified_prompt, prompt_id, f"{marker_type}_{position}"

class TemporalDispersionManager:
    """Manages temporal dispersion to avoid clustering and server-level biases."""
    
    def __init__(self, config: SamplingConfig):
        self.config = config
        self.call_times = []
        
    def get_next_call_delay(self) -> float:
        """Calculate optimal delay for next call to maximize temporal diversity."""
        if not self.call_times:
            return 0.0
            
        current_time = time.time()
        
        # Minimum delay between calls
        min_delay = self.config.min_temporal_dispersion_minutes * 60
        
        # Add randomization to avoid predictable patterns
        # Random delay between min_delay and min_delay * 3
        base_delay = random.uniform(min_delay, min_delay * 3)
        
        # Add occasional longer delays to span different time windows
        if random.random() < 0.1:  # 10% chance of much longer delay
            extended_delay = random.uniform(
                self.config.temporal_dispersion_hours * 3600,
                self.config.max_temporal_dispersion_hours * 3600
            )
            base_delay = max(base_delay, extended_delay)
            
        return base_delay
    
    def record_call_time(self):
        """Record the timestamp of an API call."""
        self.call_times.append(time.time())
        
    def get_temporal_statistics(self) -> Dict:
        """Get statistics about temporal distribution of calls."""
        if len(self.call_times) < 2:
            return {"intervals": [], "mean_interval": 0, "std_interval": 0}
            
        intervals = [self.call_times[i] - self.call_times[i-1] 
                    for i in range(1, len(self.call_times))]
        
        return {
            "intervals": intervals,
            "mean_interval": sum(intervals) / len(intervals),
            "std_interval": (sum((x - sum(intervals)/len(intervals))**2 for x in intervals) / len(intervals))**0.5,
            "min_interval": min(intervals),
            "max_interval": max(intervals),
            "total_span_hours": (max(self.call_times) - min(self.call_times)) / 3600
        }

class BatchDreamGenerator:
    def __init__(self, api_keys: Dict[str, str], model: str = 'gpt-4o', 
                 sampling_config: SamplingConfig = None):
        self.llm_interface = LLMInterface(api_keys)
        self.model = model
        self.sampling_config = sampling_config or SamplingConfig()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize entropy and temporal managers
        self.prompt_entropy = PromptEntropyGenerator(self.sampling_config.prompt_variant_types)
        self.temporal_manager = TemporalDispersionManager(self.sampling_config)
        
        # Create structured logs directory
        self.base_logs_dir = 'logs'
        self.current_language = None  # Will be set when processing a language
        os.makedirs(self.base_logs_dir, exist_ok=True)
        
        # Setup basic logging (will be enhanced per language)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{self.base_logs_dir}/batch_generation_{self.session_id}.log'),
                logging.StreamHandler()
            ]
        )
        
        # Initialize data structures for logging
        self.api_calls_data = {}  # Will be organized by language
        self.dreams_data = {}     # Will be organized by language
        self.batch_tracker = {}
        
        # Global checkpoint system
        self.checkpoint_file = f"{self.base_logs_dir}/checkpoint_{self.session_id}.json"
        self.load_checkpoint()
        
        # Data storage for CSV/JSON export (organized by language)
        self.error_data = {}
        self.rejected_data = {}
        
    def setup_language_logging(self, language: str):
        """Setup structured logging directories for a specific language and model."""
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
        if language not in self.error_data:
            self.error_data[language] = []
        if language not in self.rejected_data:
            self.rejected_data[language] = []
        
        # Setup language-specific log files
        self.error_log_file = os.path.join(self.language_logs_dir, "errors.jsonl")
        self.rejected_dreams_file = os.path.join(self.language_logs_dir, "rejected_dreams.jsonl")
        
        # CSV and JSON export files
        self.error_csv_file = os.path.join(self.language_logs_dir, "errors.csv")
        self.error_json_file = os.path.join(self.language_logs_dir, "errors.json")
        self.rejected_csv_file = os.path.join(self.language_logs_dir, "rejected_dreams.csv")
        self.rejected_json_file = os.path.join(self.language_logs_dir, "rejected_dreams.json")
        
        logging.info(f"Setup structured logging for {language} with {self.model} in: {self.language_logs_dir}")
        
    async def generate_dream_with_protocol(self, language: str, batch_id: str, 
                                         dream_number: int) -> Dict:
        """Generate a single dream using rigorous sampling protocol with enhanced entropy."""
        if language not in LANGUAGE_CONFIG:
            raise ValueError(f"Language '{language}' not found in configuration")
        
        config = LANGUAGE_CONFIG[language]
        base_prompt = config['prompt']
        
        # Generate prompt variant with entropy measures
        use_markers = (self.sampling_config.use_prompt_variants and 
                      random.random() < self.sampling_config.invisible_marker_probability)
        
        modified_prompt, prompt_id, marker_info = self.prompt_entropy.generate_prompt_variant(
            base_prompt, use_markers
        )
        
        # Apply temporal dispersion
        delay = self.temporal_manager.get_next_call_delay()
        if delay > 0:
            logging.info(f"Temporal dispersion delay: {delay:.1f} seconds ({delay/60:.1f} minutes)")
            await asyncio.sleep(delay)
        
        # Generate unique user ID for session independence with additional entropy
        entropy_suffix = f"_{random.randint(10000, 99999)}"
        user_id = (str(uuid.uuid4()) + entropy_suffix 
                  if self.sampling_config.user_id_regeneration 
                  else self.session_id + entropy_suffix)
        
        call_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Record call time for temporal analysis
        self.temporal_manager.record_call_time()
        
        # Prepare generation config with statistical controls
        generation_config = GenerationConfig(
            model=self.model,
            temperature=self.sampling_config.temperature,
            top_p=self.sampling_config.top_p,
            frequency_penalty=self.sampling_config.frequency_penalty,
            presence_penalty=self.sampling_config.presence_penalty
        )
        
        try:
            # Make API call with enhanced protocol
            dream = await self.llm_interface.generate_dream(modified_prompt, generation_config)
            end_time = time.time()
            duration = end_time - start_time
            
            # Check if dream is valid
            if dream and not self._is_error_response(dream):
                status = "success"
                dream_content = dream
            else:
                status = "filtered"
                dream_content = f"[FILTERED] {dream}"
                
                # Log rejected dream with enhanced metadata
                rejected_data = {
                    'call_id': call_id,
                    'batch_id': batch_id,
                    'user_id': user_id,
                    'timestamp': datetime.now().isoformat(),
                    'language': language,
                    'language_code': config['code'],
                    'script': config['script'],
                    'model': self.model,
                    'temperature': self.sampling_config.temperature,
                    'base_prompt': base_prompt,
                    'modified_prompt': modified_prompt,
                    'prompt_id': prompt_id,
                    'marker_info': marker_info,
                    'dream_number': dream_number,
                    'original_dream': dream,
                    'rejection_reason': 'error_response_detected',
                    'session_id': self.session_id,
                    'temporal_delay': delay
                }
                self.log_rejected_dream(rejected_data)
            
            # Log API call with enhanced statistical metadata
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
                'base_prompt': base_prompt,
                'modified_prompt': modified_prompt,
                'prompt_id': prompt_id,
                'marker_info': marker_info,
                'used_invisible_markers': use_markers,
                'dream_number': dream_number,
                'batch_size': self.sampling_config.batch_size,
                'dream': dream_content,
                'status': status,
                'duration_seconds': round(duration, 3),
                'temporal_delay_seconds': round(delay, 3),
                'start_time': datetime.fromtimestamp(start_time).isoformat(),
                'end_time': datetime.fromtimestamp(end_time).isoformat(),
                'session_id': self.session_id,
                'temporal_dispersion': self.sampling_config.temporal_dispersion_hours,
                'session_independence': self.sampling_config.session_independence
            }
            
            self.api_calls_data[language].append(api_call_data)
            
            # Log dream data for statistical analysis with entropy metadata
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
                'marker_info': marker_info,
                'temporal_delay': delay
            }
            self.dreams_data[language].append(dream_data)
            
            temporal_stats = self.temporal_manager.get_temporal_statistics()
            logging.info(f"Dream {dream_number} for {language} (Batch {batch_id}, Prompt ID: {prompt_id}): "
                        f"{status} ({duration:.3f}s, delay: {delay:.1f}s)")
            
            return {
                'call_id': call_id,
                'batch_id': batch_id,
                'user_id': user_id,
                'language': language,
                'dream': dream_content,
                'status': status,
                'duration': duration,
                'prompt_id': prompt_id,
                'marker_info': marker_info,
                'temporal_delay': delay
            }
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            # Log API error with entropy metadata
            error_data = {
                'call_id': call_id,
                'batch_id': batch_id,
                'user_id': user_id,
                'timestamp': datetime.now().isoformat(),
                'language': language,
                'language_code': config['code'],
                'script': config['script'],
                'model': self.model,
                'temperature': self.sampling_config.temperature,
                'base_prompt': base_prompt,
                'modified_prompt': modified_prompt,
                'prompt_id': prompt_id,
                'marker_info': marker_info,
                'dream_number': dream_number,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'duration_seconds': round(duration, 3),
                'temporal_delay_seconds': round(delay, 3),
                'start_time': datetime.fromtimestamp(start_time).isoformat(),
                'end_time': datetime.fromtimestamp(end_time).isoformat(),
                'session_id': self.session_id
            }
            self.log_error(error_data)
            
            # Log failed call with entropy metadata
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
                'base_prompt': base_prompt,
                'modified_prompt': modified_prompt,
                'prompt_id': prompt_id,
                'marker_info': marker_info,
                'used_invisible_markers': use_markers,
                'dream_number': dream_number,
                'batch_size': self.sampling_config.batch_size,
                'dream': f"[ERROR] {str(e)}",
                'status': 'error',
                'error_message': str(e),
                'duration_seconds': round(duration, 3),
                'temporal_delay_seconds': round(delay, 3),
                'start_time': datetime.fromtimestamp(start_time).isoformat(),
                'end_time': datetime.fromtimestamp(end_time).isoformat(),
                'session_id': self.session_id,
                'temporal_dispersion': self.sampling_config.temporal_dispersion_hours,
                'session_independence': self.sampling_config.session_independence
            }
            
            self.api_calls_data[language].append(api_call_data)
            logging.error(f"Error generating dream {dream_number} for {language}: {e}")
            
            return {
                'call_id': call_id,
                'batch_id': batch_id,
                'user_id': user_id,
                'language': language,
                'dream': f"[ERROR] {str(e)}",
                'status': 'error',
                'duration': duration
            }
    
    async def run_batch_with_temporal_dispersion(self, language: str, batch_size: int = None) -> List[Dict]:
        """Run a batch with temporal dispersion for statistical robustness."""
        batch_size = batch_size or self.sampling_config.batch_size
        batch_id = str(uuid.uuid4())
        
        temporal_status = "ENABLED" if self.sampling_config.use_temporal_dispersion else "SUSPENDED"
        logging.info(f"Starting batch {batch_id} for {language} with {batch_size} dreams [Temporal Dispersion: {temporal_status}]")
        
        results = []
        successful_calls = 0
        failed_calls = 0
        
        for i in range(batch_size):
            # Generate dream with protocol
            result = await self.generate_dream_with_protocol(language, batch_id, i + 1)
            results.append(result)
            
            if result['status'] == 'success':
                successful_calls += 1
            else:
                failed_calls += 1
            
            # Temporal dispersion: random delay between calls (SUSPENDED by default)
            if i < batch_size - 1:  # Don't delay after the last call
                if self.sampling_config.use_temporal_dispersion:
                    delay = random.uniform(1.5, 3.0)  # 1.5-3 seconds for hygiene
                    await asyncio.sleep(delay)
                else:
                    # Minimal delay for API rate limiting only
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
        
        logging.info(f"Completed batch {batch_id}: {successful_calls} successful, {failed_calls} failed")
        
        return results
    
    async def generate_dreams_for_language(self, language: str, num_dreams: int = None) -> Dict:
        """Generate dreams for a language using batched approach with temporal dispersion."""
        num_dreams = num_dreams or self.sampling_config.min_samples_per_language
        
        if language not in LANGUAGE_CONFIG:
            raise ValueError(f"Language '{language}' not found in configuration")
        
        # Setup structured logging for this language
        self.setup_language_logging(language)
        
        # Check current progress
        progress = self.get_progress(language, num_dreams)
        logging.info(f"Progress for {language}: {progress['successful_dreams']}/{num_dreams} dreams completed ({progress['progress_percent']:.1f}%)")
        
        # If already completed, return existing results
        if progress['successful_dreams'] >= num_dreams:
            logging.info(f"Already completed {num_dreams} dreams for {language}. Skipping generation.")
            return {
                'language': language,
                'successful_dreams': progress['successful_dreams'],
                'failed_dreams': progress['failed_calls'],
                'total_calls': progress['completed_calls'],
                'dreams': [call['dream'] for call in self.api_calls_data if call['language'] == language and call['status'] == 'success'],
                'batches_processed': len([batch for batch in self.batch_tracker.values() if batch['language'] == language]),
                'resumed': True
            }
        
        # Calculate remaining dreams needed
        remaining_dreams = progress['remaining']
        logging.info(f"Starting generation of {remaining_dreams} remaining dreams for {language}")
        
        all_dreams = [call['dream'] for call in self.api_calls_data.get(language, []) if call['status'] == 'success']
        total_successful = progress['successful_dreams']
        total_failed = progress['failed_calls']
        
        # Calculate number of batches needed for remaining dreams
        num_batches = (remaining_dreams + self.sampling_config.batch_size - 1) // self.sampling_config.batch_size
        
        for batch_num in range(num_batches):
            batch_start = batch_num * self.sampling_config.batch_size
            batch_end = min(batch_start + self.sampling_config.batch_size, remaining_dreams)
            current_batch_size = batch_end - batch_start
            
            logging.info(f"Processing batch {batch_num + 1}/{num_batches} for {language} ({current_batch_size} dreams)")
            
            # Run batch with temporal dispersion
            batch_results = await self.run_batch_with_temporal_dispersion(language, current_batch_size)
            
            # Collect successful dreams
            for result in batch_results:
                if result['status'] == 'success':
                    all_dreams.append(result['dream'])
                    total_successful += 1
                else:
                    total_failed += 1
            
            # Save checkpoint after each batch
            self.save_checkpoint()
            
            # Temporal dispersion between batches (SUSPENDED by default)
            if batch_num < num_batches - 1:
                if self.sampling_config.use_temporal_dispersion:
                    batch_delay = random.uniform(
                        self.sampling_config.temporal_dispersion_hours * 3600 * 0.5,  # 50% of specified hours
                        self.sampling_config.temporal_dispersion_hours * 3600
                    )
                    logging.info(f"Temporal dispersion: waiting {batch_delay/3600:.2f} hours before next batch")
                    await asyncio.sleep(batch_delay)
                else:
                    # Minimal delay between batches for API rate limiting only
                    logging.info("Temporal dispersion SUSPENDED - proceeding to next batch with minimal delay")
                    await asyncio.sleep(1.0)  # 1 second delay only
        
        # Save logs for this language
        await self._save_language_logs(language, total_successful, total_failed)
        
        # Export error logs for this language
        self.export_error_logs()
        
        logging.info(f"Completed {language}: {total_successful} successful, {total_failed} failed")
        
        return {
            'language': language,
            'successful_dreams': total_successful,
            'failed_dreams': total_failed,
            'total_calls': len(self.api_calls_data.get(language, [])),
            'dreams': all_dreams,
            'batches_processed': len([batch for batch in self.batch_tracker.values() if batch['language'] == language]),
            'resumed': progress['completed_calls'] > 0
        }
    
    def _is_error_response(self, dream: str) -> bool:
        """Check if the response is an error or invalid."""
        error_indicators = [
            'barkatu', 'sorry', 'error', 'cannot', 'unable', 'i apologize',
            'i\'m sorry', 'i am sorry', 'i cannot', 'i am unable',
            'lo siento', 'disculpe', 'no puedo', 'no soy capaz',
            'izvinjavam se', 'žao mi je', 'ne mogu', 'nisam u stanju',
            'סליחה', 'אני מצטער', 'אני לא יכול', 'אני לא מסוגל',
            'opravičujem se', 'žal mi je', 'ne morem', 'nisem sposoben'
        ]
        
        dream_lower = dream.lower()
        return any(indicator in dream_lower for indicator in error_indicators)
    
    async def _save_language_logs(self, language: str, successful_calls: int, failed_calls: int):
        """Save detailed logs for the language in structured directories with JSON and CSV formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get data for this language
        language_api_calls = self.api_calls_data.get(language, [])
        language_dreams = self.dreams_data.get(language, [])
        
        # Get temporal statistics for this language session
        temporal_stats = self.temporal_manager.get_temporal_statistics()
        
        # Get entropy statistics for this language
        language_entropy_stats = self._analyze_language_entropy(language)
        
        # Save comprehensive JSON logs
        json_filename = os.path.join(self.language_logs_dir, "session_data.json")
        json_data = {
            'metadata': {
                'language': language,
                'language_code': LANGUAGE_CONFIG[language]['code'],
                'script': LANGUAGE_CONFIG[language]['script'],
                'model': self.model,
                'session_id': self.session_id,
                'timestamp': timestamp,
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
                    # Temporal dispersion status
                    'use_temporal_dispersion': self.sampling_config.use_temporal_dispersion,
                    'temporal_dispersion_hours': self.sampling_config.temporal_dispersion_hours,
                    'min_temporal_dispersion_minutes': self.sampling_config.min_temporal_dispersion_minutes
                }
            },
            'temporal_statistics': temporal_stats,
            'entropy_statistics': language_entropy_stats,
            'api_calls': language_api_calls,
            'dreams': language_dreams
        }
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        # Save structured CSV logs
        csv_api_filename = os.path.join(self.language_logs_dir, "api_calls.csv")
        if language_api_calls:
            df_api = pd.DataFrame(language_api_calls)
            df_api.to_csv(csv_api_filename, index=False, encoding='utf-8')
        
        csv_dreams_filename = os.path.join(self.language_logs_dir, "dreams.csv")
        if language_dreams:
            df_dreams = pd.DataFrame(language_dreams)
            df_dreams.to_csv(csv_dreams_filename, index=False, encoding='utf-8')
        
        # Save temporal statistics
        temporal_filename = os.path.join(self.language_logs_dir, "temporal_statistics.json")
        with open(temporal_filename, 'w', encoding='utf-8') as f:
            json.dump({
                'session_id': self.session_id,
                'language': language,
                'timestamp': timestamp,
                'statistics': temporal_stats,
                'call_times': self.temporal_manager.call_times
            }, f, ensure_ascii=False, indent=2)
        
        logging.info(f"Saved structured logs for {language} in: {self.language_logs_dir}")
        logging.info(f"  Files: session_data.json, api_calls.csv, dreams.csv, temporal_statistics.json")
    
    def _analyze_language_entropy(self, language: str) -> Dict:
        """Analyze prompt entropy usage patterns for a specific language."""
        language_calls = self.api_calls_data.get(language, [])
        if not language_calls:
            return {}
        
        total_calls = len(language_calls)
        calls_with_markers = len([call for call in language_calls 
                                if call.get('used_invisible_markers', False)])
        
        # Analyze marker types for this language
        marker_types = {}
        prompt_ids_used = set()
        
        for call in language_calls:
            marker_info = call.get('marker_info', 'none')
            if marker_info not in marker_types:
                marker_types[marker_info] = 0
            marker_types[marker_info] += 1
            
            prompt_id = call.get('prompt_id')
            if prompt_id:
                prompt_ids_used.add(prompt_id)
        
        return {
            'language': language,
            'total_calls': total_calls,
            'calls_with_markers': calls_with_markers,
            'marker_usage_rate': calls_with_markers / total_calls if total_calls > 0 else 0,
            'marker_type_distribution': marker_types,
            'unique_prompt_ids': len(prompt_ids_used),
            'configured_marker_probability': self.sampling_config.invisible_marker_probability
        }
    
    async def generate_all_languages(self, dreams_per_language: int = 100) -> Dict:
        """Generate dreams for all configured languages."""
        results = {}
        
        for language in LANGUAGE_CONFIG.keys():
            logging.info(f"Starting batch generation for {language}")
            result = await self.generate_dreams_for_language(language, dreams_per_language)
            results[language] = result
            
            # Save comprehensive session logs
            await self._save_session_logs()
        
        return results
    
    async def _save_session_logs(self):
        """Save comprehensive session logs with temporal and entropy analysis."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get temporal statistics
        temporal_stats = self.temporal_manager.get_temporal_statistics()
        
        # Analyze prompt entropy usage
        prompt_entropy_stats = self._analyze_prompt_entropy()
        
        # Session summary with enhanced metadata
        session_summary = {
            'session_id': self.session_id,
            'timestamp': timestamp,
            'model': self.model,
            'sampling_config': {
                'temperature': self.sampling_config.temperature,
                'top_p': self.sampling_config.top_p,
                'presence_penalty': self.sampling_config.presence_penalty,
                'frequency_penalty': self.sampling_config.frequency_penalty,
                'batch_size': self.sampling_config.batch_size,
                'temporal_dispersion_hours': self.sampling_config.temporal_dispersion_hours,
                'min_temporal_dispersion_minutes': self.sampling_config.min_temporal_dispersion_minutes,
                'max_temporal_dispersion_hours': self.sampling_config.max_temporal_dispersion_hours,
                'use_prompt_variants': self.sampling_config.use_prompt_variants,
                'invisible_marker_probability': self.sampling_config.invisible_marker_probability
            },
            'total_api_calls': sum(len(calls) for calls in self.api_calls_data.values()),
            'total_dreams': sum(len(dreams) for dreams in self.dreams_data.values()),
            'languages_processed': list(self.api_calls_data.keys()),
            'success_rate': (sum(len([call for call in calls if call['status'] == 'success']) 
                               for calls in self.api_calls_data.values()) / 
                           sum(len(calls) for calls in self.api_calls_data.values()) 
                           if any(self.api_calls_data.values()) else 0),
            'temporal_statistics': temporal_stats,
            'prompt_entropy_statistics': prompt_entropy_stats
        }
        
        # Save session summary
        summary_filename = f"{self.base_logs_dir}/session_summary_{self.session_id}.json"
        with open(summary_filename, 'w', encoding='utf-8') as f:
            json.dump(session_summary, f, ensure_ascii=False, indent=2)
        
        # Save all API calls (flatten all languages)
        all_calls_filename = f"{self.base_logs_dir}/all_api_calls_{self.session_id}.csv"
        if self.api_calls_data:
            all_calls = []
            for language_calls in self.api_calls_data.values():
                all_calls.extend(language_calls)
            if all_calls:
                df_all_calls = pd.DataFrame(all_calls)
                df_all_calls.to_csv(all_calls_filename, index=False, encoding='utf-8')
        
        # Save all dreams (flatten all languages)
        all_dreams_filename = f"{self.base_logs_dir}/all_dreams_{self.session_id}.csv"
        if self.dreams_data:
            all_dreams = []
            for language_dreams in self.dreams_data.values():
                all_dreams.extend(language_dreams)
            if all_dreams:
                df_all_dreams = pd.DataFrame(all_dreams)
                df_all_dreams.to_csv(all_dreams_filename, index=False, encoding='utf-8')
        
        # Save batch tracker
        batch_filename = f"{self.logs_dir}/batch_tracker_{self.session_id}.json"
        with open(batch_filename, 'w', encoding='utf-8') as f:
            json.dump(self.batch_tracker, f, ensure_ascii=False, indent=2)
        
        # Export final error logs
        self.export_error_logs()
        
        # Save temporal statistics separately for detailed analysis
        temporal_filename = f"{self.logs_dir}/temporal_statistics_{self.session_id}.json"
        with open(temporal_filename, 'w', encoding='utf-8') as f:
            json.dump({
                'session_id': self.session_id,
                'timestamp': timestamp,
                'temporal_statistics': temporal_stats,
                'call_times': self.temporal_manager.call_times,
                'prompt_entropy_statistics': prompt_entropy_stats
            }, f, ensure_ascii=False, indent=2)
        
        logging.info(f"Session logs saved: {summary_filename}, {all_calls_filename}, {all_dreams_filename}, {batch_filename}, {temporal_filename}")

    def _analyze_prompt_entropy(self) -> Dict:
        """Analyze prompt entropy usage patterns across all languages."""
        if not self.api_calls_data:
            return {}
        
        # Flatten all calls from all languages
        all_calls = []
        for language_calls in self.api_calls_data.values():
            all_calls.extend(language_calls)
        
        if not all_calls:
            return {}
        
        total_calls = len(all_calls)
        calls_with_markers = len([call for call in all_calls 
                                if call.get('used_invisible_markers', False)])
        
        # Analyze marker types
        marker_types = {}
        prompt_ids_used = set()
        
        for call in all_calls:
            marker_info = call.get('marker_info', 'none')
            if marker_info not in marker_types:
                marker_types[marker_info] = 0
            marker_types[marker_info] += 1
            
            prompt_id = call.get('prompt_id')
            if prompt_id:
                prompt_ids_used.add(prompt_id)
        
        # Language-specific entropy analysis
        language_entropy = {}
        for language, language_calls in self.api_calls_data.items():
            language_markers = [call for call in language_calls 
                              if call.get('used_invisible_markers', False)]
            
            language_entropy[language] = {
                'total_calls': len(language_calls),
                'calls_with_markers': len(language_markers),
                'marker_usage_rate': len(language_markers) / len(language_calls) if language_calls else 0,
                'unique_prompt_ids': len(set(call.get('prompt_id') for call in language_calls if call.get('prompt_id')))
            }
        
        return {
            'total_calls': total_calls,
            'calls_with_markers': calls_with_markers,
            'marker_usage_rate': calls_with_markers / total_calls if total_calls > 0 else 0,
            'marker_type_distribution': marker_types,
            'unique_prompt_ids': len(prompt_ids_used),
            'language_specific_entropy': language_entropy,
            'configured_marker_probability': self.sampling_config.invisible_marker_probability
        }

    def load_checkpoint(self):
        """Load checkpoint if it exists."""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                    self.batch_tracker = checkpoint_data.get('batch_tracker', {})
                    self.api_calls_data = checkpoint_data.get('api_calls_data', {})
                    self.dreams_data = checkpoint_data.get('dreams_data', {})
                    self.error_data = checkpoint_data.get('error_data', {})
                    self.rejected_data = checkpoint_data.get('rejected_data', {})
                    
                    # Restore temporal manager state
                    temporal_data = checkpoint_data.get('temporal_data', {})
                    if 'call_times' in temporal_data:
                        self.temporal_manager.call_times = temporal_data['call_times']
                    
                    total_calls = sum(len(calls) for calls in self.api_calls_data.values())
                    total_dreams = sum(len(dreams) for dreams in self.dreams_data.values())
                    logging.info(f"Loaded checkpoint: {total_calls} API calls, {total_dreams} dreams, {len(self.temporal_manager.call_times)} temporal records")
                    return True
            except Exception as e:
                logging.warning(f"Failed to load checkpoint: {e}")
                self.batch_tracker = {}
                self.api_calls_data = {}
                self.dreams_data = {}
                self.error_data = {}
                self.rejected_data = {}
        return False
    
    def save_checkpoint(self):
        """Save current progress to checkpoint file."""
        try:
            checkpoint_data = {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'batch_tracker': self.batch_tracker,
                'api_calls_data': self.api_calls_data,
                'dreams_data': self.dreams_data,
                'error_data': self.error_data,
                'rejected_data': self.rejected_data,
                'temporal_data': {
                    'call_times': self.temporal_manager.call_times,
                    'temporal_statistics': self.temporal_manager.get_temporal_statistics()
                }
            }
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            total_calls = sum(len(calls) for calls in self.api_calls_data.values())
            total_dreams = sum(len(dreams) for dreams in self.dreams_data.values())
            logging.info(f"Checkpoint saved: {total_calls} API calls, {total_dreams} dreams")
            
            # Export error logs to CSV and JSON
            self.export_error_logs()
            
        except Exception as e:
            logging.error(f"Failed to save checkpoint: {e}")
    
    def get_progress(self, language: str, total_dreams: int) -> Dict:
        """Get current progress for a language."""
        language_calls = self.api_calls_data.get(language, [])
        successful_calls = [call for call in language_calls if call['status'] == 'success']
        
        return {
            'language': language,
            'total_requested': total_dreams,
            'completed_calls': len(language_calls),
            'successful_dreams': len(successful_calls),
            'failed_calls': len(language_calls) - len(successful_calls),
            'remaining': max(0, total_dreams - len(successful_calls)),
            'progress_percent': (len(successful_calls) / total_dreams * 100) if total_dreams > 0 else 0
        }
    
    def log_error(self, error_data: Dict):
        """Log error details to language-specific error log file."""
        try:
            language = error_data.get('language', 'unknown')
            
            # Save to JSONL for streaming
            if hasattr(self, 'error_log_file'):
                with open(self.error_log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(error_data, ensure_ascii=False) + '\n')
            
            # Store for CSV/JSON export
            if language not in self.error_data:
                self.error_data[language] = []
            self.error_data[language].append(error_data)
            
        except Exception as e:
            logging.error(f"Failed to log error: {e}")
    
    def log_rejected_dream(self, rejected_data: Dict):
        """Log rejected dream details to language-specific rejected dreams file."""
        try:
            language = rejected_data.get('language', 'unknown')
            
            # Save to JSONL for streaming
            if hasattr(self, 'rejected_dreams_file'):
                with open(self.rejected_dreams_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(rejected_data, ensure_ascii=False) + '\n')
            
            # Store for CSV/JSON export
            if language not in self.rejected_data:
                self.rejected_data[language] = []
            self.rejected_data[language].append(rejected_data)
            
        except Exception as e:
            logging.error(f"Failed to log rejected dream: {e}")
    
    def export_error_logs(self):
        """Export error and rejection logs to CSV and JSON formats in structured directories."""
        try:
            if not hasattr(self, 'language_logs_dir'):
                return  # No structured logging setup yet
                
            current_language = self.current_language
            if not current_language:
                return
                
            # Export error data for current language
            language_errors = self.error_data.get(current_language, [])
            if language_errors:
                # CSV export
                df_errors = pd.DataFrame(language_errors)
                df_errors.to_csv(self.error_csv_file, index=False, encoding='utf-8')
                
                # JSON export
                with open(self.error_json_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'metadata': {
                            'session_id': self.session_id,
                            'language': current_language,
                            'timestamp': datetime.now().isoformat(),
                            'total_errors': len(language_errors)
                        },
                        'errors': language_errors
                    }, f, ensure_ascii=False, indent=2)
                
                logging.info(f"Exported {len(language_errors)} errors for {current_language}")
            
            # Export rejected dreams data for current language
            language_rejected = self.rejected_data.get(current_language, [])
            if language_rejected:
                # CSV export
                df_rejected = pd.DataFrame(language_rejected)
                df_rejected.to_csv(self.rejected_csv_file, index=False, encoding='utf-8')
                
                # JSON export
                with open(self.rejected_json_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'metadata': {
                            'session_id': self.session_id,
                            'language': current_language,
                            'timestamp': datetime.now().isoformat(),
                            'total_rejections': len(language_rejected)
                        },
                        'rejected_dreams': language_rejected
                    }, f, ensure_ascii=False, indent=2)
                
                logging.info(f"Exported {len(language_rejected)} rejected dreams for {current_language}")
                
        except Exception as e:
            logging.error(f"Failed to export error logs: {e}")

async def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Batch Dream Generator with Enhanced Statistical Protocols and Entropy Controls',
        epilog='NOTE: Temporal dispersion controls are SUSPENDED by default for faster generation. Use --enable-temporal-dispersion to enable them.'
    )
    parser.add_argument('--language', type=str, help='Specific language to generate dreams for')
    parser.add_argument('--model', type=str, default='gpt-4o', help='LLM model to use')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for generation')
    # Temporal dispersion controls (SUSPENDED by default)
    parser.add_argument('--enable-temporal-dispersion', action='store_true', help='Enable temporal dispersion (SUSPENDED by default)')
    parser.add_argument('--temporal-dispersion', type=int, default=2, help='Hours between batches (only if temporal dispersion enabled)')
    parser.add_argument('--min-temporal-dispersion', type=int, default=30, help='Minimum minutes between individual calls (only if temporal dispersion enabled)')
    parser.add_argument('--max-temporal-dispersion', type=int, default=24, help='Maximum hours for temporal diversity (only if temporal dispersion enabled)')
    parser.add_argument('--dreams-per-language', type=int, default=500, help='Minimum dreams per language')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for generation')
    parser.add_argument('--use-prompt-variants', action='store_true', default=True, help='Enable prompt variants with invisible markers')
    parser.add_argument('--no-prompt-variants', action='store_true', help='Disable prompt variants (overrides --use-prompt-variants)')
    parser.add_argument('--marker-probability', type=float, default=0.3, help='Probability of adding invisible markers (0.0-1.0)')
    parser.add_argument('--prompt-variant-types', type=int, default=5, help='Number of different invisible marker types')
    parser.add_argument('--api-keys', type=str, help='Path to API keys file')
    parser.add_argument('--force-restart', action='store_true', help='Force restart and ignore existing checkpoints')
    
    args = parser.parse_args()
    
    # Load API keys
    api_keys = {}
    if args.api_keys and os.path.exists(args.api_keys):
        with open(args.api_keys, 'r') as f:
            api_keys = json.load(f)
    else:
        # Try environment variables
        api_keys = {
            'openai': os.getenv('OPENAI_API_KEY'),
            'anthropic': os.getenv('ANTHROPIC_API_KEY'),
            'openrouter': os.getenv('OPENROUTER_API_KEY')
        }
    
    # Defensive: ensure SamplingConfig is always constructed correctly with enhanced entropy controls
    use_prompt_variants = args.use_prompt_variants and not args.no_prompt_variants
    
    sampling_config = SamplingConfig(
        batch_size=int(args.batch_size),
        # Temporal dispersion controls (SUSPENDED by default)
        use_temporal_dispersion=args.enable_temporal_dispersion,  # SUSPENDED: False by default
        temporal_dispersion_hours=int(args.temporal_dispersion),
        min_temporal_dispersion_minutes=int(args.min_temporal_dispersion),
        max_temporal_dispersion_hours=int(args.max_temporal_dispersion),
        min_samples_per_language=int(args.dreams_per_language),
        temperature=float(args.temperature),
        use_prompt_variants=use_prompt_variants,
        invisible_marker_probability=float(args.marker_probability),
        prompt_variant_types=int(args.prompt_variant_types)
    )
    
    # Type check
    if not isinstance(sampling_config, SamplingConfig):
        print("[ERROR] Internal error: sampling_config is not a SamplingConfig instance.")
        print(f"Type: {type(sampling_config)} Value: {sampling_config}")
        return
    
    # Initialize generator
    generator = BatchDreamGenerator(api_keys, args.model, sampling_config)
    
    # Display temporal dispersion status clearly
    if sampling_config.use_temporal_dispersion:
        print("🕐 TEMPORAL DISPERSION: ENABLED")
        print(f"   - Between calls: {sampling_config.min_temporal_dispersion_minutes} min minimum")
        print(f"   - Between batches: {sampling_config.temporal_dispersion_hours} hours")
        print("   - This will significantly slow down generation for statistical robustness")
    else:
        print("⚡ TEMPORAL DISPERSION: SUSPENDED (Default)")
        print("   - Fast generation with minimal delays for API rate limiting only")
        print("   - Use --enable-temporal-dispersion to enable temporal controls")
    print()
    
    # If force restart, clear checkpoint
    if args.force_restart:
        if os.path.exists(generator.checkpoint_file):
            os.remove(generator.checkpoint_file)
            print("Checkpoint cleared. Starting fresh.")
        generator.api_calls_data = []
        generator.dreams_data = []
        generator.batch_tracker = {}
        generator.error_data = []
        generator.rejected_data = []
    
    if args.language:
        # Generate for specific language
        if args.language not in LANGUAGE_CONFIG:
            print(f"Error: Language '{args.language}' not found. Available languages: {list(LANGUAGE_CONFIG.keys())}")
            return
        
        # Show progress before starting
        progress = generator.get_progress(args.language, args.dreams_per_language)
        if progress['completed_calls'] > 0:
            print(f"Found existing progress: {progress['successful_dreams']}/{args.dreams_per_language} dreams completed ({progress['progress_percent']:.1f}%)")
            if not args.force_restart:
                print("Will resume from checkpoint. Use --force-restart to start over.")
        
        result = await generator.generate_dreams_for_language(args.language, args.dreams_per_language)
        print(f"Generated {result['successful_dreams']} dreams for {args.language}")
        print(f"Processed {result['batches_processed']} batches")
        if result.get('resumed', False):
            print("✓ Resumed from checkpoint successfully")
    else:
        # Generate for all languages
        results = await generator.generate_all_languages(args.dreams_per_language)
        total_successful = sum(r['successful_dreams'] for r in results.values())
        total_batches = sum(r['batches_processed'] for r in results.values())
        print(f"Generated {total_successful} total dreams across all languages")
        print(f"Processed {total_batches} total batches")

if __name__ == "__main__":
    asyncio.run(main()) 