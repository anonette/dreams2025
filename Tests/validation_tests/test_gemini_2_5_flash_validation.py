#!/usr/bin/env python3
"""
Comprehensive Gemini 2.5 Flash Validation and Optimization Test Script

This script validates the new Gemini 2.5 Flash configuration and helps optimize
settings for the dream generation batch process. It tests:

1. Model availability and compatibility
2. Performance benchmarking vs current setup
3. Rate limiting optimization
4. Dream quality assessment
5. Error handling validation
6. Configuration parameter tuning

Run this before updating your main generation script.
"""

import asyncio
import json
import os
import time
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from dataclasses import dataclass
from tqdm.asyncio import tqdm

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import your existing modules
from optimized_dream_languages import LANGUAGE_CONFIG
from src.models.llm_interface import LLMInterface, GenerationConfig

@dataclass
class TestResult:
    """Container for test results"""
    model: str
    success: bool
    response_time: float
    dream_content: str
    error_message: Optional[str] = None
    char_count: int = 0
    word_count: int = 0
    timestamp: str = ""

@dataclass
class ModelTestConfig:
    """Configuration for model testing"""
    model_name: str
    display_name: str
    temperature: float = 1.1
    top_p: float = 0.98
    max_tokens: int = 1000

class GeminiFlashValidator:
    """Comprehensive validator for Gemini 2.5 Flash configuration"""
    
    def __init__(self):
        self.api_keys = {'gemini': os.getenv('GEMINI_API_KEY')}
        if not self.api_keys['gemini']:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        # Setup LLM interface
        self.llm_interface = LLMInterface(self.api_keys)
        modified_api_keys = self.api_keys.copy()
        modified_api_keys['openai'] = self.api_keys['gemini']
        self.llm_interface = LLMInterface(modified_api_keys)
        self.llm_interface.setup_google_gemini_client()
        
        # Test configurations
        self.models_to_test = [
            ModelTestConfig("gemini-2.5-flash", "Gemini 2.5 Flash"),
            ModelTestConfig("gemini-2.5-flash-latest", "Gemini 2.5 Flash Latest"),
            ModelTestConfig("models/gemini-2.5-flash", "Gemini 2.5 Flash (models/)"),
            ModelTestConfig("gemini-1.5-pro", "Gemini 1.5 Pro (Current)"),
            ModelTestConfig("gemini-1.5-pro-latest", "Gemini 1.5 Pro Latest"),
        ]
        
        # Languages to test (subset for validation)
        self.test_languages = ['english', 'basque']  # Start with 2 languages
        
        # Results storage
        self.results = []
        self.performance_data = {}
        self.rate_limit_data = {}
        
        print("🚀 Gemini 2.5 Flash Validation Suite")
        print(f"🔑 API Key: {self.api_keys['gemini'][:20]}...")
        print(f"🧪 Testing {len(self.models_to_test)} model configurations")
        print(f"🌍 Testing {len(self.test_languages)} languages")
        print("="*80)
    
    async def test_model_availability(self) -> Dict[str, bool]:
        """Test which model names are available and working"""
        print("\n📋 PHASE 1: Model Availability Testing")
        print("-" * 50)
        
        availability_results = {}
        
        for model_config in self.models_to_test:
            print(f"🔄 Testing {model_config.display_name}...")
            
            try:
                gen_config = GenerationConfig(
                    model=model_config.model_name,
                    temperature=0.7,  # Conservative for testing
                    max_tokens=50,    # Small for quick test
                    top_p=0.9
                )
                
                start_time = time.time()
                result = await self.llm_interface.generate_dream(
                    "Write a very short dream about flying.",
                    gen_config,
                    None
                )
                end_time = time.time()
                
                availability_results[model_config.model_name] = True
                response_time = end_time - start_time
                
                print(f"  ✅ SUCCESS: {model_config.display_name}")
                print(f"     Response time: {response_time:.2f}s")
                print(f"     Content length: {len(result)} chars")
                print(f"     Preview: {result[:100]}...")
                
                # Store detailed result
                test_result = TestResult(
                    model=model_config.model_name,
                    success=True,
                    response_time=response_time,
                    dream_content=result,
                    char_count=len(result),
                    word_count=len(result.split()),
                    timestamp=datetime.now().isoformat()
                )
                self.results.append(test_result)
                
            except Exception as e:
                availability_results[model_config.model_name] = False
                print(f"  ❌ FAILED: {model_config.display_name}")
                print(f"     Error: {str(e)[:100]}...")
                
                # Store failure result
                test_result = TestResult(
                    model=model_config.model_name,
                    success=False,
                    response_time=0,
                    dream_content="",
                    error_message=str(e),
                    timestamp=datetime.now().isoformat()
                )
                self.results.append(test_result)
            
            # Rate limiting delay
            print("     Waiting 35s for rate limiting...")
            await asyncio.sleep(35)
        
        return availability_results
    
    async def benchmark_performance(self, working_models: List[str]) -> Dict[str, Dict]:
        """Benchmark performance of working models"""
        print("\n⚡ PHASE 2: Performance Benchmarking")
        print("-" * 50)
        
        benchmark_results = {}
        
        # Test prompt for consistency
        test_prompt = LANGUAGE_CONFIG['english']['prompt']
        
        for model_name in working_models:
            print(f"📊 Benchmarking {model_name}...")
            
            model_results = {
                'response_times': [],
                'char_counts': [],
                'word_counts': [],
                'dreams': []
            }
            
            # Run 3 tests per model for statistical significance
            for i in range(3):
                try:
                    gen_config = GenerationConfig(
                        model=model_name,
                        temperature=1.1,
                        max_tokens=1000,
                        top_p=0.98
                    )
                    
                    start_time = time.time()
                    dream = await self.llm_interface.generate_dream(
                        test_prompt,
                        gen_config,
                        None
                    )
                    end_time = time.time()
                    
                    response_time = end_time - start_time
                    char_count = len(dream)
                    word_count = len(dream.split())
                    
                    model_results['response_times'].append(response_time)
                    model_results['char_counts'].append(char_count)
                    model_results['word_counts'].append(word_count)
                    model_results['dreams'].append(dream)
                    
                    print(f"  Test {i+1}/3: {response_time:.2f}s, {char_count} chars, {word_count} words")
                    
                    # Rate limiting delay
                    if i < 2:  # Don't wait after last test
                        await asyncio.sleep(35)
                        
                except Exception as e:
                    print(f"  ❌ Test {i+1}/3 failed: {str(e)[:50]}...")
                    continue
            
            # Calculate statistics
            if model_results['response_times']:
                benchmark_results[model_name] = {
                    'avg_response_time': statistics.mean(model_results['response_times']),
                    'min_response_time': min(model_results['response_times']),
                    'max_response_time': max(model_results['response_times']),
                    'avg_char_count': statistics.mean(model_results['char_counts']),
                    'avg_word_count': statistics.mean(model_results['word_counts']),
                    'success_rate': len(model_results['response_times']) / 3 * 100,
                    'sample_dreams': model_results['dreams']
                }
                
                print(f"  📈 Results: {benchmark_results[model_name]['avg_response_time']:.2f}s avg, "
                      f"{benchmark_results[model_name]['avg_char_count']:.0f} chars avg")
            else:
                benchmark_results[model_name] = {'error': 'All tests failed'}
                print(f"  ❌ All benchmark tests failed for {model_name}")
        
        self.performance_data = benchmark_results
        return benchmark_results
    
    async def test_rate_limiting(self, best_model: str) -> Dict[str, Any]:
        """Test rate limiting behavior and find optimal delays"""
        print(f"\n🚦 PHASE 3: Rate Limiting Analysis for {best_model}")
        print("-" * 50)
        
        rate_limit_results = {
            'successful_requests': 0,
            'failed_requests': 0,
            'rate_limit_errors': 0,
            'optimal_delay': 30,
            'test_delays': []
        }
        
        # Test different delay intervals
        test_delays = [25, 30, 35]  # Test around the current 31s delay
        
        for delay in test_delays:
            print(f"🔄 Testing {delay}s delay interval...")
            
            delay_results = {
                'delay': delay,
                'successes': 0,
                'failures': 0,
                'rate_limits': 0,
                'avg_response_time': 0
            }
            
            response_times = []
            
            # Test 3 consecutive requests with this delay
            for i in range(3):
                try:
                    gen_config = GenerationConfig(
                        model=best_model,
                        temperature=1.0,
                        max_tokens=200,  # Smaller for rate limit testing
                        top_p=0.95
                    )
                    
                    start_time = time.time()
                    dream = await self.llm_interface.generate_dream(
                        "Write a short dream about water.",
                        gen_config,
                        None
                    )
                    end_time = time.time()
                    
                    response_time = end_time - start_time
                    response_times.append(response_time)
                    delay_results['successes'] += 1
                    
                    print(f"  ✅ Request {i+1}/3: {response_time:.2f}s")
                    
                    if i < 2:  # Don't wait after last request
                        await asyncio.sleep(delay)
                        
                except Exception as e:
                    error_str = str(e).lower()
                    if 'rate limit' in error_str or 'quota' in error_str:
                        delay_results['rate_limits'] += 1
                        print(f"  🛑 Request {i+1}/3: Rate limited")
                    else:
                        delay_results['failures'] += 1
                        print(f"  ❌ Request {i+1}/3: {str(e)[:50]}...")
                    
                    if i < 2:
                        await asyncio.sleep(delay + 10)  # Extra delay after error
            
            if response_times:
                delay_results['avg_response_time'] = statistics.mean(response_times)
            
            rate_limit_results['test_delays'].append(delay_results)
            
            success_rate = delay_results['successes'] / 3 * 100
            print(f"  📊 {delay}s delay: {success_rate:.0f}% success rate, "
                  f"{delay_results['rate_limits']} rate limits")
            
            # Wait longer between delay tests
            await asyncio.sleep(60)
        
        # Determine optimal delay
        best_delay = 30
        best_success_rate = 0
        
        for delay_test in rate_limit_results['test_delays']:
            success_rate = delay_test['successes'] / 3
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_delay = delay_test['delay']
        
        rate_limit_results['optimal_delay'] = best_delay
        rate_limit_results['best_success_rate'] = best_success_rate * 100
        
        self.rate_limit_data = rate_limit_results
        return rate_limit_results
    
    async def test_multilingual_quality(self, best_model: str) -> Dict[str, Any]:
        """Test dream generation quality across languages"""
        print(f"\n🌍 PHASE 4: Multilingual Quality Assessment for {best_model}")
        print("-" * 50)
        
        quality_results = {}
        
        for language in self.test_languages:
            print(f"🔄 Testing {language.title()} dream generation...")
            
            config = LANGUAGE_CONFIG[language]
            prompt = config['prompt']
            
            try:
                gen_config = GenerationConfig(
                    model=best_model,
                    temperature=1.1,
                    max_tokens=1000,
                    top_p=0.98
                )
                
                start_time = time.time()
                dream = await self.llm_interface.generate_dream(prompt, gen_config, None)
                end_time = time.time()
                
                quality_results[language] = {
                    'success': True,
                    'response_time': end_time - start_time,
                    'char_count': len(dream),
                    'word_count': len(dream.split()),
                    'dream_content': dream,
                    'language_code': config['code'],
                    'script': config['script']
                }
                
                print(f"  ✅ {language.title()}: {len(dream)} chars, {len(dream.split())} words")
                print(f"     Preview: {dream[:100]}...")
                
            except Exception as e:
                quality_results[language] = {
                    'success': False,
                    'error': str(e),
                    'language_code': config['code'],
                    'script': config['script']
                }
                print(f"  ❌ {language.title()}: {str(e)[:50]}...")
            
            # Rate limiting delay
            await asyncio.sleep(35)
        
        return quality_results
    
    def generate_report(self, availability: Dict, performance: Dict, 
                       rate_limits: Dict, quality: Dict) -> str:
        """Generate comprehensive test report"""
        
        report = []
        report.append("=" * 80)
        report.append("🧪 GEMINI 2.5 FLASH VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Model Availability Summary
        report.append("📋 MODEL AVAILABILITY RESULTS")
        report.append("-" * 40)
        working_models = []
        for model, available in availability.items():
            status = "✅ WORKING" if available else "❌ FAILED"
            report.append(f"  {model:<30} {status}")
            if available:
                working_models.append(model)
        report.append(f"\n✅ Working models: {len(working_models)}/{len(availability)}")
        report.append("")
        
        # Performance Comparison
        if performance:
            report.append("⚡ PERFORMANCE BENCHMARK RESULTS")
            report.append("-" * 40)
            
            # Find best performing model
            best_model = None
            best_time = float('inf')
            
            for model, data in performance.items():
                if 'avg_response_time' in data:
                    avg_time = data['avg_response_time']
                    avg_chars = data['avg_char_count']
                    success_rate = data['success_rate']
                    
                    report.append(f"  {model}:")
                    report.append(f"    Response Time: {avg_time:.2f}s")
                    report.append(f"    Content Length: {avg_chars:.0f} chars")
                    report.append(f"    Success Rate: {success_rate:.0f}%")
                    report.append("")
                    
                    if avg_time < best_time and success_rate > 66:
                        best_time = avg_time
                        best_model = model
            
            if best_model:
                report.append(f"🏆 RECOMMENDED MODEL: {best_model}")
                report.append(f"   Best performance: {best_time:.2f}s average response time")
                report.append("")
        
        # Rate Limiting Analysis
        if rate_limits:
            report.append("🚦 RATE LIMITING ANALYSIS")
            report.append("-" * 40)
            report.append(f"  Optimal Delay: {rate_limits['optimal_delay']}s")
            report.append(f"  Best Success Rate: {rate_limits.get('best_success_rate', 0):.0f}%")
            report.append("")
            
            for delay_test in rate_limits['test_delays']:
                delay = delay_test['delay']
                successes = delay_test['successes']
                rate_limits_count = delay_test['rate_limits']
                report.append(f"  {delay}s delay: {successes}/3 success, {rate_limits_count} rate limits")
            report.append("")
        
        # Quality Assessment
        if quality:
            report.append("🌍 MULTILINGUAL QUALITY ASSESSMENT")
            report.append("-" * 40)
            for language, data in quality.items():
                if data['success']:
                    report.append(f"  {language.title()}: ✅ {data['char_count']} chars, {data['word_count']} words")
                else:
                    report.append(f"  {language.title()}: ❌ {data.get('error', 'Unknown error')}")
            report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS")
        report.append("-" * 40)
        
        if working_models:
            # Recommend best model
            if 'gemini-2.5-flash' in working_models:
                report.append("✅ UPGRADE TO GEMINI 2.5 FLASH")
                report.append("   - Model is available and working")
                report.append("   - Should provide better performance than 1.5 Pro")
            elif 'gemini-2.5-flash-latest' in working_models:
                report.append("✅ USE GEMINI 2.5 FLASH LATEST")
                report.append("   - Latest version available and working")
            else:
                report.append("⚠️  STICK WITH CURRENT MODEL")
                report.append("   - Gemini 2.5 Flash variants not available")
        
        # Rate limiting recommendations
        if rate_limits and rate_limits.get('optimal_delay'):
            optimal = rate_limits['optimal_delay']
            current = 31
            if optimal < current:
                report.append(f"⚡ REDUCE DELAY to {optimal}s (from {current}s)")
                report.append("   - Can speed up generation without rate limits")
            elif optimal > current:
                report.append(f"🐌 INCREASE DELAY to {optimal}s (from {current}s)")
                report.append("   - Needed to avoid rate limiting")
            else:
                report.append("✅ CURRENT DELAY (31s) is optimal")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    async def save_results(self, report: str):
        """Save all test results and report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(f"gemini_flash_validation_{timestamp}")
        results_dir.mkdir(exist_ok=True)
        
        # Save report
        report_file = results_dir / "validation_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Save detailed results as JSON
        detailed_results = {
            'timestamp': timestamp,
            'test_results': [
                {
                    'model': r.model,
                    'success': r.success,
                    'response_time': r.response_time,
                    'char_count': r.char_count,
                    'word_count': r.word_count,
                    'error_message': r.error_message,
                    'timestamp': r.timestamp
                }
                for r in self.results
            ],
            'performance_data': self.performance_data,
            'rate_limit_data': self.rate_limit_data
        }
        
        json_file = results_dir / "detailed_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {results_dir}/")
        return results_dir
    
    async def run_full_validation(self):
        """Run complete validation suite"""
        print("🚀 Starting Gemini 2.5 Flash Validation Suite...")
        print("This will take approximately 15-20 minutes due to rate limiting.")
        print("")
        
        try:
            # Phase 1: Test model availability
            availability = await self.test_model_availability()
            working_models = [model for model, works in availability.items() if works]
            
            if not working_models:
                print("❌ No working models found! Check your API key and configuration.")
                return
            
            # Phase 2: Performance benchmarking
            performance = await self.benchmark_performance(working_models)
            
            # Find best model for further testing
            best_model = None
            if 'gemini-2.5-flash' in working_models:
                best_model = 'gemini-2.5-flash'
            elif 'gemini-2.5-flash-latest' in working_models:
                best_model = 'gemini-2.5-flash-latest'
            elif working_models:
                best_model = working_models[0]
            
            # Phase 3: Rate limiting analysis
            rate_limits = {}
            if best_model:
                rate_limits = await self.test_rate_limiting(best_model)
            
            # Phase 4: Quality assessment
            quality = {}
            if best_model:
                quality = await self.test_multilingual_quality(best_model)
            
            # Generate and display report
            report = self.generate_report(availability, performance, rate_limits, quality)
            print("\n" + report)
            
            # Save results
            results_dir = await self.save_results(report)
            
            print(f"\n🎉 Validation complete! Check {results_dir}/ for detailed results.")
            
            return {
                'availability': availability,
                'performance': performance,
                'rate_limits': rate_limits,
                'quality': quality,
                'best_model': best_model,
                'results_dir': str(results_dir)
            }
            
        except KeyboardInterrupt:
            print("\n⚠️  Validation interrupted by user")
            return None
        except Exception as e:
            print(f"\n❌ Validation failed: {e}")
            raise

async def main():
    """Main execution function"""
    
    # Check API key
    if not os.getenv('GEMINI_API_KEY'):
        print("❌ Error: GEMINI_API_KEY environment variable not set")
        print("Please set your Google Gemini API key:")
        print("export GEMINI_API_KEY='your-gemini-api-key-here'")
        return
    
    print("🔑 Gemini API key detected")
    
    # Create validator and run tests
    validator = GeminiFlashValidator()
    
    # Confirm before starting
    print("\n📋 This validation will:")
    print("   • Test Gemini 2.5 Flash model availability")
    print("   • Benchmark performance vs current setup")
    print("   • Optimize rate limiting settings")
    print("   • Validate multilingual dream generation")
    print("   • Generate optimization recommendations")
    print(f"\n⏱️  Estimated time: 15-20 minutes")
    print("💰 Estimated cost: ~$0.10-0.20 (very low)")
    
    try:
        input("\nPress Enter to start validation (Ctrl+C to cancel)...")
    except KeyboardInterrupt:
        print("\n👋 Validation cancelled")
        return
    
    # Run validation
    results = await validator.run_full_validation()
    
    if results:
        print("\n✅ Validation completed successfully!")
        print("📊 Use the recommendations to optimize your main generation script.")
    else:
        print("\n⚠️  Validation was interrupted or failed.")

if __name__ == "__main__":
    asyncio.run(main())