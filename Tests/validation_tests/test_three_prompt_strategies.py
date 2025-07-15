#!/usr/bin/env python3
"""
Test script comparing three different prompt strategies for dream generation:
1. Original research prompt
2. Experimental multi-voice prompt  
3. Immediate dream scenario prompt
"""

import asyncio
import sys
import json
import os
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append('src')
sys.path.append('.')

from src.models.llm_interface import LLMInterface, GenerationConfig
from src.config.languages import LANGUAGE_CONFIG as ORIGINAL_CONFIG
from experimental_languages import LANGUAGE_CONFIG as EXPERIMENTAL_CONFIG
from immediate_dream_languages import LANGUAGE_CONFIG as IMMEDIATE_CONFIG

class ThreePromptStrategyTester:
    def __init__(self):
        # Initialize API keys
        api_keys = {
            'openai': os.getenv('OPENAI_API_KEY'),
            'anthropic': os.getenv('ANTHROPIC_API_KEY'),
            'google': os.getenv('GOOGLE_API_KEY')
        }
        
        self.llm_interface = LLMInterface(api_keys)
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Strategy configurations
        self.strategies = {
            'original': {
                'name': 'Original Research',
                'config': ORIGINAL_CONFIG,
                'description': 'Cross-cultural dream study participant'
            },
            'experimental': {
                'name': 'Multi-Voice Cultural',
                'config': EXPERIMENTAL_CONFIG,
                'description': 'Shifting voices, cultures, and imaginaries'
            },
            'immediate': {
                'name': 'Immediate Dream Scenario',
                'config': IMMEDIATE_CONFIG,
                'description': 'Just woke up and wrote dream down'
            }
        }
        
    async def generate_dream_with_strategy(self, language: str, strategy_name: str) -> dict:
        """Generate a single dream with specified strategy"""
        strategy = self.strategies[strategy_name]
        config = strategy['config']
        
        prompt = config[language]['prompt']
        system_message = config[language]['system_message']
        
        generation_config = GenerationConfig(
            model='gpt-4o',
            temperature=1.0,
            max_tokens=1000,
            top_p=0.95
        )
        
        try:
            dream_text = await self.llm_interface.generate_dream(
                prompt, generation_config, system_message
            )
            
            return {
                'success': True,
                'dream_text': dream_text,
                'language': language,
                'strategy': strategy_name,
                'strategy_name': strategy['name'],
                'prompt': prompt,
                'system_message': system_message,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'language': language,
                'strategy': strategy_name,
                'strategy_name': strategy['name'],
                'timestamp': datetime.now().isoformat()
            }
    
    async def test_language_all_strategies(self, language: str) -> dict:
        """Test all three strategies for a single language"""
        print(f"\n🔄 Testing {language} with all three strategies...")
        
        language_results = {'language': language}
        
        for strategy_name, strategy_info in self.strategies.items():
            print(f"   🎯 {strategy_info['name']}...")
            result = await self.generate_dream_with_strategy(language, strategy_name)
            language_results[strategy_name] = result
            
            # Show preview
            if result['success']:
                preview = result['dream_text'][:150] + "..."
                print(f"      ✅ {preview}")
            else:
                print(f"      ❌ {result['error']}")
        
        return language_results
    
    async def run_comprehensive_test(self, languages: list = None) -> dict:
        """Run comprehensive test across all strategies and languages"""
        if languages is None:
            languages = ['english', 'basque', 'hebrew']  # Test subset first
        
        print(f"🚀 THREE PROMPT STRATEGIES COMPARISON TEST")
        print(f"=" * 60)
        print(f"Testing languages: {', '.join(languages)}")
        print(f"Timestamp: {self.timestamp}")
        print()
        
        # Show strategy descriptions
        print("📋 STRATEGIES BEING TESTED:")
        for strategy_name, strategy_info in self.strategies.items():
            print(f"   {strategy_info['name']}: {strategy_info['description']}")
        print()
        
        results = {}
        
        for language in languages:
            try:
                language_results = await self.test_language_all_strategies(language)
                results[language] = language_results
                
            except Exception as e:
                print(f"   ❌ {language}: Test failed - {e}")
                results[language] = {'error': str(e)}
        
        return results
    
    def save_results(self, results: dict):
        """Save comprehensive test results"""
        output_dir = Path("three_strategies_output")
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        results_file = output_dir / f"three_strategies_results_{self.timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save readable comparison
        comparison_file = output_dir / f"three_strategies_comparison_{self.timestamp}.md"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            f.write("# Three Prompt Strategies Comparison\n\n")
            f.write(f"**Test Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Strategy descriptions
            f.write("## Strategies Tested\n\n")
            for strategy_name, strategy_info in self.strategies.items():
                f.write(f"### {strategy_info['name']}\n")
                f.write(f"**Description**: {strategy_info['description']}\n\n")
                
                # Show system message
                sample_config = strategy_info['config']['english']
                if sample_config['system_message']:
                    f.write("**System Message**:\n")
                    f.write(f"```\n{sample_config['system_message']}\n```\n\n")
                else:
                    f.write("**System Message**: None\n\n")
                
                # Show prompt
                f.write("**User Prompt**:\n")
                f.write(f"```\n{sample_config['prompt']}\n```\n\n")
                f.write("---\n\n")
            
            # Results by language
            f.write("## Results by Language\n\n")
            
            for language, lang_results in results.items():
                if 'error' in lang_results:
                    f.write(f"### {language.title()}\n")
                    f.write(f"**Error**: {lang_results['error']}\n\n")
                    continue
                
                f.write(f"### {language.title()}\n\n")
                
                for strategy_name, strategy_info in self.strategies.items():
                    if strategy_name not in lang_results:
                        continue
                    
                    result = lang_results[strategy_name]
                    f.write(f"#### {strategy_info['name']}\n")
                    
                    if result['success']:
                        f.write(f"**Length**: {len(result['dream_text'])} characters\n\n")
                        f.write(f"```\n{result['dream_text']}\n```\n\n")
                    else:
                        f.write(f"**Error**: {result['error']}\n\n")
                
                f.write("---\n\n")
        
        print(f"✅ Results saved:")
        print(f"   📄 Detailed: {results_file}")
        print(f"   📄 Comparison: {comparison_file}")
        
        return results_file, comparison_file
    
    def analyze_strategy_differences(self, results: dict):
        """Analyze differences between the three strategies"""
        print("\n📊 ANALYZING STRATEGY DIFFERENCES")
        print("=" * 60)
        
        strategy_analysis = {}
        
        for strategy_name, strategy_info in self.strategies.items():
            strategy_analysis[strategy_name] = {
                'name': strategy_info['name'],
                'total_length': 0,
                'successful_dreams': 0,
                'has_disclaimers': 0,
                'avg_length': 0,
                'languages': []
            }
        
        # Analyze each language
        for language, lang_results in results.items():
            if 'error' in lang_results:
                continue
            
            print(f"\n{language.title()}:")
            
            for strategy_name, strategy_info in self.strategies.items():
                if strategy_name not in lang_results:
                    continue
                
                result = lang_results[strategy_name]
                analysis = strategy_analysis[strategy_name]
                
                if result['success']:
                    text = result['dream_text']
                    text_length = len(text)
                    
                    # Check for AI disclaimers
                    has_disclaimer = any(phrase in text.lower() for phrase in [
                        "i don't actually", "i'm an ai", "as an ai", "i don't have", 
                        "i cannot", "i'm unable", "as a language model"
                    ])
                    
                    analysis['total_length'] += text_length
                    analysis['successful_dreams'] += 1
                    analysis['languages'].append(language)
                    
                    if has_disclaimer:
                        analysis['has_disclaimers'] += 1
                    
                    print(f"   {strategy_info['name']}: {text_length} chars, disclaimer: {has_disclaimer}")
                else:
                    print(f"   {strategy_info['name']}: FAILED")
        
        # Calculate averages and show summary
        print(f"\n📈 STRATEGY SUMMARY:")
        print(f"=" * 40)
        
        for strategy_name, analysis in strategy_analysis.items():
            if analysis['successful_dreams'] > 0:
                analysis['avg_length'] = analysis['total_length'] / analysis['successful_dreams']
                disclaimer_rate = (analysis['has_disclaimers'] / analysis['successful_dreams']) * 100
                
                print(f"\n{analysis['name']}:")
                print(f"   Success rate: {analysis['successful_dreams']}/{len(results)} languages")
                print(f"   Average length: {analysis['avg_length']:.0f} characters")
                print(f"   Disclaimer rate: {disclaimer_rate:.1f}%")
        
        return strategy_analysis

async def main():
    """Main test function"""
    tester = ThreePromptStrategyTester()
    
    # Run comprehensive test
    results = await tester.run_comprehensive_test(['english', 'basque', 'hebrew'])
    
    # Save results
    results_file, comparison_file = tester.save_results(results)
    
    # Analyze differences
    analysis = tester.analyze_strategy_differences(results)
    
    print(f"\n🎯 THREE STRATEGIES TEST COMPLETE!")
    print(f"Check {comparison_file} for detailed comparison")

if __name__ == "__main__":
    asyncio.run(main()) 