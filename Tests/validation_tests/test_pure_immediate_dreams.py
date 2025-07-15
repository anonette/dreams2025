#!/usr/bin/env python3
"""
Test script comparing immediate dream scenarios:
1. With interpretation request (original)
2. Pure dream narrative only (modified)
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
from immediate_dream_languages import LANGUAGE_CONFIG as IMMEDIATE_WITH_INTERPRETATION
from pure_immediate_dream_languages import LANGUAGE_CONFIG as PURE_IMMEDIATE_CONFIG

class PureImmediateDreamTester:
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
            'with_interpretation': {
                'name': 'Immediate Dream + Interpretation',
                'config': IMMEDIATE_WITH_INTERPRETATION,
                'description': 'Asks what dream means after writing it down'
            },
            'pure_immediate': {
                'name': 'Pure Immediate Dream',
                'config': PURE_IMMEDIATE_CONFIG,
                'description': 'Just write down the dream, no interpretation'
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
    
    async def test_language_both_strategies(self, language: str) -> dict:
        """Test both immediate dream strategies for a single language"""
        print(f"\n🔄 Testing {language} with both immediate dream strategies...")
        
        language_results = {'language': language}
        
        for strategy_name, strategy_info in self.strategies.items():
            print(f"   🎯 {strategy_info['name']}...")
            result = await self.generate_dream_with_strategy(language, strategy_name)
            language_results[strategy_name] = result
            
            # Show preview
            if result['success']:
                preview = result['dream_text'][:200] + "..."
                print(f"      ✅ {preview}")
            else:
                print(f"      ❌ {result['error']}")
        
        return language_results
    
    async def run_comparison_test(self, languages: list = None) -> dict:
        """Run comparison test for immediate dream strategies"""
        if languages is None:
            languages = ['english', 'basque', 'hebrew']  # Test subset first
        
        print(f"🚀 PURE IMMEDIATE DREAM COMPARISON TEST")
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
                language_results = await self.test_language_both_strategies(language)
                results[language] = language_results
                
            except Exception as e:
                print(f"   ❌ {language}: Test failed - {e}")
                results[language] = {'error': str(e)}
        
        return results
    
    def save_results(self, results: dict):
        """Save test results"""
        output_dir = Path("pure_immediate_dreams_output")
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        results_file = output_dir / f"pure_immediate_results_{self.timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save readable comparison
        comparison_file = output_dir / f"pure_immediate_comparison_{self.timestamp}.md"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            f.write("# Pure Immediate Dream Comparison\n\n")
            f.write(f"**Test Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Strategy descriptions
            f.write("## Strategies Tested\n\n")
            for strategy_name, strategy_info in self.strategies.items():
                f.write(f"### {strategy_info['name']}\n")
                f.write(f"**Description**: {strategy_info['description']}\n\n")
                
                # Show prompt
                sample_config = strategy_info['config']['english']
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
    
    def analyze_interpretation_differences(self, results: dict):
        """Analyze differences between with/without interpretation"""
        print("\n📊 ANALYZING INTERPRETATION DIFFERENCES")
        print("=" * 60)
        
        for language, lang_results in results.items():
            if 'error' in lang_results:
                continue
            
            print(f"\n{language.title()}:")
            
            with_interp = lang_results.get('with_interpretation', {})
            pure_immed = lang_results.get('pure_immediate', {})
            
            if with_interp.get('success') and pure_immed.get('success'):
                with_text = with_interp['dream_text']
                pure_text = pure_immed['dream_text']
                
                # Check for interpretation keywords
                interpretation_keywords = [
                    'what', 'mean', 'interpret', 'symbol', 'represent', 'suggest',
                    'reflect', 'indicate', 'could', 'might', 'perhaps', 'maybe',
                    'analysis', 'explanation', 'understand', 'significance'
                ]
                
                with_interp_count = sum(1 for keyword in interpretation_keywords 
                                      if keyword in with_text.lower())
                pure_interp_count = sum(1 for keyword in interpretation_keywords 
                                      if keyword in pure_text.lower())
                
                print(f"   With interpretation: {len(with_text)} chars, {with_interp_count} interp keywords")
                print(f"   Pure immediate: {len(pure_text)} chars, {pure_interp_count} interp keywords")
                print(f"   Interpretation reduced: {with_interp_count - pure_interp_count} keywords")
                
                if pure_interp_count == 0:
                    print("   ✅ Successfully eliminated interpretation!")
                elif pure_interp_count < with_interp_count:
                    print("   ⚠️ Reduced but not eliminated interpretation")
                else:
                    print("   ❌ No reduction in interpretation")
            else:
                print(f"   ❌ Cannot compare - one or both strategies failed")

async def main():
    """Main test function"""
    tester = PureImmediateDreamTester()
    
    # Run comparison test
    results = await tester.run_comparison_test(['english', 'basque', 'hebrew'])
    
    # Save results
    results_file, comparison_file = tester.save_results(results)
    
    # Analyze differences
    tester.analyze_interpretation_differences(results)
    
    print(f"\n🎯 PURE IMMEDIATE DREAM TEST COMPLETE!")
    print(f"Check {comparison_file} for detailed comparison")

if __name__ == "__main__":
    asyncio.run(main()) 