#!/usr/bin/env python3
"""
Test script for experimental multi-voice dream generation
Compares original vs experimental system prompts
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

class ExperimentalDreamTester:
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
        
    async def generate_dream_with_config(self, language: str, config: dict, config_type: str) -> dict:
        """Generate a single dream with specified configuration"""
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
                'config_type': config_type,
                'prompt': prompt,
                'system_message': system_message[:100] + "..." if len(system_message) > 100 else system_message,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'language': language,
                'config_type': config_type,
                'timestamp': datetime.now().isoformat()
            }
    
    async def test_language_comparison(self, language: str) -> dict:
        """Test both original and experimental prompts for a language"""
        print(f"\n🔄 Testing {language} with both system prompts...")
        
        # Generate with original prompt
        original_result = await self.generate_dream_with_config(
            language, ORIGINAL_CONFIG, 'original'
        )
        
        # Generate with experimental prompt
        experimental_result = await self.generate_dream_with_config(
            language, EXPERIMENTAL_CONFIG, 'experimental'
        )
        
        return {
            'language': language,
            'original': original_result,
            'experimental': experimental_result
        }
    
    async def run_comparison_test(self, languages: list = None) -> dict:
        """Run comparison test for specified languages"""
        if languages is None:
            languages = ['english', 'basque', 'hebrew']  # Test subset first
        
        print(f"🚀 EXPERIMENTAL DREAM GENERATION TEST")
        print(f"=" * 50)
        print(f"Testing languages: {', '.join(languages)}")
        print(f"Timestamp: {self.timestamp}")
        
        results = {}
        
        for language in languages:
            try:
                language_results = await self.test_language_comparison(language)
                results[language] = language_results
                
                # Show preview of results
                if language_results['original']['success']:
                    original_preview = language_results['original']['dream_text'][:200] + "..."
                    print(f"   ✅ Original ({language}): {original_preview}")
                else:
                    print(f"   ❌ Original ({language}): {language_results['original']['error']}")
                
                if language_results['experimental']['success']:
                    experimental_preview = language_results['experimental']['dream_text'][:200] + "..."
                    print(f"   ✅ Experimental ({language}): {experimental_preview}")
                else:
                    print(f"   ❌ Experimental ({language}): {language_results['experimental']['error']}")
                
                print()
                
            except Exception as e:
                print(f"   ❌ {language}: Test failed - {e}")
                results[language] = {'error': str(e)}
        
        return results
    
    def save_results(self, results: dict):
        """Save test results to file"""
        output_dir = Path("experimental_dreams_output")
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        results_file = output_dir / f"experimental_test_results_{self.timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save readable comparison
        comparison_file = output_dir / f"experimental_comparison_{self.timestamp}.md"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            f.write("# Experimental Dream Generation Comparison\n\n")
            f.write(f"**Test Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## System Prompts Compared\n\n")
            f.write("### Original System Prompt\n")
            f.write("```\n")
            f.write(ORIGINAL_CONFIG['english']['system_message'])
            f.write("\n```\n\n")
            
            f.write("### Experimental System Prompt\n")
            f.write("```\n")
            f.write(EXPERIMENTAL_CONFIG['english']['system_message'])
            f.write("\n```\n\n")
            
            f.write("## Results by Language\n\n")
            
            for language, lang_results in results.items():
                if 'error' in lang_results:
                    f.write(f"### {language.title()}\n")
                    f.write(f"**Error**: {lang_results['error']}\n\n")
                    continue
                
                f.write(f"### {language.title()}\n\n")
                
                f.write("#### Original System Prompt Result\n")
                if lang_results['original']['success']:
                    f.write(f"```\n{lang_results['original']['dream_text']}\n```\n\n")
                else:
                    f.write(f"**Error**: {lang_results['original']['error']}\n\n")
                
                f.write("#### Experimental System Prompt Result\n")
                if lang_results['experimental']['success']:
                    f.write(f"```\n{lang_results['experimental']['dream_text']}\n```\n\n")
                else:
                    f.write(f"**Error**: {lang_results['experimental']['error']}\n\n")
                
                f.write("---\n\n")
        
        print(f"✅ Results saved:")
        print(f"   📄 Detailed: {results_file}")
        print(f"   📄 Comparison: {comparison_file}")
        
        return results_file, comparison_file
    
    def analyze_differences(self, results: dict):
        """Analyze differences between original and experimental results"""
        print("\n📊 ANALYZING DIFFERENCES")
        print("=" * 50)
        
        differences = {}
        
        for language, lang_results in results.items():
            if 'error' in lang_results:
                continue
            
            original = lang_results['original']
            experimental = lang_results['experimental']
            
            if not (original['success'] and experimental['success']):
                continue
            
            # Basic metrics
            original_length = len(original['dream_text'])
            experimental_length = len(experimental['dream_text'])
            
            # Check for AI disclaimers
            original_has_disclaimer = any(phrase in original['dream_text'].lower() for phrase in [
                "i don't actually", "i'm an ai", "as an ai", "i don't have", "i cannot"
            ])
            experimental_has_disclaimer = any(phrase in experimental['dream_text'].lower() for phrase in [
                "i don't actually", "i'm an ai", "as an ai", "i don't have", "i cannot"
            ])
            
            differences[language] = {
                'original_length': original_length,
                'experimental_length': experimental_length,
                'length_difference': experimental_length - original_length,
                'original_has_disclaimer': original_has_disclaimer,
                'experimental_has_disclaimer': experimental_has_disclaimer,
                'disclaimer_removed': original_has_disclaimer and not experimental_has_disclaimer
            }
            
            print(f"\n{language.title()}:")
            print(f"   Length: {original_length} → {experimental_length} ({experimental_length - original_length:+d})")
            print(f"   AI Disclaimer: {original_has_disclaimer} → {experimental_has_disclaimer}")
            if original_has_disclaimer and not experimental_has_disclaimer:
                print("   ✅ Experimental prompt eliminated AI disclaimer!")
        
        return differences

async def main():
    """Main test function"""
    tester = ExperimentalDreamTester()
    
    # Run comparison test
    results = await tester.run_comparison_test(['english', 'basque', 'hebrew'])
    
    # Save results
    results_file, comparison_file = tester.save_results(results)
    
    # Analyze differences
    differences = tester.analyze_differences(results)
    
    print(f"\n🎯 EXPERIMENTAL TEST COMPLETE!")
    print(f"Check {comparison_file} for detailed comparison")

if __name__ == "__main__":
    asyncio.run(main()) 