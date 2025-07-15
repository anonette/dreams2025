#!/usr/bin/env python3
"""
Test script comparing original vs refined idiomatic translations
to see if linguistic improvements enhance dream generation quality
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

class RefinedTranslationTester:
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
        
        # Original vs refined translations
        self.translation_sets = {
            'original': {
                'name': 'Original Translations',
                'config': {
                    'english': "You woke up and immediately wrote down your dream. What did you write?",
                    'basque': "Esnatu zara eta berehala idatzi duzu zure ametsa. Zer idatzi duzu?",
                    'serbian': "Пробудио си се и одмах записао сан. Шта си написао?",
                    'hebrew': "התעוררת ומיד רשמת את החלום. מה כתבת?",
                    'slovenian': "Zbudil si se in takoj zapisal sanje. Kaj si zapisal?"
                }
            },
            'refined': {
                'name': 'Refined Idiomatic Translations',
                'config': {
                    'english': "You woke up and immediately wrote down your dream. What did you write?",
                    'basque': "Esnatu eta berehala zure ametsa idatzi duzu. Zer idatzi duzu?",
                    'serbian': "Пробудио си се и одмах записао свој сан. Шта си написао?",
                    'hebrew': "התעוררת ומיד כתבת את החלום שלך. מה כתבת?",
                    'slovenian': "Zbudil si se in takoj zapisal svoje sanje. Kaj si zapisal?"
                }
            }
        }
        
    async def generate_dream_with_translation(self, language: str, translation_set_name: str) -> dict:
        """Generate a single dream with specified translation"""
        translation_set = self.translation_sets[translation_set_name]
        prompt = translation_set['config'][language]
        
        # Use enhanced parameters for richer content
        generation_config = GenerationConfig(
            model='gpt-4o',
            temperature=1.1,
            max_tokens=1000,
            top_p=0.98
        )
        
        try:
            dream_text = await self.llm_interface.generate_dream(
                prompt, generation_config, None  # No system message
            )
            
            return {
                'success': True,
                'dream_text': dream_text,
                'language': language,
                'translation_set': translation_set_name,
                'translation_set_name': translation_set['name'],
                'prompt': prompt,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'language': language,
                'translation_set': translation_set_name,
                'translation_set_name': translation_set['name'],
                'timestamp': datetime.now().isoformat()
            }
    
    async def test_language_both_translations(self, language: str) -> dict:
        """Test both translation sets for a single language"""
        print(f"\n🔄 Testing {language} with both translation sets...")
        
        language_results = {'language': language}
        
        for translation_name, translation_info in self.translation_sets.items():
            print(f"   🎯 {translation_info['name']}...")
            result = await self.generate_dream_with_translation(language, translation_name)
            language_results[translation_name] = result
            
            # Show preview
            if result['success']:
                preview = result['dream_text'][:200] + "..."
                print(f"      ✅ {preview}")
            else:
                print(f"      ❌ {result['error']}")
        
        return language_results
    
    async def run_translation_comparison(self, languages: list = None) -> dict:
        """Run translation comparison test"""
        if languages is None:
            # Test languages that had changes
            languages = ['basque', 'serbian', 'hebrew', 'slovenian']
        
        print(f"🚀 REFINED TRANSLATION COMPARISON TEST")
        print(f"=" * 60)
        print(f"Testing languages: {', '.join(languages)}")
        print(f"Timestamp: {self.timestamp}")
        print()
        
        # Show translation differences
        print("📋 TRANSLATION CHANGES:")
        for lang in languages:
            if lang == 'english':
                continue  # No change for English
            original = self.translation_sets['original']['config'][lang]
            refined = self.translation_sets['refined']['config'][lang]
            print(f"   {lang.title()}:")
            print(f"      Original:  {original}")
            print(f"      Refined:   {refined}")
        print()
        
        results = {}
        
        for language in languages:
            try:
                language_results = await self.test_language_both_translations(language)
                results[language] = language_results
                
            except Exception as e:
                print(f"   ❌ {language}: Test failed - {e}")
                results[language] = {'error': str(e)}
        
        return results
    
    def analyze_translation_impact(self, results: dict):
        """Analyze impact of refined translations"""
        print("\n📊 ANALYZING TRANSLATION IMPACT")
        print("=" * 60)
        
        # Analyze naturalness indicators
        naturalness_indicators = [
            'natural', 'flow', 'smooth', 'clear', 'vivid', 'detailed'
        ]
        
        personal_indicators = [
            'i', 'my', 'me', 'myself', 'personal', 'own'
        ]
        
        for language, lang_results in results.items():
            if 'error' in lang_results:
                continue
            
            print(f"\n{language.title()}:")
            
            original_result = lang_results.get('original', {})
            refined_result = lang_results.get('refined', {})
            
            if original_result.get('success') and refined_result.get('success'):
                original_text = original_result['dream_text'].lower()
                refined_text = refined_result['dream_text'].lower()
                
                # Count naturalness indicators
                original_natural = sum(1 for word in naturalness_indicators if word in original_text)
                refined_natural = sum(1 for word in naturalness_indicators if word in refined_text)
                
                # Count personal pronouns (indicating personal engagement)
                original_personal = sum(1 for word in personal_indicators if word in original_text)
                refined_personal = sum(1 for word in personal_indicators if word in refined_text)
                
                # Basic metrics
                original_length = len(original_result['dream_text'])
                refined_length = len(refined_result['dream_text'])
                
                print(f"   Original translation: {original_length} chars, {original_personal} personal words")
                print(f"   Refined translation:  {refined_length} chars, {refined_personal} personal words")
                
                length_change = refined_length - original_length
                personal_change = refined_personal - original_personal
                
                if length_change > 0:
                    print(f"   ✅ Content increased: +{length_change} characters")
                elif length_change < 0:
                    print(f"   ⚠️ Content decreased: {length_change} characters")
                else:
                    print(f"   ➡️ Content length unchanged")
                
                if personal_change > 0:
                    print(f"   ✅ More personal engagement: +{personal_change} personal words")
                elif personal_change < 0:
                    print(f"   ⚠️ Less personal engagement: {personal_change} personal words")
                else:
                    print(f"   ➡️ Personal engagement unchanged")
    
    def save_results(self, results: dict):
        """Save translation comparison results"""
        output_dir = Path("refined_translations_output")
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        results_file = output_dir / f"refined_translations_results_{self.timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save readable comparison
        comparison_file = output_dir / f"refined_translations_comparison_{self.timestamp}.md"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            f.write("# Refined Translations Comparison\n\n")
            f.write(f"**Test Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Research Question\n")
            f.write("Do idiomatic refinements in translation prompts improve the quality and naturalness of AI-generated dreams?\n\n")
            
            # Translation changes
            f.write("## Translation Changes Made\n\n")
            for lang in ['basque', 'serbian', 'hebrew', 'slovenian']:
                original = self.translation_sets['original']['config'][lang]
                refined = self.translation_sets['refined']['config'][lang]
                
                f.write(f"### {lang.title()}\n")
                f.write(f"**Original**: {original}\n\n")
                f.write(f"**Refined**: {refined}\n\n")
                
                # Explain the change
                changes = {
                    'basque': 'Removed subject pronoun "zara" for more natural dream-journal rhythm',
                    'serbian': 'Added "свој" to clarify possession of the dream',
                    'hebrew': 'Added "שלך" for more personal tone',
                    'slovenian': 'Added "svoje" for more idiomatic precision'
                }
                f.write(f"**Change**: {changes[lang]}\n\n")
                f.write("---\n\n")
            
            # Results by language
            f.write("## Results by Language\n\n")
            
            for language, lang_results in results.items():
                if 'error' in lang_results:
                    f.write(f"### {language.title()}\n")
                    f.write(f"**Error**: {lang_results['error']}\n\n")
                    continue
                
                f.write(f"### {language.title()}\n\n")
                
                for translation_name, translation_info in self.translation_sets.items():
                    if translation_name not in lang_results:
                        continue
                    
                    result = lang_results[translation_name]
                    f.write(f"#### {translation_info['name']}\n")
                    f.write(f"**Prompt**: {result.get('prompt', 'N/A')}\n")
                    
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

async def main():
    """Main test function"""
    tester = RefinedTranslationTester()
    
    # Run translation comparison (excluding English since it didn't change)
    results = await tester.run_translation_comparison(['basque', 'serbian', 'hebrew', 'slovenian'])
    
    # Analyze translation impact
    tester.analyze_translation_impact(results)
    
    # Save results
    results_file, comparison_file = tester.save_results(results)
    
    print(f"\n🎯 REFINED TRANSLATION TEST COMPLETE!")
    print(f"Check {comparison_file} for detailed comparison")

if __name__ == "__main__":
    asyncio.run(main()) 