#!/usr/bin/env python3
"""
Test script comparing parameter settings for cultural richness:
1. Standard parameters (temperature=1.0, top_p=0.95)
2. Enhanced parameters (temperature=1.1, top_p=0.98) for deeper cultural content
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
from pure_immediate_dream_languages import LANGUAGE_CONFIG as PURE_IMMEDIATE_CONFIG

class EnhancedParameterTester:
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
        
        # Parameter configurations
        self.parameter_sets = {
            'standard': {
                'name': 'Standard Parameters',
                'temperature': 1.0,
                'top_p': 0.95,
                'description': 'Balanced creativity and coherence'
            },
            'enhanced': {
                'name': 'Enhanced Cultural Parameters',
                'temperature': 1.1,
                'top_p': 0.98,
                'description': 'Higher temperature + wider top_p for cultural richness'
            }
        }
        
    async def generate_dream_with_params(self, language: str, param_set_name: str) -> dict:
        """Generate a single dream with specified parameters"""
        param_set = self.parameter_sets[param_set_name]
        
        prompt = PURE_IMMEDIATE_CONFIG[language]['prompt']
        system_message = PURE_IMMEDIATE_CONFIG[language]['system_message']
        
        generation_config = GenerationConfig(
            model='gpt-4o',
            temperature=param_set['temperature'],
            max_tokens=1000,
            top_p=param_set['top_p']
        )
        
        try:
            dream_text = await self.llm_interface.generate_dream(
                prompt, generation_config, system_message
            )
            
            return {
                'success': True,
                'dream_text': dream_text,
                'language': language,
                'param_set': param_set_name,
                'param_set_name': param_set['name'],
                'temperature': param_set['temperature'],
                'top_p': param_set['top_p'],
                'prompt': prompt,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'language': language,
                'param_set': param_set_name,
                'param_set_name': param_set['name'],
                'timestamp': datetime.now().isoformat()
            }
    
    async def test_language_both_params(self, language: str) -> dict:
        """Test both parameter sets for a single language"""
        print(f"\n🔄 Testing {language} with both parameter sets...")
        
        language_results = {'language': language}
        
        for param_set_name, param_info in self.parameter_sets.items():
            print(f"   🎯 {param_info['name']} (T={param_info['temperature']}, top_p={param_info['top_p']})...")
            result = await self.generate_dream_with_params(language, param_set_name)
            language_results[param_set_name] = result
            
            # Show preview
            if result['success']:
                preview = result['dream_text'][:200] + "..."
                print(f"      ✅ {preview}")
            else:
                print(f"      ❌ {result['error']}")
        
        return language_results
    
    async def run_parameter_comparison(self, languages: list = None) -> dict:
        """Run parameter comparison test"""
        if languages is None:
            languages = ['english', 'basque', 'hebrew']
        
        print(f"🚀 ENHANCED PARAMETER COMPARISON TEST")
        print(f"=" * 60)
        print(f"Testing languages: {', '.join(languages)}")
        print(f"Timestamp: {self.timestamp}")
        print()
        
        # Show parameter descriptions
        print("📋 PARAMETER SETS BEING TESTED:")
        for param_name, param_info in self.parameter_sets.items():
            print(f"   {param_info['name']}: {param_info['description']}")
            print(f"      Temperature: {param_info['temperature']}, Top_p: {param_info['top_p']}")
        print()
        
        results = {}
        
        for language in languages:
            try:
                language_results = await self.test_language_both_params(language)
                results[language] = language_results
                
            except Exception as e:
                print(f"   ❌ {language}: Test failed - {e}")
                results[language] = {'error': str(e)}
        
        return results
    
    def analyze_cultural_richness(self, results: dict):
        """Analyze cultural richness and symbolic density"""
        print("\n📊 ANALYZING CULTURAL RICHNESS")
        print("=" * 60)
        
        # Cultural/symbolic keywords by language
        cultural_keywords = {
            'english': ['myth', 'ancient', 'ritual', 'spirit', 'sacred', 'mystical', 'folklore', 'legend'],
            'basque': ['euskal', 'basajaun', 'mari', 'lamina', 'sorgin', 'etxe', 'mendi', 'itsaso'],
            'hebrew': ['קדוש', 'נשמה', 'רוח', 'מיסטי', 'קבלה', 'תפילה', 'שמש', 'ירח', 'כוכב'],
            'serbian': ['дух', 'душа', 'света', 'древн', 'мит', 'легенд', 'рит', 'тради'],
            'slovenian': ['duh', 'duša', 'sveta', 'star', 'mit', 'legend', 'ritual', 'tradicija']
        }
        
        for language, lang_results in results.items():
            if 'error' in lang_results:
                continue
            
            print(f"\n{language.title()}:")
            
            standard_result = lang_results.get('standard', {})
            enhanced_result = lang_results.get('enhanced', {})
            
            if standard_result.get('success') and enhanced_result.get('success'):
                standard_text = standard_result['dream_text'].lower()
                enhanced_text = enhanced_result['dream_text'].lower()
                
                # Count cultural keywords
                lang_keywords = cultural_keywords.get(language, [])
                standard_cultural = sum(1 for keyword in lang_keywords if keyword in standard_text)
                enhanced_cultural = sum(1 for keyword in lang_keywords if keyword in enhanced_text)
                
                # Count unique words (vocabulary richness)
                standard_words = set(standard_text.split())
                enhanced_words = set(enhanced_text.split())
                
                print(f"   Standard (T=1.0): {len(standard_result['dream_text'])} chars, {standard_cultural} cultural keywords, {len(standard_words)} unique words")
                print(f"   Enhanced (T=1.1): {len(enhanced_result['dream_text'])} chars, {enhanced_cultural} cultural keywords, {len(enhanced_words)} unique words")
                
                cultural_improvement = enhanced_cultural - standard_cultural
                vocab_improvement = len(enhanced_words) - len(standard_words)
                
                if cultural_improvement > 0:
                    print(f"   ✅ Cultural richness improved: +{cultural_improvement} keywords")
                elif cultural_improvement < 0:
                    print(f"   ⚠️ Cultural richness decreased: {cultural_improvement} keywords")
                else:
                    print(f"   ➡️ Cultural richness unchanged")
                
                if vocab_improvement > 0:
                    print(f"   ✅ Vocabulary richness improved: +{vocab_improvement} unique words")
                elif vocab_improvement < 0:
                    print(f"   ⚠️ Vocabulary richness decreased: {vocab_improvement} unique words")
                else:
                    print(f"   ➡️ Vocabulary richness unchanged")
                
    def save_results(self, results: dict):
        """Save parameter comparison results"""
        output_dir = Path("enhanced_parameters_output")
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        results_file = output_dir / f"enhanced_params_results_{self.timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save readable comparison
        comparison_file = output_dir / f"enhanced_params_comparison_{self.timestamp}.md"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            f.write("# Enhanced Parameters Comparison\n\n")
            f.write(f"**Test Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Research Question\n")
            f.write("Does increasing temperature to 1.1 and top_p to 0.98 enhance cultural richness and symbolic density in AI-generated dreams?\n\n")
            
            # Parameter descriptions
            f.write("## Parameter Sets Tested\n\n")
            for param_name, param_info in self.parameter_sets.items():
                f.write(f"### {param_info['name']}\n")
                f.write(f"**Temperature**: {param_info['temperature']}\n")
                f.write(f"**Top_p**: {param_info['top_p']}\n")
                f.write(f"**Description**: {param_info['description']}\n\n")
                f.write("---\n\n")
            
            # Results by language
            f.write("## Results by Language\n\n")
            
            for language, lang_results in results.items():
                if 'error' in lang_results:
                    f.write(f"### {language.title()}\n")
                    f.write(f"**Error**: {lang_results['error']}\n\n")
                    continue
                
                f.write(f"### {language.title()}\n\n")
                
                for param_name, param_info in self.parameter_sets.items():
                    if param_name not in lang_results:
                        continue
                    
                    result = lang_results[param_name]
                    f.write(f"#### {param_info['name']}\n")
                    f.write(f"**Parameters**: Temperature={param_info['temperature']}, Top_p={param_info['top_p']}\n")
                    
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
    tester = EnhancedParameterTester()
    
    # Run parameter comparison
    results = await tester.run_parameter_comparison(['english', 'basque', 'hebrew'])
    
    # Analyze cultural richness
    tester.analyze_cultural_richness(results)
    
    # Save results
    results_file, comparison_file = tester.save_results(results)
    
    print(f"\n🎯 ENHANCED PARAMETER TEST COMPLETE!")
    print(f"Check {comparison_file} for detailed comparison")

if __name__ == "__main__":
    asyncio.run(main()) 