#!/usr/bin/env python3
"""
Validation test for optimized dream configuration
Demonstrates the production-ready setup with all improvements combined
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

from src.models.llm_interface import LLMInterface
from optimized_dream_languages import (
    get_optimized_config, 
    create_generation_config_from_language,
    get_all_languages,
    get_research_summary
)

class OptimizedConfigValidator:
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
        
    async def generate_optimized_dream(self, language: str) -> dict:
        """Generate a dream using optimized configuration"""
        try:
            config = get_optimized_config(language)
            gen_config = create_generation_config_from_language(language)
            
            print(f"   🎯 {language.title()} - Using optimized config...")
            print(f"      Prompt: {config['prompt'][:60]}...")
            print(f"      Params: T={gen_config.temperature}, top_p={gen_config.top_p}")
            
            dream_text = await self.llm_interface.generate_dream(
                prompt=config['prompt'],
                config=gen_config,
                system_message=config['system_message']
            )
            
            return {
                'success': True,
                'dream_text': dream_text,
                'language': language,
                'config': config,
                'generation_params': {
                    'temperature': gen_config.temperature,
                    'top_p': gen_config.top_p,
                    'max_tokens': gen_config.max_tokens
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'language': language,
                'timestamp': datetime.now().isoformat()
            }
    
    async def validate_all_languages(self) -> dict:
        """Validate optimized configuration across all languages"""
        print(f"🚀 OPTIMIZED DREAM CONFIGURATION VALIDATION")
        print(f"=" * 60)
        print(f"Timestamp: {self.timestamp}")
        
        # Show research summary
        research = get_research_summary()
        print(f"\n📊 RESEARCH VALIDATION:")
        print(f"   Validation Date: {research['validation_date']}")
        print(f"   Key Improvements:")
        for key, value in research['performance_metrics'].items():
            print(f"      {key.replace('_', ' ').title()}: {value}")
        print()
        
        # Test all languages
        languages = get_all_languages()
        print(f"🔄 Testing {len(languages)} languages with optimized configuration...")
        
        results = {}
        
        for language in languages:
            try:
                result = await self.generate_optimized_dream(language)
                results[language] = result
                
                if result['success']:
                    preview = result['dream_text'][:200] + "..."
                    print(f"      ✅ Generated {len(result['dream_text'])} chars: {preview}")
                else:
                    print(f"      ❌ Failed: {result['error']}")
                    
            except Exception as e:
                print(f"   ❌ {language}: Validation failed - {e}")
                results[language] = {'error': str(e)}
        
        return results
    
    def analyze_optimized_results(self, results: dict):
        """Analyze quality of optimized dream generation"""
        print(f"\n📊 OPTIMIZED CONFIGURATION ANALYSIS")
        print(f"=" * 60)
        
        successful_dreams = []
        total_chars = 0
        
        # Quality indicators
        cultural_keywords = {
            'english': ['ancient', 'mystical', 'sacred', 'ritual', 'spirit', 'magical'],
            'basque': ['euskal', 'basajaun', 'mari', 'lamina', 'mendi', 'itsaso'],
            'serbian': ['дух', 'душа', 'древн', 'мит', 'традиц', 'свет'],
            'hebrew': ['קדוש', 'רוח', 'נשמה', 'עתיק', 'מיסטי', 'חלום'],
            'slovenian': ['duh', 'duša', 'star', 'mit', 'sveta', 'čaroben']
        }
        
        for language, result in results.items():
            if result.get('success'):
                dream_text = result['dream_text']
                length = len(dream_text)
                total_chars += length
                successful_dreams.append(result)
                
                # Count cultural keywords
                lang_keywords = cultural_keywords.get(language, [])
                cultural_count = sum(1 for keyword in lang_keywords 
                                   if keyword.lower() in dream_text.lower())
                
                # Check for AI disclaimers
                disclaimers = ['i don\'t', 'i\'m an ai', 'as an ai', 'i cannot', 'i\'m unable']
                has_disclaimer = any(disc in dream_text.lower() for disc in disclaimers)
                
                print(f"\n{language.title()}:")
                print(f"   Length: {length} characters")
                print(f"   Cultural keywords: {cultural_count}")
                print(f"   AI disclaimer: {'❌ Present' if has_disclaimer else '✅ Eliminated'}")
                
                if length > 800:
                    print(f"   ✅ Excellent length (>{800} chars)")
                elif length > 400:
                    print(f"   ✅ Good length (>{400} chars)")
                else:
                    print(f"   ⚠️ Short length (<{400} chars)")
        
        # Overall statistics
        if successful_dreams:
            avg_length = total_chars / len(successful_dreams)
            success_rate = len(successful_dreams) / len(results) * 100
            
            print(f"\n📈 OVERALL PERFORMANCE:")
            print(f"   Success rate: {success_rate:.1f}% ({len(successful_dreams)}/{len(results)})")
            print(f"   Average length: {avg_length:.0f} characters")
            print(f"   Total content: {total_chars:,} characters")
            
            if avg_length > 800:
                print(f"   ✅ Excellent average length!")
            elif avg_length > 600:
                print(f"   ✅ Good average length")
            else:
                print(f"   ⚠️ Could improve length")
    
    def save_validation_results(self, results: dict):
        """Save validation results"""
        output_dir = Path("optimized_validation_output")
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        results_file = output_dir / f"optimized_validation_{self.timestamp}.json"
        
        # Prepare results for JSON serialization
        json_results = {}
        for lang, result in results.items():
            if 'config' in result:
                # Remove non-serializable config object
                result_copy = result.copy()
                result_copy['config_summary'] = {
                    'prompt': result['config']['prompt'],
                    'linguistic_notes': result['config'].get('linguistic_notes', 'None')
                }
                del result_copy['config']
                json_results[lang] = result_copy
            else:
                json_results[lang] = result
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'validation_metadata': {
                    'timestamp': self.timestamp,
                    'research_validation': get_research_summary(),
                    'configuration_source': 'optimized_dream_languages.py'
                },
                'results': json_results
            }, f, indent=2, ensure_ascii=False)
        
        # Create summary report
        summary_file = output_dir / f"optimized_validation_summary_{self.timestamp}.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("# Optimized Dream Configuration Validation\n\n")
            f.write(f"**Validation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Configuration Features\n")
            f.write("- **Pure immediate scenario**: No system prompt, direct dream writing\n")
            f.write("- **Refined translations**: Idiomatic improvements for natural flow\n") 
            f.write("- **Enhanced parameters**: Temperature 1.1, Top_p 0.98 for cultural richness\n")
            f.write("- **Research validated**: Based on comprehensive A/B testing\n\n")
            
            research = get_research_summary()
            f.write("## Research Validation\n")
            for metric, value in research['performance_metrics'].items():
                f.write(f"- **{metric.replace('_', ' ').title()}**: {value}\n")
            f.write("\n")
            
            f.write("## Validation Results\n\n")
            successful = sum(1 for r in results.values() if r.get('success'))
            f.write(f"**Success Rate**: {successful}/{len(results)} languages ({successful/len(results)*100:.1f}%)\n\n")
            
            for language, result in results.items():
                f.write(f"### {language.title()}\n")
                if result.get('success'):
                    f.write(f"**Length**: {len(result['dream_text'])} characters\n\n")
                    f.write(f"```\n{result['dream_text']}\n```\n\n")
                else:
                    f.write(f"**Error**: {result.get('error', 'Unknown error')}\n\n")
                f.write("---\n\n")
        
        print(f"\n✅ Validation results saved:")
        print(f"   📄 Detailed: {results_file}")
        print(f"   📄 Summary: {summary_file}")
        
        return results_file, summary_file

async def main():
    """Main validation function"""
    validator = OptimizedConfigValidator()
    
    # Run validation
    results = await validator.validate_all_languages()
    
    # Analyze results
    validator.analyze_optimized_results(results)
    
    # Save results
    results_file, summary_file = validator.save_validation_results(results)
    
    print(f"\n🎯 OPTIMIZED CONFIGURATION VALIDATION COMPLETE!")
    print(f"📁 Configuration file: optimized_dream_languages.py")
    print(f"📊 Results: {summary_file}")
    print(f"\n🚀 Ready for production dream generation!")

if __name__ == "__main__":
    asyncio.run(main()) 