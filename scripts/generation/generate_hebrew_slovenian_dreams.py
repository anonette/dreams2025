#!/usr/bin/env python3
"""
Generate exactly 100 dreams for Slovenian and Hebrew languages
Uses the proven OptimizedBatchV2 configuration
"""

import asyncio
import os
from datetime import datetime
from optimized_batch_v2 import OptimizedBatchV2

async def generate_hebrew_slovenian_dreams():
    """Generate exactly 100 dreams for Hebrew and Slovenian languages"""
    
    print("🎯 GENERATING 100 DREAMS FOR HEBREW AND SLOVENIAN")
    print("=" * 50)
    print("📊 Target: 100 dreams per language (200 total)")
    print("🌍 Languages: Hebrew, Slovenian")
    print("⚙️  Using proven OptimizedBatchV2 configuration")
    print("🔄 Will resume from existing progress")
    print()
    
    # Load API keys
    api_keys = {
        'openai': os.getenv('OPENAI_API_KEY'),
        'anthropic': os.getenv('ANTHROPIC_API_KEY')
    }
    
    if not api_keys['openai']:
        print("❌ OPENAI_API_KEY not found in environment variables")
        return
    
    # Create batch generator
    batch_generator = OptimizedBatchV2(api_keys)
    
    # Configure for 100 dreams per language
    batch_generator.config.dreams_per_language = 100
    batch_generator.config.total_target_dreams = 200  # 100 × 2 languages
    
    # Save to main logs directory (same as existing dreams)
    batch_generator.base_logs_dir = 'logs'
    
    # Create session ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_generator.session_id = f"HEB_SLO_{timestamp}"
    
    print(f"🆔 Session ID: {batch_generator.session_id}")
    print(f"📁 Logs directory: {batch_generator.base_logs_dir}/")
    print()
    
    # Generate for Hebrew and Slovenian only
    target_languages = ['hebrew', 'slovenian']
    
    try:
        results = {}
        for language in target_languages:
            print(f"🔄 Generating dreams for {language.title()}...")
            result = await batch_generator.generate_language_batch(language)
            results[language] = result
            print(f"✅ {language.title()}: {result['successful_dreams']} dreams generated")
        
        print("\n🎉 GENERATION COMPLETE!")
        print(f"📊 Results:")
        
        total_new_dreams = 0
        for lang, result in results.items():
            new_dreams = result['successful_dreams']
            total_new_dreams += new_dreams
            print(f"  {lang.title():>10}: {new_dreams} dreams")
        
        print(f"\n✅ Total new dreams: {total_new_dreams}")
        print(f"🏁 You now have 100 additional dreams for Hebrew and Slovenian!")
        
    except KeyboardInterrupt:
        print("\n🛑 Generation interrupted")
        print("💾 Progress has been saved - you can resume by running this script again")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(generate_hebrew_slovenian_dreams()) 