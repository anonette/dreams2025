#!/usr/bin/env python3
"""
Simple runner script for Gemini 2.5 Flash validation
"""

import asyncio
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking requirements...")
    
    # Check API key
    if not os.getenv('GEMINI_API_KEY'):
        print("❌ GEMINI_API_KEY environment variable not set")
        print("\nTo fix this:")
        print("1. Get your API key from: https://ai.google.dev/")
        print("2. Set the environment variable:")
        print("   export GEMINI_API_KEY='your-api-key-here'")
        print("   # or add it to your .env file")
        return False
    
    # Check required files
    required_files = [
        'optimized_dream_languages.py',
        'src/models/llm_interface.py'
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"❌ Required file missing: {file_path}")
            return False
    
    # Check Python packages
    try:
        import openai
        import pandas as pd
        import tqdm
        print("✅ All requirements met")
        return True
    except ImportError as e:
        print(f"❌ Missing Python package: {e}")
        print("\nTo fix this, run:")
        print("pip install openai pandas tqdm python-dotenv")
        return False

async def main():
    """Main runner function"""
    print("🚀 Gemini 2.5 Flash Validation Runner")
    print("=" * 50)
    
    if not check_requirements():
        print("\n❌ Requirements not met. Please fix the issues above.")
        sys.exit(1)
    
    print("\n✅ All requirements met. Starting validation...")
    
    # Import and run the validation
    try:
        from test_gemini_2_5_flash_validation import GeminiFlashValidator
        
        validator = GeminiFlashValidator()
        results = await validator.run_full_validation()
        
        if results and results.get('best_model'):
            print(f"\n🎯 QUICK SUMMARY:")
            print(f"   Best Model: {results['best_model']}")
            if 'rate_limits' in results and results['rate_limits'].get('optimal_delay'):
                print(f"   Optimal Delay: {results['rate_limits']['optimal_delay']}s")
            print(f"   Results saved to: {results['results_dir']}")
            
            # Quick recommendation
            if 'gemini-2.5-flash' in results['best_model']:
                print(f"\n💡 RECOMMENDATION: Upgrade to Gemini 2.5 Flash!")
            else:
                print(f"\n💡 RECOMMENDATION: Check the detailed report for next steps.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required files are in the correct location.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())