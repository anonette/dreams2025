#!/usr/bin/env python3
"""
Check which API keys are available and working.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_api_keys():
    """Check which API keys are available."""
    print("🔑 API Key Status Check")
    print("=" * 30)
    
    keys = {
        'OpenAI': os.getenv('OPENAI_API_KEY'),
        'Anthropic (Claude)': os.getenv('ANTHROPIC_API_KEY'),
        'OpenRouter': os.getenv('OPENROUTER_API_KEY')
    }
    
    for provider, key in keys.items():
        if key:
            # Check if key looks valid
            if key.startswith('sk-'):
                print(f"✅ {provider}: Key found and looks valid")
            else:
                print(f"⚠️  {provider}: Key found but format looks unusual")
        else:
            print(f"❌ {provider}: No key found")
    
    print("\n📋 Summary:")
    available = [provider for provider, key in keys.items() if key]
    
    if available:
        print(f"✅ Available providers: {', '.join(available)}")
        
        if 'OpenAI' in available:
            print("🚀 Ready to run your first test with GPT-4o!")
        else:
            print("⚠️  You need OpenAI key for the current test")
    else:
        print("❌ No API keys found!")
        print("\n📝 To get started:")
        print("1. Get OpenAI key from: https://platform.openai.com")
        print("2. Add to .env file: OPENAI_API_KEY=your-key-here")
    
    print(f"\n💡 Optional keys for future experiments:")
    print("- Anthropic: https://console.anthropic.com")
    print("- OpenRouter: https://openrouter.ai")

if __name__ == "__main__":
    check_api_keys() 