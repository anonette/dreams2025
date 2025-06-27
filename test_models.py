#!/usr/bin/env python3
"""
Test script to check which models are available and working.
"""

import asyncio
import os
import sys
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path
sys.path.append('.')

from src.models.llm_interface import LLMInterface, GenerationConfig

# Test models for each provider
TEST_MODELS = {
    'openai': [
        'gpt-4o',
        'gpt-4o-mini',
        'gpt-4',
        'gpt-4-turbo',
        'gpt-3.5-turbo',
        'gpt-3.5-turbo-16k'
    ],
    'anthropic': [
        'claude-3-opus-20240229',
        'claude-3-sonnet-20240229',
        'claude-3-haiku-20240307'
    ],
    'openrouter': [
        'mistral-7b-instruct',
        'deepseek-chat',
        'meta-llama/llama-2-70b-chat',
        'google/gemini-pro'
    ]
}

async def test_model(interface: LLMInterface, model: str, provider: str) -> bool:
    """Test if a specific model works."""
    try:
        print(f"  Testing {model}...", end=" ")
        
        config = GenerationConfig(
            model=model,
            temperature=0.7,
            max_tokens=50  # Short response for testing
        )
        
        result = await interface.generate_dream(
            "Finish: Last night I dream of flying...", 
            config
        )
        
        if result and len(result.strip()) > 10:
            print("✓ Working")
            return True
        else:
            print("✗ No response")
            return False
            
    except Exception as e:
        print(f"✗ Error: {str(e)[:50]}...")
        return False

async def main():
    """Test all available models."""
    print("Dream Research System - Model Testing")
    print("=" * 50)
    
    # Get API keys
    api_keys = {
        'openai': os.getenv('OPENAI_API_KEY'),
        'anthropic': os.getenv('ANTHROPIC_API_KEY'),
        'openrouter': os.getenv('OPENROUTER_API_KEY')
    }
    
    # Check which providers have API keys
    available_providers = []
    for provider, key in api_keys.items():
        if key:
            available_providers.append(provider)
            print(f"✓ {provider.upper()} API key found")
        else:
            print(f"✗ {provider.upper()} API key missing")
    
    if not available_providers:
        print("\n❌ No API keys found! Please set up your .env file.")
        print("Required keys: OPENAI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY")
        return
    
    print(f"\nTesting models for providers: {', '.join(available_providers)}")
    
    # Initialize interface
    interface = LLMInterface(api_keys)
    
    # Test models by provider
    working_models = []
    
    for provider in available_providers:
        print(f"\n{provider.upper()} Models:")
        print("-" * 20)
        
        for model in TEST_MODELS[provider]:
            if await test_model(interface, model, provider):
                working_models.append(model)
        
        # Add delay between providers
        await asyncio.sleep(1)
    
    # Summary
    print(f"\n" + "=" * 50)
    print(f"SUMMARY: {len(working_models)} models working out of {sum(len(TEST_MODELS[p]) for p in available_providers)} tested")
    
    if working_models:
        print("\n✅ Working models:")
        for model in working_models:
            print(f"  - {model}")
        
        print(f"\n💡 To use these models, update src/pipeline/dream_generator.py:")
        print("   self.models = [")
        for model in working_models[:3]:  # Show first 3
            print(f"       '{model}',")
        print("   ]")
    else:
        print("\n❌ No models are working. Check your API keys and internet connection.")
    
    print(f"\n🔧 For more models, see MODELS.md")

if __name__ == "__main__":
    asyncio.run(main()) 