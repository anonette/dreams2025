#!/usr/bin/env python3
"""
Test a single dream generation to debug the main script
"""

import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

from optimized_dream_languages import LANGUAGE_CONFIG
from src.models.llm_interface import LLMInterface, GenerationConfig

async def test_single_dream():
    """Test generating a single dream"""
    
    # Setup API keys
    api_keys = {'gemini': os.getenv('GEMINI_API_KEY')}
    
    # Create LLM interface
    llm_interface = LLMInterface(api_keys)
    llm_interface.setup_google_gemini_client()
    
    # Get English prompt
    english_config = LANGUAGE_CONFIG['english']
    prompt = english_config['prompt']
    
    print(f"🔄 Testing single dream generation...")
    print(f"📝 Prompt: {prompt}")
    
    # Create generation config
    gen_config = GenerationConfig(
        model="gemini-1.5-pro-latest",
        temperature=1.1,
        max_tokens=1000,
        top_p=0.98
    )
    
    try:
        # Generate dream
        dream_content = await llm_interface.generate_dream(prompt, gen_config, None)
        
        print(f"✅ Dream generated successfully!")
        print(f"📊 Length: {len(dream_content)} chars, {len(dream_content.split())} words")
        print(f"📝 Content preview: {dream_content[:200]}...")
        
        return dream_content
        
    except Exception as e:
        print(f"❌ Error generating dream: {e}")
        print(f"Error type: {type(e).__name__}")
        raise e

if __name__ == "__main__":
    asyncio.run(test_single_dream())
