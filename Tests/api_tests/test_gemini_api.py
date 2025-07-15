#!/usr/bin/env python3
"""
Simple test of Gemini API to debug the issue
"""

import os
from dotenv import load_dotenv
load_dotenv()

try:
    import openai
    print("✅ OpenAI library available")
except ImportError:
    print("❌ OpenAI library not available")
    exit(1)

# Test 1: Check API key
gemini_key = os.getenv('GEMINI_API_KEY')
if gemini_key:
    print(f"✅ Gemini API key found: {gemini_key[:20]}...")
else:
    print("❌ No Gemini API key found")
    exit(1)

# Test 2: Try to create client
try:
    client = openai.OpenAI(
        api_key=gemini_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    print("✅ OpenAI client created with Gemini endpoint")
except Exception as e:
    print(f"❌ Failed to create client: {e}")
    exit(1)

# Test 3: Try a simple API call
try:
    print("🔄 Testing API call...")
    response = client.chat.completions.create(
        model="gemini-1.5-pro-latest",
        messages=[
            {"role": "user", "content": "Write a short dream about flying."}
        ],
        temperature=1.0,
        max_tokens=100
    )
    
    dream = response.choices[0].message.content
    print(f"✅ API call successful!")
    print(f"📝 Generated dream: {dream[:100]}...")
    
except Exception as e:
    print(f"❌ API call failed: {e}")
    print(f"Error type: {type(e).__name__}")
