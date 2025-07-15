#!/usr/bin/env python3
"""
Test different Gemini model names to find the correct one
"""

import os
from dotenv import load_dotenv
load_dotenv()
import openai

# Common Gemini model names to try
model_names = [
    "gemini-1.5-pro",
    "gemini-pro", 
    "models/gemini-1.5-pro",
    "models/gemini-pro",
    "gemini-1.5-pro-latest",
    "gemini-pro-latest"
]

gemini_key = os.getenv('GEMINI_API_KEY')
client = openai.OpenAI(
    api_key=gemini_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

for model_name in model_names:
    try:
        print(f"🔄 Testing model: {model_name}")
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        print(f"✅ SUCCESS: {model_name} works!")
        print(f"Response: {response.choices[0].message.content}")
        break
    except Exception as e:
        print(f"❌ FAILED: {model_name} - {str(e)[:100]}...")
        continue
