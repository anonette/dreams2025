#!/usr/bin/env python3
"""
Test Translation Sample - Quick test of the translation system
"""

from translation_manager import TranslationManager
import pandas as pd
from pathlib import Path

def test_translation_sample():
    print("🧪 Testing Translation Sample")
    print("=" * 40)
    
    # Initialize manager
    manager = TranslationManager()
    
    # Find latest session
    session_id = manager.find_latest_session()
    print(f"📊 Using session: {session_id}")
    
    # Test loading dreams for one language
    print(f"\n🔍 Testing Basque dream loading...")
    basque_dreams = manager.load_dreams_for_language('basque', session_id)
    print(f"✅ Loaded {len(basque_dreams)} Basque dreams")
    
    if basque_dreams:
        # Show sample dream
        sample = basque_dreams[0]
        print(f"\n📝 Sample dream:")
        print(f"ID: {sample['dream_id']}")
        print(f"Original: {sample['original_text'][:100]}...")
        
        # Test translation of just this one dream
        print(f"\n🔄 Testing translation...")
        try:
            translated = manager.translate_text(sample['original_text'], 'eu')
            print(f"✅ Translation successful!")
            print(f"Translated: {translated[:100]}...")
            print(f"Original length: {len(sample['original_text'])} chars")
            print(f"Translated length: {len(translated)} chars")
            
            # Test if keywords would be found
            english_keywords = ['car', 'water', 'home', 'fly', 'nature', 'light']
            found_keywords = []
            for keyword in english_keywords:
                if keyword.lower() in translated.lower():
                    found_keywords.append(keyword)
            
            print(f"\n🔍 Keywords found in translation: {found_keywords}")
            
        except Exception as e:
            print(f"❌ Translation failed: {e}")
    
    # Test Hebrew (special case)
    print(f"\n🔍 Testing Hebrew dream loading...")
    hebrew_dreams = manager.load_dreams_for_language('hebrew', session_id)
    print(f"✅ Loaded {len(hebrew_dreams)} Hebrew dreams")
    
    if hebrew_dreams:
        sample = hebrew_dreams[0]
        print(f"\n📝 Sample Hebrew dream:")
        print(f"Original: {sample['original_text'][:100]}...")
        
        # Test Hebrew translation (using auto-detection)
        try:
            translated = manager.translate_text(sample['original_text'], 'auto')
            print(f"✅ Hebrew translation successful!")
            print(f"Translated: {translated[:100]}...")
        except Exception as e:
            print(f"❌ Hebrew translation failed: {e}")
    
    print(f"\n🎯 Sample test complete!")

if __name__ == "__main__":
    test_translation_sample() 