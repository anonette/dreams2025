#!/usr/bin/env python3
"""
Test script to debug translation loading
"""

from dream_thematic_analysis import DreamThematicAnalyzer

def test_translations():
    print("🧪 Testing translation loading...")
    
    analyzer = DreamThematicAnalyzer()
    
    # Just load the dreams (which should use existing translations)
    analyzer.load_dreams()
    
    # Print what was loaded
    print("\n📊 Results:")
    for lang, dreams in analyzer.dreams_by_language.items():
        print(f"  {lang}: {len(dreams)} dreams")
        if dreams:
            sample = dreams[0]
            print(f"    Sample text: {sample['text'][:100]}...")
            print(f"    Has translation: {'translated_text' in sample and sample['translated_text'] is not None}")

if __name__ == "__main__":
    test_translations() 