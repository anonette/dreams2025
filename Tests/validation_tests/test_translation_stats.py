#!/usr/bin/env python3
"""
Simple test to check translation data and Basque theme detection
"""

import json
from pathlib import Path
import numpy as np

def check_translation_data():
    """Check what's actually in the translation files"""
    print("🔍 CHECKING TRANSLATION DATA")
    print("=" * 40)
    
    translations_dir = Path("translations")
    if not translations_dir.exists():
        print("❌ No translations directory found!")
        return
    
    # Find session files
    json_files = list(translations_dir.glob("*_translations_*.json"))
    print(f"📁 Found {len(json_files)} translation files")
    
    for file_path in json_files:
        if "session_20250706_093349" in file_path.name:
            lang = file_path.name.split('_translations_')[0]
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                print(f"\n📊 {lang.upper()}:")
                print(f"   Total dreams: {data['total_dreams']}")
                
                # Check sample translations
                if data['dreams']:
                    sample = data['dreams'][0]
                    print(f"   Sample original: {sample['original_text'][:100]}...")
                    if sample.get('translated_text'):
                        print(f"   Sample translation: {sample['translated_text'][:100]}...")
                    else:
                        print(f"   Sample translation: None (English)")
                
            except Exception as e:
                print(f"   ❌ Error reading {lang}: {e}")

def simple_theme_test():
    """Simple theme detection test on Basque"""
    print("\n🎯 SIMPLE BASQUE THEME TEST")
    print("=" * 40)
    
    # Load Basque translations
    basque_file = Path("translations/basque_translations_session_20250706_093349.json")
    
    if not basque_file.exists():
        print("❌ Basque translation file not found!")
        return
    
    with open(basque_file, 'r', encoding='utf-8') as f:
        basque_data = json.load(f)
    
    print(f"📊 Basque dreams loaded: {basque_data['total_dreams']}")
    
    # Simple keyword counting on translated text
    theme_keywords = {
        'transportation': ['car', 'bus', 'train', 'plane', 'bike', 'drive', 'travel', 'road', 'vehicle', 'boat', 'ship'],
        'animals': ['dog', 'cat', 'bird', 'animal', 'pet', 'wild', 'horse', 'sheep', 'cow', 'fish'],
        'nature': ['tree', 'forest', 'mountain', 'sea', 'river', 'flower', 'plant', 'nature', 'outdoor'],
        'water': ['water', 'ocean', 'sea', 'river', 'lake', 'rain', 'swimming', 'waves'],
        'flying': ['fly', 'flying', 'soar', 'float', 'air', 'wings', 'flight']
    }
    
    theme_counts = {}
    total_dreams = len(basque_data['dreams'])
    
    for theme, keywords in theme_keywords.items():
        count = 0
        for dream in basque_data['dreams']:
            translated_text = dream.get('translated_text', '').lower()
            if any(keyword in translated_text for keyword in keywords):
                count += 1
        
        percentage = (count / total_dreams) * 100 if total_dreams > 0 else 0
        theme_counts[theme] = {
            'count': count,
            'percentage': percentage
        }
        
        print(f"   {theme}: {count}/{total_dreams} dreams ({percentage:.1f}%)")
    
    # Show if Basque has meaningful content
    non_zero_themes = sum(1 for data in theme_counts.values() if data['percentage'] > 0)
    print(f"\n✅ Basque themes with content: {non_zero_themes}/{len(theme_keywords)}")
    
    if non_zero_themes > 0:
        print("🎉 SUCCESS: Basque is NOT showing all zeros!")
    else:
        print("❌ PROBLEM: Basque still showing zeros")
    
    return theme_counts

if __name__ == "__main__":
    check_translation_data()
    theme_results = simple_theme_test() 