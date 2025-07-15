#!/usr/bin/env python3
"""
Test Hebrew translation with deep-translator
"""

from deep_translator import GoogleTranslator

# Test Hebrew language codes
test_text = "שלום עולם"  # "Hello world" in Hebrew

print("Testing Hebrew language codes with deep-translator...")

# Test with 'iw' (what Google uses)
try:
    translator_iw = GoogleTranslator(source='iw', target='en')
    result_iw = translator_iw.translate(test_text)
    print(f"✓ 'iw' works: '{test_text}' -> '{result_iw}'")
except Exception as e:
    print(f"✗ 'iw' failed: {e}")

# Test with 'he' (ISO standard)
try:
    translator_he = GoogleTranslator(source='he', target='en')
    result_he = translator_he.translate(test_text)
    print(f"✓ 'he' works: '{test_text}' -> '{result_he}'")
except Exception as e:
    print(f"✗ 'he' failed: {e}")

# Test with 'hebrew' (full name)
try:
    translator_hebrew = GoogleTranslator(source='hebrew', target='en')
    result_hebrew = translator_hebrew.translate(test_text)
    print(f"✓ 'hebrew' works: '{test_text}' -> '{result_hebrew}'")
except Exception as e:
    print(f"✗ 'hebrew' failed: {e}")

# Test auto-detection
try:
    translator_auto = GoogleTranslator(source='auto', target='en')
    result_auto = translator_auto.translate(test_text)
    print(f"✓ 'auto' works: '{test_text}' -> '{result_auto}'")
except Exception as e:
    print(f"✗ 'auto' failed: {e}")

print("\nSummary: Use the working language code in the thematic analysis script.") 