#!/usr/bin/env python3
"""
Check if all required dependencies are installed and working.
"""

import sys
import importlib

def check_dependency(module_name, package_name=None):
    """Check if a dependency is available."""
    try:
        importlib.import_module(module_name)
        print(f"✅ {package_name or module_name}")
        return True
    except ImportError:
        print(f"❌ {package_name or module_name} - NOT INSTALLED")
        return False

def main():
    """Check all dependencies."""
    print("📦 Dependency Check")
    print("=" * 30)
    
    # Core dependencies
    print("\n🔧 Core Dependencies:")
    core_deps = [
        ('openai', 'openai'),
        ('anthropic', 'anthropic'),
        ('httpx', 'httpx'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('sklearn', 'scikit-learn'),
        ('spacy', 'spacy'),
        ('googletrans', 'googletrans'),
    ]
    
    core_missing = []
    for module, package in core_deps:
        if not check_dependency(module, package):
            core_missing.append(package)
    
    # Visualization dependencies
    print("\n📊 Visualization Dependencies:")
    viz_deps = [
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('plotly', 'plotly'),
    ]
    
    viz_missing = []
    for module, package in viz_deps:
        if not check_dependency(module, package):
            viz_missing.append(package)
    
    # Async and utilities
    print("\n⚡ Async & Utilities:")
    util_deps = [
        ('aiohttp', 'aiohttp'),
        ('dotenv', 'python-dotenv'),
        ('tqdm', 'tqdm'),
    ]
    
    util_missing = []
    for module, package in util_deps:
        if not check_dependency(module, package):
            util_missing.append(package)
    
    # Built-in modules (should always be available)
    print("\n🐍 Built-in Modules:")
    builtin_deps = [
        ('asyncio', 'asyncio (built-in)'),
        ('json', 'json (built-in)'),
        ('os', 'os (built-in)'),
        ('sys', 'sys (built-in)'),
        ('logging', 'logging (built-in)'),
        ('datetime', 'datetime (built-in)'),
        ('typing', 'typing (built-in)'),
        ('collections', 'collections (built-in)'),
        ('re', 're (built-in)'),
        ('dataclasses', 'dataclasses (built-in)'),
    ]
    
    for module, name in builtin_deps:
        check_dependency(module, name)
    
    # Summary
    print(f"\n" + "=" * 30)
    all_missing = core_missing + viz_missing + util_missing
    
    if all_missing:
        print(f"❌ Missing dependencies: {', '.join(all_missing)}")
        print(f"\n📥 Install missing packages:")
        print(f"pip install {' '.join(all_missing)}")
        
        print(f"\n📥 Or install all requirements:")
        print(f"pip install -r requirements.txt")
    else:
        print(f"✅ All dependencies are installed!")
        
        # Check spaCy model
        print(f"\n🧠 Checking spaCy model...")
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            print(f"✅ spaCy English model loaded successfully")
        except OSError:
            print(f"❌ spaCy English model not found!")
            print(f"📥 Install with: python -m spacy download en_core_web_sm")
        except Exception as e:
            print(f"⚠️  spaCy model error: {e}")
    
    print(f"\n🚀 Ready to run dream research experiments!")

if __name__ == "__main__":
    main() 