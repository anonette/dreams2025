#!/usr/bin/env python3
"""
Quick Launcher for Scaled Dream Generation
Simple interface to start large-scale no-system-prompt dream generation
"""

import asyncio
import subprocess
import sys

def main():
    """Launch the scaled dream generation system"""
    print("🚀 LAUNCHING SCALED DREAM GENERATION SYSTEM")
    print("=" * 55)
    print()
    print("This will launch the interactive scaling menu where you can choose:")
    print("• Small Scale: 1,000 dreams (200 per language)")  
    print("• Medium Scale: 2,500 dreams (500 per language)")
    print("• Large Scale: 5,000 dreams (1,000 per language)")
    print("• Massive Scale: 10,000 dreams (2,000 per language)")
    print()
    print("All options use your proven no-system-prompt configuration")
    print("with 100% success rate and full resumption support.")
    print()
    
    try:
        # Run the scaling system
        result = subprocess.run([sys.executable, "scale_dream_generation.py"], 
                              check=True, text=True)
        print("✅ Scaling system completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running scaling system: {e}")
    except KeyboardInterrupt:
        print(f"\n🛑 Cancelled by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main() 