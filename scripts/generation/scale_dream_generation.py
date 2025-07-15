#!/usr/bin/env python3
"""
Scaled Dream Generation System
Large-scale no-system-prompt dream generation with multiple scaling options.

Based on the successful OptimizedBatchV2 configuration.
"""

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

# Import the successful configuration
from optimized_batch_v2 import OptimizedBatchV2, OptimizedV2Config

@dataclass 
class ScalingConfig:
    """Configuration options for scaled dream generation"""
    
    # Scale options
    SMALL_SCALE = {
        "name": "Small Scale Research",
        "total_dreams": 1000,  # 200 per language
        "dreams_per_language": 200,
        "estimated_time_minutes": 100,
        "description": "Double your current dataset for stronger statistical analysis"
    }
    
    MEDIUM_SCALE = {
        "name": "Medium Scale Study", 
        "total_dreams": 2500,  # 500 per language
        "dreams_per_language": 500,
        "estimated_time_minutes": 250,
        "description": "Research-grade dataset suitable for publication"
    }
    
    LARGE_SCALE = {
        "name": "Large Scale Analysis",
        "total_dreams": 5000,  # 1000 per language  
        "dreams_per_language": 1000,
        "estimated_time_minutes": 500,
        "description": "Comprehensive dataset for robust cross-linguistic analysis"
    }
    
    MASSIVE_SCALE = {
        "name": "Massive Research Dataset",
        "total_dreams": 10000,  # 2000 per language
        "dreams_per_language": 2000, 
        "estimated_time_minutes": 1000,
        "description": "Publication-ready massive dataset for comprehensive research"
    }

class ScaledDreamGenerator:
    """Large-scale dream generation system based on successful OptimizedBatchV2"""
    
    def __init__(self, api_keys: Dict[str, str], scale_config: Dict):
        self.scale_config = scale_config
        self.api_keys = api_keys
        
        # Create modified config for scaling
        self.base_config = OptimizedV2Config()
        self.base_config.dreams_per_language = scale_config["dreams_per_language"]
        self.base_config.total_target_dreams = scale_config["total_dreams"]
        
        # Session management
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"SCALED_{scale_config['dreams_per_language']}_{timestamp}"
        
        # Progress tracking
        self.batches_completed = 0
        self.total_batches = self.calculate_total_batches()
        
        print(f"🚀 SCALED DREAM GENERATION SYSTEM")
        print(f"📊 Scale: {scale_config['name']}")
        print(f"🎯 Target: {scale_config['total_dreams']} dreams ({scale_config['dreams_per_language']} per language)")
        print(f"⏱️  Estimated time: {scale_config['estimated_time_minutes']} minutes")
        print(f"📝 Session: {self.session_id}")
        print(f"🔄 Will run in {self.total_batches} batches of {self.base_config.batch_size} dreams each")
    
    def calculate_total_batches(self) -> int:
        """Calculate total number of batches needed"""
        # Each batch generates batch_size dreams per language
        dreams_per_batch = self.base_config.batch_size * 5  # 5 languages
        return (self.base_config.total_target_dreams + dreams_per_batch - 1) // dreams_per_batch
    
    async def run_scaled_generation(self):
        """Run the scaled generation process"""
        print(f"\n🎬 Starting scaled generation...")
        
        total_start_time = time.time()
        
        try:
            # Create the batch generator with scaled config
            batch_generator = OptimizedBatchV2(self.api_keys)
            
            # Override the configuration
            batch_generator.config.dreams_per_language = self.scale_config["dreams_per_language"] 
            batch_generator.config.total_target_dreams = self.scale_config["total_dreams"]
            batch_generator.session_id = self.session_id
            
            # Run the generation
            results = await batch_generator.generate_all_languages()
            
            total_duration = time.time() - total_start_time
            
            # Generate completion report
            await self.generate_scaling_report(results, total_duration)
            
            print(f"\n🎉 SCALED GENERATION COMPLETE!")
            print(f"✅ Generated {results['total_successful_dreams']}/{self.scale_config['total_dreams']} dreams")
            print(f"⏱️  Total time: {total_duration/60:.1f} minutes")
            print(f"📊 Success rate: {results['global_success_rate']:.1f}%")
            
            return results
            
        except KeyboardInterrupt:
            print(f"\n🛑 Generation interrupted by user")
            print(f"💾 Progress has been saved and can be resumed")
            return None
        except Exception as e:
            print(f"❌ Error during scaled generation: {e}")
            return None
    
    async def generate_scaling_report(self, results: Dict, duration: float):
        """Generate comprehensive scaling report"""
        report_file = f"scaling_report_{self.session_id}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# Scaled Dream Generation Report\n\n")
            f.write(f"**Session**: {self.session_id}\n")
            f.write(f"**Scale**: {self.scale_config['name']}\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"## Summary\n\n")
            f.write(f"- **Target Dreams**: {self.scale_config['total_dreams']:,}\n")
            f.write(f"- **Successful Dreams**: {results['total_successful_dreams']:,}\n")
            f.write(f"- **Success Rate**: {results['global_success_rate']:.1f}%\n")
            f.write(f"- **Total Duration**: {duration/60:.1f} minutes\n")
            f.write(f"- **Average per Dream**: {duration/results['total_successful_dreams']:.2f} seconds\n\n")
            
            f.write(f"## Language Breakdown\n\n")
            f.write(f"| Language | Dreams | Success Rate | Avg Words | Avg Chars |\n")
            f.write(f"|----------|--------|--------------|-----------|----------|\n")
            
            for lang, lang_results in results['language_results'].items():
                f.write(f"| {lang.title()} | {lang_results['successful_dreams']:,} | ")
                f.write(f"{lang_results['success_rate']:.1f}% | ")
                f.write(f"{lang_results['avg_words']:.0f} | ")
                f.write(f"{lang_results['avg_chars']:.0f} |\n")
            
            f.write(f"\n## Configuration\n\n")
            f.write(f"- **No System Prompt**: ✅\n")
            f.write(f"- **Temperature**: {self.base_config.temperature}\n")
            f.write(f"- **Top-p**: {self.base_config.top_p}\n")
            f.write(f"- **Batch Size**: {self.base_config.batch_size}\n")
            
            f.write(f"\n## Performance Analysis\n\n")
            dreams_per_minute = results['total_successful_dreams'] / (duration / 60)
            f.write(f"- **Dreams per minute**: {dreams_per_minute:.1f}\n")
            f.write(f"- **Estimated cost**: Based on GPT-4o pricing\n")
            f.write(f"- **Data size**: ~{results['total_successful_dreams'] * 800:,} characters\n")
        
        print(f"📄 Scaling report saved: {report_file}")

def display_scaling_menu():
    """Display interactive scaling options menu"""
    print(f"\n🎯 DREAM GENERATION SCALING OPTIONS")
    print(f"=" * 60)
    
    options = [
        ScalingConfig.SMALL_SCALE,
        ScalingConfig.MEDIUM_SCALE, 
        ScalingConfig.LARGE_SCALE,
        ScalingConfig.MASSIVE_SCALE
    ]
    
    for i, option in enumerate(options, 1):
        print(f"{i}. {option['name']}")
        print(f"   • Dreams: {option['total_dreams']:,} ({option['dreams_per_language']} per language)")
        print(f"   • Time: ~{option['estimated_time_minutes']} minutes") 
        print(f"   • Use: {option['description']}")
        print()
    
    print(f"📝 All options use your proven no-system-prompt configuration")
    print(f"✅ Based on your successful 100% success rate baseline")
    print(f"🔄 Full resumption support if interrupted")
    
    return options

async def main():
    """Main scaling interface"""
    print(f"🌙 DREAM GENERATION SCALING SYSTEM")
    print(f"Based on your successful OptimizedBatchV2 configuration")
    
    # Load API keys from environment variables (same as working scripts)
    import os
    api_keys = {
        'openai': os.getenv('OPENAI_API_KEY'),
        'anthropic': os.getenv('ANTHROPIC_API_KEY'), 
        'google': os.getenv('GOOGLE_API_KEY')
    }
    
    # Check if we have at least OpenAI key
    if not api_keys['openai']:
        print(f"❌ OpenAI API key not found!")
        print(f"Please set your environment variable:")
        print(f"  export OPENAI_API_KEY='your-api-key-here'")
        print(f"Or in PowerShell:")
        print(f"  $env:OPENAI_API_KEY='your-api-key-here'")
        return
    
    print(f"✅ Found OpenAI API key: {'*' * (len(api_keys['openai']) - 4)}{api_keys['openai'][-4:]}")
    
    # Display options
    options = display_scaling_menu()
    
    print(f"\n🚀 Choose your scaling level:")
    choice = input(f"Enter 1-4 (or 'q' to quit): ").strip()
    
    if choice.lower() == 'q':
        print(f"👋 Goodbye!")
        return
    
    try:
        choice_idx = int(choice) - 1
        if 0 <= choice_idx < len(options):
            selected_config = options[choice_idx]
            
            print(f"\n✅ Selected: {selected_config['name']}")
            print(f"🎯 Will generate {selected_config['total_dreams']:,} dreams")
            
            confirm = input(f"Proceed? (y/n): ").strip().lower()
            if confirm == 'y':
                # Create and run scaled generator
                generator = ScaledDreamGenerator(api_keys, selected_config)
                await generator.run_scaled_generation()
            else:
                print(f"❌ Cancelled")
        else:
            print(f"❌ Invalid choice")
    except ValueError:
        print(f"❌ Invalid input")

if __name__ == "__main__":
    asyncio.run(main()) 