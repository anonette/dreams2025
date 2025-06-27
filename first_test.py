#!/usr/bin/env python3
"""
First test experiment: GPT-4o with temperature 0.9
Generate dreams across all languages to test the system.
"""

import asyncio
import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path
sys.path.append('.')

from src.pipeline.dream_generator import DreamResearchPipeline
from src.visualization.report_generator import DreamReportGenerator

async def first_test():
    """Run the first test with GPT-4o and temperature 0.9."""
    print("🌙 Dream Research - First Test Experiment")
    print("=" * 50)
    print("Model: GPT-4o")
    print("Temperature: 0.9")
    print("Analysis: LLM-based theme identification")
    print("Languages: English, Basque, Serbian, Hebrew, Slovenian")
    print("=" * 50)
    
    # Get API keys
    api_keys = {
        'openai': os.getenv('OPENAI_API_KEY'),
        'anthropic': os.getenv('ANTHROPIC_API_KEY'),
        'openrouter': os.getenv('OPENROUTER_API_KEY')
    }
    
    # Check if OpenAI key is available
    if not api_keys['openai']:
        print("❌ OPENAI_API_KEY not found in environment variables!")
        print("Please set up your .env file with your OpenAI API key.")
        return
    
    print("✅ OpenAI API key found")
    
    # Initialize pipeline with LLM-based analyzer
    pipeline = DreamResearchPipeline(api_keys, analysis_model='gpt-4o')
    report_generator = DreamReportGenerator()
    
    # Generate dreams (small number for first test)
    dreams_per_language = 3  # Start with 3 dreams per language
    print(f"\n🚀 Generating {dreams_per_language} dreams per language...")
    
    try:
        generated_dreams = await pipeline.generate_dreams(dreams_per_language)
        
        print("✅ Dream generation completed!")
        
        # Analyze results using LLM
        print("\n🧠 Analyzing results with LLM...")
        analysis_results = await pipeline.analyze_results(generated_dreams)
        
        # Create output directory
        output_dir = "results/first_test"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pipeline.save_results(generated_dreams, analysis_results, output_dir)
        
        # Create visualizations
        print("📈 Creating visualizations...")
        
        # Theme comparison chart
        theme_chart = report_generator.create_theme_comparison_chart(
            analysis_results['cultural_patterns']
        )
        theme_chart.write_html(f"{output_dir}/theme_comparison.html")
        
        # Language comparison dashboard
        dashboard = report_generator.create_language_comparison_dashboard(
            analysis_results
        )
        dashboard.write_html(f"{output_dir}/dashboard.html")
        
        # Print summary
        print(f"\n🎉 First test completed successfully!")
        print(f"📁 Results saved to: {output_dir}/")
        
        # Show some sample dreams
        print(f"\n📝 Sample dreams generated:")
        print("-" * 30)
        
        for language, config in generated_dreams.items():
            print(f"\n{language.upper()}:")
            for i, dream in enumerate(config['gpt-4o'][0.9][:2]):  # Show first 2 dreams
                print(f"  {i+1}. {dream[:100]}...")
        
        # Show analysis summary
        stats = analysis_results['summary_stats']
        print(f"\n📊 Analysis Summary:")
        print(f"  Total dreams: {stats['total_dreams']}")
        print(f"  Languages: {stats['languages']}")
        print(f"  Models: {stats['models']}")
        print(f"  Temperatures: {stats['temperatures']}")
        
        # Show LLM analysis results
        if 'cross_linguistic_analysis' in analysis_results:
            cross_analysis = analysis_results['cross_linguistic_analysis']
            print(f"\n🧠 LLM Analysis Results:")
            if 'common_themes' in cross_analysis:
                print(f"  Common themes: {cross_analysis['common_themes']}")
            if 'cultural_differences' in cross_analysis:
                print(f"  Cultural insights: {cross_analysis['cultural_differences'][:100]}...")
        
        print(f"\n🔍 Check the HTML files in {output_dir}/ for detailed visualizations!")
        print(f"📄 Check the JSON files for complete LLM analysis results!")
        
    except Exception as e:
        print(f"❌ Error during experiment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(first_test()) 