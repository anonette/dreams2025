"""
Main execution script for the cross-linguistic dream research system.
"""

import asyncio
import os
import logging
from typing import Dict
import argparse
import json

from src.pipeline.dream_generator import DreamResearchPipeline
from src.visualization.report_generator import DreamReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def main():
    parser = argparse.ArgumentParser(description='Cross-Linguistic Dream Research System')
    parser.add_argument('--api-keys', type=str, help='Path to API keys file')
    parser.add_argument('--dreams-per-config', type=int, default=10, 
                       help='Number of dreams to generate per configuration')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--skip-generation', action='store_true',
                       help='Skip dream generation and load existing results')
    
    args = parser.parse_args()
    
    # Load API keys
    api_keys = {}
    if args.api_keys and os.path.exists(args.api_keys):
        with open(args.api_keys, 'r') as f:
            api_keys = json.load(f)
    else:
        # Try environment variables
        api_keys = {
            'openai': os.getenv('OPENAI_API_KEY'),
            'anthropic': os.getenv('ANTHROPIC_API_KEY')
        }
    
    # Initialize pipeline
    pipeline = DreamResearchPipeline(api_keys)
    report_generator = DreamReportGenerator()
    
    if not args.skip_generation:
        logging.info("Starting dream generation...")
        generated_dreams = await pipeline.generate_dreams(args.dreams_per_config)
        
        logging.info("Analyzing results...")
        analysis_results = pipeline.analyze_results(generated_dreams)
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        pipeline.save_results(generated_dreams, analysis_results, args.output_dir)
        
        logging.info("Generating visualizations...")
        
        # Create visualizations
        theme_chart = report_generator.create_theme_comparison_chart(
            analysis_results['cultural_patterns']
        )
        theme_chart.write_html(f"{args.output_dir}/theme_comparison.html")
        
        cluster_viz = report_generator.create_cluster_visualization(
            analysis_results['clustering']
        )
        cluster_viz.write_html(f"{args.output_dir}/clusters.html")
        
        dashboard = report_generator.create_language_comparison_dashboard(
            analysis_results
        )
        dashboard.write_html(f"{args.output_dir}/dashboard.html")
        
        logging.info(f"Results saved to {args.output_dir}/")
        
    else:
        # Load existing results for analysis
        logging.info("Loading existing results...")
        # Implementation for loading existing results would go here
        
    logging.info("Research pipeline completed!")

if __name__ == "__main__":
    asyncio.run(main())