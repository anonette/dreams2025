#!/usr/bin/env python3
"""
Command-line interface for generating research reports from Dreams project data.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.reporting.research_reporter import ResearchReporter, ResearchMetadata

def main():
    parser = argparse.ArgumentParser(
        description="Generate research reports from Dreams project log data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate report for specific sessions
  python generate_research_report.py --sessions 20250625_155722 20250625_154026
  
  # Generate report with custom metadata
  python generate_research_report.py --sessions 20250625_155722 --title "My Study" --authors "Dr. Smith"
  
  # Generate report without data package
  python generate_research_report.py --sessions 20250625_155722 --no-data-package
  
  # Use config file for metadata
  python generate_research_report.py --sessions 20250625_155722 --config metadata.json
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--sessions', 
        nargs='+', 
        required=True,
        help='Session IDs to include in the report (e.g., 20250625_155722)'
    )
    
    # Optional metadata arguments
    parser.add_argument(
        '--title',
        default="Cross-Linguistic Patterns in AI-Generated Dream Narratives",
        help='Study title'
    )
    
    parser.add_argument(
        '--authors',
        nargs='+',
        default=["Researcher"],
        help='Study authors (space-separated)'
    )
    
    parser.add_argument(
        '--institution',
        default="Research Institution",
        help='Institution name'
    )
    
    parser.add_argument(
        '--keywords',
        nargs='+',
        default=["cross-linguistic", "AI", "dreams", "cultural bias", "LLM"],
        help='Study keywords (space-separated)'
    )
    
    parser.add_argument(
        '--abstract',
        help='Study abstract (optional - will be auto-generated if not provided)'
    )
    
    # Configuration options
    parser.add_argument(
        '--config',
        help='JSON file containing metadata configuration'
    )
    
    parser.add_argument(
        '--logs-dir',
        default="logs",
        help='Directory containing log files (default: logs)'
    )
    
    parser.add_argument(
        '--output-dir',
        default="research_reports",
        help='Output directory for reports (default: research_reports)'
    )
    
    parser.add_argument(
        '--no-data-package',
        action='store_true',
        help='Skip creating data package for sharing'
    )
    
    parser.add_argument(
        '--study-id',
        help='Custom study ID (default: auto-generated)'
    )
    
    args = parser.parse_args()
    
    # Load configuration from file if provided
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            
            # Override command line arguments with config file values
            for key, value in config.items():
                if hasattr(args, key) and value is not None:
                    setattr(args, key, value)
                    
        except Exception as e:
            print(f"Error loading config file: {e}")
            sys.exit(1)
    
    # Detect languages and models from session data
    logs_path = Path(args.logs_dir)
    languages_detected = []
    models_detected = []
    total_dreams = 0
    
    for lang_dir in logs_path.iterdir():
        if lang_dir.is_dir() and not lang_dir.name.startswith('batch'):
            languages_detected.append(lang_dir.name)
            
            # Check for models
            for model_dir in lang_dir.iterdir():
                if model_dir.is_dir() and model_dir.name not in models_detected:
                    models_detected.append(model_dir.name)
                    
                    # Count dreams
                    for session_id in args.sessions:
                        session_path = model_dir / f"session_{session_id}"
                        if session_path.exists():
                            dreams_file = session_path / "dreams.csv"
                            if dreams_file.exists():
                                try:
                                    import pandas as pd
                                    df = pd.read_csv(dreams_file)
                                    total_dreams += len(df)
                                except:
                                    pass
    
    # Create metadata
    metadata = ResearchMetadata(
        study_id=args.study_id or f"DREAM_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        title=args.title,
        authors=args.authors,
        institution=args.institution,
        date_generated=datetime.now().strftime("%Y-%m-%d"),
        languages_analyzed=languages_detected,
        models_used=models_detected or ["gpt-4o"],
        total_dreams=total_dreams,
        analysis_methods=["Descriptive Statistics", "Chi-square Tests", "ANOVA", "Content Analysis"],
        keywords=args.keywords,
        abstract=args.abstract or ""
    )
    
    # Initialize reporter
    reporter = ResearchReporter(args.logs_dir, args.output_dir)
    
    # Generate report
    print(f"🔬 Generating research report...")
    print(f"📊 Study ID: {metadata.study_id}")
    print(f"📝 Title: {metadata.title}")
    print(f"👥 Authors: {', '.join(metadata.authors)}")
    print(f"🌐 Languages: {', '.join(metadata.languages_analyzed)}")
    print(f"🤖 Models: {', '.join(metadata.models_used)}")
    print(f"💭 Total dreams: {metadata.total_dreams}")
    print(f"📁 Sessions: {', '.join(args.sessions)}")
    print()
    
    try:
        result = reporter.generate_research_report(
            session_ids=args.sessions,
            metadata=metadata,
            include_data_package=not args.no_data_package
        )
        
        print("✅ Research report generated successfully!")
        print(f"📄 Report ID: {result['report_id']}")
        print(f"📂 Report directory: {result['report_dir']}")
        if result['sharing_package']:
            print(f"📦 Sharing package: {result['sharing_package']}")
        print()
        
        print("📋 Generated files:")
        for file_path in result['files_generated']:
            if Path(file_path).is_file():
                print(f"  - {file_path}")
        
        print()
        print("📊 Summary statistics:")
        summary = result['summary']
        if 'total_observations' in summary:
            print(f"  - Total API calls: {summary['total_observations']}")
        if 'unique_languages' in summary:
            print(f"  - Languages analyzed: {summary['unique_languages']}")
        if 'overall_success_rate' in summary:
            print(f"  - Overall success rate: {summary['overall_success_rate']:.1%}")
        
        if 'significant_findings' in summary and summary['significant_findings']:
            print("\n🔍 Significant findings:")
            for finding in summary['significant_findings']:
                print(f"  - {finding['test']}: p = {finding['p_value']:.3f}")
        
        print(f"\n🎉 Report generation complete!")
        print(f"📖 View the report: {result['report_dir']}/{result['report_id']}_report.md")
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        sys.exit(1)

def create_sample_config():
    """Create a sample configuration file."""
    config = {
        "title": "Cross-Linguistic Patterns in AI-Generated Dream Narratives",
        "authors": ["Dr. Jane Smith", "Dr. John Doe", "Dr. Maria Garcia"],
        "institution": "University of Computational Linguistics",
        "keywords": ["cross-linguistic", "AI", "dreams", "cultural bias", "LLM", "multilingual"],
        "abstract": "This study examines how large language models generate dream narratives across different languages and cultural contexts. We analyze patterns in success rates, content characteristics, and temporal factors to understand potential cultural biases in AI-generated creative content."
    }
    
    with open("sample_metadata.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Sample configuration saved to sample_metadata.json")
    print("Edit this file and use it with: --config sample_metadata.json")

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--create-sample-config":
        create_sample_config()
    else:
        main() 