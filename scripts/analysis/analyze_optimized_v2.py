#!/usr/bin/env python3
"""
Analyze Optimized Dreams v2.0
Simplified analysis script for optimized dream data without heavy statistical dependencies.
"""

import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
from collections import Counter
import sys

class OptimizedDreamAnalyzerV2:
    """Simplified analyzer for optimized dream data"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.logs_dir = Path('logs_optimized_v2')  # Look in the v2 logs directory
        self.results_dir = Path(f'analysis_output/session_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.languages = ['english', 'basque', 'hebrew', 'serbian', 'slovenian']
        self.data = {}
        self.analysis_results = {}
        
        print(f"🔍 Optimized Dream Analyzer v2.0 initialized")
        print(f"📊 Session: {session_id}")
        print(f"📁 Looking for data in: {self.logs_dir}")
        print(f"📁 Results will be saved to: {self.results_dir}")
    
    def load_optimized_data(self):
        """Load optimized dream data from session logs"""
        print(f"\n📥 Loading optimized data from session {self.session_id}...")
        
        # First check if global files exist
        all_dreams_file = self.logs_dir / f'all_dreams_{self.session_id}.csv'
        all_api_calls_file = self.logs_dir / f'all_api_calls_{self.session_id}.csv'
        session_summary_file = self.logs_dir / f'session_summary_{self.session_id}.json'
        
        if all_dreams_file.exists():
            print(f"  ✅ Found global dreams file: {all_dreams_file}")
            all_dreams_df = pd.read_csv(all_dreams_file)
            
            # Load session summary if available
            if session_summary_file.exists():
                with open(session_summary_file, 'r', encoding='utf-8') as f:
                    session_summary = json.load(f)
                print(f"  ✅ Found session summary: {session_summary_file}")
            else:
                session_summary = {}
            
            # Split dreams by language
            for language in self.languages:
                lang_dreams = all_dreams_df[all_dreams_df['language'] == language]
                successful_dreams = lang_dreams[lang_dreams['status'] == 'success']
                
                if len(successful_dreams) > 0:
                    self.data[language] = {
                        'dreams': successful_dreams,
                        'session_summary': session_summary
                    }
                    print(f"    ✅ {language}: {len(successful_dreams)} successful dreams")
                else:
                    print(f"    ⚠️  {language}: No successful dreams found")
        else:
            print(f"  ❌ Global dreams file not found: {all_dreams_file}")
            print(f"  🔍 Checking individual language directories...")
            
            # Fallback: check individual language directories
            for language in self.languages:
                lang_dir = self.logs_dir / language / 'gpt-4o' / f'session_{self.session_id}'
                
                if not lang_dir.exists():
                    print(f"    ⚠️  No directory found for {language}")
                    continue
                
                dreams_file = lang_dir / 'dreams.csv'
                session_file = lang_dir / 'session_data.json'
                
                if dreams_file.exists():
                    dreams_df = pd.read_csv(dreams_file)
                    successful_dreams = dreams_df[dreams_df['status'] == 'success']
                    
                    session_data = {}
                    if session_file.exists():
                        with open(session_file, 'r', encoding='utf-8') as f:
                            session_data = json.load(f)
                    
                    self.data[language] = {
                        'dreams': successful_dreams,
                        'session_data': session_data
                    }
                    
                    print(f"    ✅ {language}: {len(successful_dreams)} successful dreams")
                else:
                    print(f"    ❌ {language}: No dreams.csv found")
        
        print(f"📊 Successfully loaded data for {len(self.data)} languages")
        
        if not self.data:
            print(f"❌ No data found for session {self.session_id}")
            print(f"💡 Make sure you've run the optimized batch generator first")
            return False
        
        return True
    
    def analyze_dream_quality(self):
        """Analyze dream quality metrics"""
        print(f"\n🔬 Analyzing dream quality...")
        
        quality_analysis = {
            'by_language': {},
            'overall_stats': {},
            'configuration_impact': {}
        }
        
        all_dreams = []
        
        for language, data in self.data.items():
            dreams = data['dreams']
            
            if len(dreams) == 0:
                continue
            
            # Basic text metrics
            dreams_text = dreams['dream'].astype(str)
            char_counts = dreams_text.str.len()
            word_counts = dreams_text.str.split().str.len()
            
            # Vocabulary analysis
            all_text = ' '.join(dreams_text).lower()
            all_words = all_text.split()
            unique_words = set(all_words)
            
            # Quality metrics for this language
            lang_stats = {
                'dream_count': len(dreams),
                'avg_characters': char_counts.mean(),
                'avg_words': word_counts.mean(),
                'median_characters': char_counts.median(),
                'median_words': word_counts.median(),
                'std_characters': char_counts.std(),
                'std_words': word_counts.std(),
                'min_characters': char_counts.min(),
                'max_characters': char_counts.max(),
                'min_words': word_counts.min(),
                'max_words': word_counts.max(),
                'total_words': len(all_words),
                'unique_words': len(unique_words),
                'vocabulary_ratio': len(unique_words) / len(all_words) if all_words else 0,
                'avg_unique_words_per_dream': len(unique_words) / len(dreams) if dreams is not None and len(dreams) > 0 else 0
            }
            
            quality_analysis['by_language'][language] = lang_stats
            
            # Add to overall collection
            for _, dream_row in dreams.iterrows():
                all_dreams.append({
                    'language': language,
                    'characters': len(str(dream_row['dream'])),
                    'words': len(str(dream_row['dream']).split()),
                    'content': str(dream_row['dream'])
                })
            
            print(f"  📊 {language}: {lang_stats['avg_words']:.0f} avg words, "
                  f"{lang_stats['unique_words']} unique words, "
                  f"{lang_stats['vocabulary_ratio']:.3f} vocab ratio")
        
        # Overall statistics
        if all_dreams:
            all_chars = [d['characters'] for d in all_dreams]
            all_words = [d['words'] for d in all_dreams]
            
            quality_analysis['overall_stats'] = {
                'total_dreams': len(all_dreams),
                'avg_characters': np.mean(all_chars),
                'avg_words': np.mean(all_words),
                'median_characters': np.median(all_chars),
                'median_words': np.median(all_words),
                'total_languages': len(self.data)
            }
        
        self.analysis_results['quality'] = quality_analysis
        return quality_analysis
    
    def analyze_content_patterns(self):
        """Analyze content patterns in dreams"""
        print(f"\n🎨 Analyzing content patterns...")
        
        # Simple pattern analysis without heavy dependencies
        patterns = {
            'common_themes': {
                'flying': ['fly', 'flying', 'soar', 'float', 'air'],
                'water': ['water', 'ocean', 'sea', 'river', 'swim', 'waves'],
                'family': ['mother', 'father', 'family', 'mom', 'dad', 'parent'],
                'animals': ['dog', 'cat', 'bird', 'animal', 'pet'],
                'house': ['house', 'home', 'room', 'door', 'window'],
                'fear': ['afraid', 'fear', 'scared', 'terror', 'frightened'],
                'running': ['run', 'running', 'chase', 'escape', 'flee'],
                'death': ['death', 'dead', 'die', 'dying', 'funeral']
            }
        }
        
        content_analysis = {
            'by_language': {},
            'cross_language_comparison': {}
        }
        
        for language, data in self.data.items():
            dreams = data['dreams']
            
            if len(dreams) == 0:
                continue
            
            # Combine all dreams for this language
            all_text = ' '.join(dreams['dream'].astype(str)).lower()
            
            lang_patterns = {}
            
            # Analyze each theme
            for theme, keywords in patterns['common_themes'].items():
                theme_count = 0
                dreams_with_theme = 0
                
                for _, dream_row in dreams.iterrows():
                    dream_text = str(dream_row['dream']).lower()
                    found_in_dream = False
                    
                    for keyword in keywords:
                        keyword_count = dream_text.count(keyword)
                        theme_count += keyword_count
                        if keyword_count > 0:
                            found_in_dream = True
                    
                    if found_in_dream:
                        dreams_with_theme += 1
                
                lang_patterns[theme] = {
                    'total_mentions': theme_count,
                    'dreams_with_theme': dreams_with_theme,
                    'percentage_of_dreams': (dreams_with_theme / len(dreams)) * 100 if len(dreams) > 0 else 0,
                    'mentions_per_dream': theme_count / len(dreams) if len(dreams) > 0 else 0
                }
            
            content_analysis['by_language'][language] = lang_patterns
            
            # Print summary for this language
            top_themes = sorted(lang_patterns.items(), 
                              key=lambda x: x[1]['percentage_of_dreams'], 
                              reverse=True)[:3]
            
            print(f"  📊 {language} top themes: " + 
                  ", ".join([f"{theme}({data['percentage_of_dreams']:.1f}%)" 
                           for theme, data in top_themes]))
        
        # Cross-language comparison
        all_themes = list(patterns['common_themes'].keys())
        comparison = {}
        
        for theme in all_themes:
            comparison[theme] = {}
            percentages = []
            
            for language in self.data.keys():
                if language in content_analysis['by_language']:
                    percentage = content_analysis['by_language'][language][theme]['percentage_of_dreams']
                    comparison[theme][language] = percentage
                    percentages.append(percentage)
                else:
                    comparison[theme][language] = 0
                    percentages.append(0)
            
            if percentages:
                comparison[theme]['average'] = np.mean(percentages)
                comparison[theme]['max'] = np.max(percentages)
                comparison[theme]['min'] = np.min(percentages)
                comparison[theme]['range'] = np.max(percentages) - np.min(percentages)
        
        content_analysis['cross_language_comparison'] = comparison
        
        self.analysis_results['content'] = content_analysis
        return content_analysis
    
    def generate_simplified_report(self):
        """Generate a simplified analysis report"""
        print(f"\n📝 Generating analysis report...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.results_dir / f'optimized_analysis_report_{timestamp}.md'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# Optimized Dreams Analysis Report v2.0\n\n")
            f.write(f"**Session ID**: {self.session_id}\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Analyzer Version**: v2.0 (Simplified)\n\n")
            
            # Executive Summary
            if 'quality' in self.analysis_results:
                quality = self.analysis_results['quality']
                f.write("## Executive Summary\n\n")
                f.write(f"- **Total Dreams Analyzed**: {quality['overall_stats']['total_dreams']}\n")
                f.write(f"- **Languages**: {quality['overall_stats']['total_languages']}\n")
                f.write(f"- **Average Dream Length**: {quality['overall_stats']['avg_words']:.0f} words\n")
                f.write(f"- **Average Characters**: {quality['overall_stats']['avg_characters']:.0f} characters\n\n")
            
            # Quality Analysis
            if 'quality' in self.analysis_results:
                quality = self.analysis_results['quality']
                f.write("## Dream Quality Analysis\n\n")
                f.write("| Language | Dreams | Avg Words | Avg Chars | Unique Words | Vocab Ratio |\n")
                f.write("|----------|--------|-----------|-----------|--------------|-------------|\n")
                
                for language, stats in quality['by_language'].items():
                    f.write(f"| {language.capitalize()} | "
                           f"{stats['dream_count']} | "
                           f"{stats['avg_words']:.0f} | "
                           f"{stats['avg_characters']:.0f} | "
                           f"{stats['unique_words']} | "
                           f"{stats['vocabulary_ratio']:.3f} |\n")
                
                f.write("\n### Key Quality Metrics\n\n")
                for language, stats in quality['by_language'].items():
                    f.write(f"**{language.capitalize()}**:\n")
                    f.write(f"- Word count range: {stats['min_words']} - {stats['max_words']} words\n")
                    f.write(f"- Character range: {stats['min_characters']} - {stats['max_characters']} characters\n")
                    f.write(f"- Vocabulary diversity: {stats['avg_unique_words_per_dream']:.1f} unique words per dream\n\n")
            
            # Content Analysis
            if 'content' in self.analysis_results:
                content = self.analysis_results['content']
                f.write("## Content Pattern Analysis\n\n")
                
                f.write("### Theme Prevalence by Language\n\n")
                f.write("| Theme | " + " | ".join([lang.capitalize() for lang in self.data.keys()]) + " | Average |\n")
                f.write("|-------|" + "|".join(["-------" for _ in self.data.keys()]) + "|----------|\n")
                
                if 'cross_language_comparison' in content:
                    comparison = content['cross_language_comparison']
                    
                    for theme, theme_data in comparison.items():
                        f.write(f"| {theme.capitalize()} | ")
                        for language in self.data.keys():
                            percentage = theme_data.get(language, 0)
                            f.write(f"{percentage:.1f}% | ")
                        f.write(f"{theme_data.get('average', 0):.1f}% |\n")
                
                f.write("\n### Most Variable Themes Across Languages\n\n")
                if 'cross_language_comparison' in content:
                    # Sort by range (variation)
                    sorted_themes = sorted(content['cross_language_comparison'].items(), 
                                         key=lambda x: x[1].get('range', 0), 
                                         reverse=True)
                    
                    for theme, data in sorted_themes[:5]:
                        f.write(f"- **{theme.capitalize()}**: {data.get('range', 0):.1f}% variation "
                               f"(from {data.get('min', 0):.1f}% to {data.get('max', 0):.1f}%)\n")
                
                f.write("\n### Most Common Themes Overall\n\n")
                if 'cross_language_comparison' in content:
                    # Sort by average
                    sorted_themes = sorted(content['cross_language_comparison'].items(), 
                                         key=lambda x: x[1].get('average', 0), 
                                         reverse=True)
                    
                    for theme, data in sorted_themes[:5]:
                        f.write(f"- **{theme.capitalize()}**: {data.get('average', 0):.1f}% average across languages\n")
            
            # Data Files
            f.write("\n## Data Files\n\n")
            f.write("### Source Data\n")
            f.write(f"- Global dreams: `{self.logs_dir}/all_dreams_{self.session_id}.csv`\n")
            f.write(f"- Global API calls: `{self.logs_dir}/all_api_calls_{self.session_id}.csv`\n")
            f.write(f"- Session summary: `{self.logs_dir}/session_summary_{self.session_id}.json`\n\n")
            
            f.write("### Individual Language Data\n")
            for language in self.languages:
                if language in self.data:
                    f.write(f"- **{language.capitalize()}**: `{self.logs_dir}/{language}/gpt-4o/session_{self.session_id}/`\n")
            
            f.write("\n### Analysis Results\n")
            f.write(f"- This report: `{report_file}`\n")
            f.write(f"- Quality analysis: `{self.results_dir}/quality_analysis.json`\n")
            f.write(f"- Content analysis: `{self.results_dir}/content_analysis.json`\n")
            
            f.write("\n## Usage Instructions\n\n")
            f.write("### Loading the Data\n")
            f.write("```python\n")
            f.write("import pandas as pd\n")
            f.write("import json\n\n")
            f.write(f"# Load all dreams\n")
            f.write(f"dreams_df = pd.read_csv('{self.logs_dir}/all_dreams_{self.session_id}.csv')\n")
            f.write("successful_dreams = dreams_df[dreams_df['status'] == 'success']\n\n")
            f.write("# Load session summary\n")
            f.write(f"with open('{self.logs_dir}/session_summary_{self.session_id}.json') as f:\n")
            f.write("    summary = json.load(f)\n")
            f.write("```\n\n")
            
            f.write("### Running Further Analysis\n")
            f.write("```bash\n")
            f.write("# Run this analysis again\n")
            f.write(f"python analyze_optimized_v2.py {self.session_id}\n\n")
            f.write("# Run thematic analysis (if dependencies are available)\n")
            f.write("python dream_thematic_analysis.py\n")
            f.write("```\n\n")
        
        print(f"  📝 Report saved to: {report_file}")
        
        # Save JSON data
        if self.analysis_results:
            if 'quality' in self.analysis_results:
                quality_file = self.results_dir / 'quality_analysis.json'
                with open(quality_file, 'w', encoding='utf-8') as f:
                    json.dump(self.analysis_results['quality'], f, ensure_ascii=False, indent=2, default=str)
            
            if 'content' in self.analysis_results:
                content_file = self.results_dir / 'content_analysis.json'
                with open(content_file, 'w', encoding='utf-8') as f:
                    json.dump(self.analysis_results['content'], f, ensure_ascii=False, indent=2, default=str)
        
        return report_file
    
    def run_complete_analysis(self):
        """Run the complete simplified analysis"""
        print(f"🚀 Running Optimized Dreams Analysis v2.0 for session {self.session_id}")
        
        # Load data
        if not self.load_optimized_data():
            return
        
        # Run analyses
        self.analyze_dream_quality()
        self.analyze_content_patterns()
        
        # Generate report
        report_file = self.generate_simplified_report()
        
        print(f"\n🎉 Analysis complete!")
        print(f"📁 Results directory: {self.results_dir}")
        print(f"📄 Main report: {report_file.name}")
        
        # Summary
        if 'quality' in self.analysis_results:
            quality = self.analysis_results['quality']
            print(f"\n📊 Quick Summary:")
            print(f"  • Total dreams: {quality['overall_stats']['total_dreams']}")
            print(f"  • Average length: {quality['overall_stats']['avg_words']:.0f} words")
            print(f"  • Languages analyzed: {quality['overall_stats']['total_languages']}")
            
            # Show best performing language
            if quality['by_language']:
                best_lang = max(quality['by_language'].items(), 
                              key=lambda x: x[1]['avg_words'])
                print(f"  • Longest dreams: {best_lang[0]} ({best_lang[1]['avg_words']:.0f} avg words)")

def main():
    """Main execution function"""
    if len(sys.argv) < 2:
        print("Usage: python analyze_optimized_v2.py <session_id>")
        print("Example: python analyze_optimized_v2.py OPT_V2_20250707_080149")
        print("\nAvailable sessions:")
        
        # Show available sessions
        logs_dir = Path('logs_optimized_v2')
        if logs_dir.exists():
            session_files = list(logs_dir.glob('session_summary_*.json'))
            for session_file in session_files:
                session_name = session_file.stem.replace('session_summary_', '')
                print(f"  • {session_name}")
        else:
            print("  No optimized v2 logs directory found")
        
        return
    
    session_id = sys.argv[1]
    
    analyzer = OptimizedDreamAnalyzerV2(session_id)
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main() 