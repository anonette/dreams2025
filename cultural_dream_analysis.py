#!/usr/bin/env python3
"""
Cultural Dream Analysis Tool
Implements Hall-Van de Castle (HVdC) system combined with Gottschalk-Gleser 
and cultural scripts theory for cross-linguistic dream analysis.
"""

import csv
import json
import re
import os
from collections import defaultdict, Counter
from pathlib import Path
import pandas as pd
from datetime import datetime

class CulturalDreamAnalyzer:
    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.languages = ["english", "serbian", "hebrew", "slovenian", "basque"]
        self.dream_data = {}
        self.analysis_results = {}
        
        # Cultural markers for different categories
        self.initialize_cultural_markers()
        
    def initialize_cultural_markers(self):
        """Initialize cultural and psychological markers for analysis"""
        
        # 1. Characters and Social Roles (HVdC)
        self.characters = {
            'familiar': ['family', 'mother', 'father', 'friend', 'brother', 'sister', 'child', 'parent'],
            'authority': ['teacher', 'boss', 'police', 'leader', 'king', 'queen', 'priest', 'doctor'],
            'gender_male': ['man', 'men', 'boy', 'boys', 'male', 'father', 'brother', 'son'],
            'gender_female': ['woman', 'women', 'girl', 'girls', 'female', 'mother', 'sister', 'daughter'],
            'cultural_figures': ['elder', 'shaman', 'guru', 'sage', 'ancestor', 'spirit', 'ghost'],
            'animals': ['dog', 'cat', 'bird', 'horse', 'lion', 'wolf', 'deer', 'owl', 'rabbit', 'fish']
        }
        
        # 2. Social Interactions (HVdC + Cultural Scripts)
        self.interactions = {
            'cooperative': ['help', 'assist', 'share', 'together', 'collaborate', 'support', 'join'],
            'aggressive': ['fight', 'attack', 'angry', 'argue', 'conflict', 'violence', 'war'],
            'sexual': ['love', 'romance', 'kiss', 'embrace', 'intimate', 'passion', 'attraction'],
            'punitive': ['punish', 'scold', 'blame', 'guilt', 'shame', 'judgment', 'disapproval'],
            'ceremonial': ['ritual', 'ceremony', 'celebration', 'wedding', 'funeral', 'festival'],
            'politeness': ['please', 'thank', 'respect', 'honor', 'bow', 'courtesy', 'formal']
        }
        
        # 3. Settings and Locales (HVdC + Cultural Scripts)
        self.settings = {
            'home': ['house', 'home', 'room', 'bedroom', 'kitchen', 'family', 'domestic'],
            'public': ['street', 'market', 'square', 'city', 'crowd', 'public', 'town'],
            'sacred': ['church', 'temple', 'mosque', 'shrine', 'holy', 'sacred', 'divine'],
            'secular': ['office', 'shop', 'mall', 'store', 'business', 'commercial'],
            'nature': ['forest', 'tree', 'mountain', 'river', 'ocean', 'garden', 'field', 'beach'],
            'built': ['building', 'structure', 'city', 'urban', 'construction', 'architecture'],
            'mythological': ['ancient', 'mystical', 'magical', 'enchanted', 'otherworldly'],
            'futuristic': ['future', 'technology', 'modern', 'advanced', 'digital']
        }
        
        # 4. Activities and Behaviors (HVdC + Cultural Scripts)
        self.activities = {
            'ritual': ['pray', 'worship', 'ceremony', 'ritual', 'blessing', 'meditation'],
            'cultural_gestures': ['bow', 'kneel', 'dance', 'sing', 'chant', 'gesture'],
            'communal': ['eat', 'feast', 'gather', 'meeting', 'community', 'share'],
            'technological': ['computer', 'phone', 'internet', 'digital', 'electronic'],
            'travel': ['journey', 'travel', 'walk', 'fly', 'drive', 'move', 'explore'],
            'creative': ['create', 'art', 'music', 'write', 'paint', 'craft', 'build']
        }
        
        # 5. Themes and Motifs (HVdC + Semiotic Analysis)
        self.themes = {
            'transformation': ['change', 'transform', 'become', 'shift', 'metamorphosis'],
            'exile': ['lost', 'alone', 'exile', 'banish', 'separate', 'isolated'],
            'confrontation': ['face', 'confront', 'challenge', 'oppose', 'struggle'],
            'freedom': ['free', 'freedom', 'liberate', 'escape', 'independence'],
            'purity': ['pure', 'clean', 'innocent', 'sacred', 'holy', 'pristine'],
            'progress': ['advance', 'progress', 'improve', 'grow', 'develop'],
            'death_rebirth': ['death', 'die', 'rebirth', 'reborn', 'resurrection'],
            'quest': ['search', 'seek', 'quest', 'journey', 'adventure', 'explore']
        }
        
        # 6. Emotions and Subjective States (Gottschalk-Gleser)
        self.emotions = {
            'anxiety': ['anxious', 'worried', 'nervous', 'fear', 'afraid', 'scared', 'panic'],
            'guilt': ['guilt', 'guilty', 'shame', 'ashamed', 'regret', 'remorse'],
            'hope': ['hope', 'hopeful', 'optimistic', 'positive', 'expect', 'anticipate'],
            'triumph': ['triumph', 'victory', 'success', 'achievement', 'accomplish'],
            'peace': ['peace', 'peaceful', 'calm', 'serene', 'tranquil', 'quiet'],
            'wonder': ['wonder', 'amazing', 'awe', 'marvel', 'astonish', 'magical'],
            'joy': ['joy', 'happy', 'delight', 'pleasure', 'bliss', 'elated'],
            'sadness': ['sad', 'sorrow', 'grief', 'melancholy', 'despair', 'mourn']
        }
        
        # 7. Dream Self and Agency (HVdC + Linguistic Anthropology)
        self.agency_markers = {
            'active_agency': ['I decided', 'I chose', 'I acted', 'I created', 'I led'],
            'passive_agency': ['I was led', 'I was taken', 'I found myself', 'I was given'],
            'observer': ['I watched', 'I saw', 'I observed', 'I witnessed', 'I noticed'],
            'participant': ['I joined', 'I participated', 'I engaged', 'I interacted'],
            'transformed': ['I became', 'I turned into', 'I transformed', 'I changed'],
            'reflexive': ['I realized', 'I understood', 'I dreamed', 'I knew']
        }

    def load_dream_data(self):
        """Load dream data from all language sessions"""
        print("Loading dream data from all languages...")
        
        for language in self.languages:
            lang_data = []
            lang_dir = self.logs_dir / language / "gpt-4o"
            
            if not lang_dir.exists():
                print(f"  Warning: No data directory found for {language}")
                continue
                
            # Find session directories
            session_dirs = [d for d in lang_dir.iterdir() if d.is_dir() and d.name.startswith("session_")]
            
            for session_dir in session_dirs:
                dreams_file = session_dir / "dreams.csv"
                if dreams_file.exists():
                    try:
                        with open(dreams_file, 'r', encoding='utf-8') as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                if row['status'] == 'success' and row['dream']:
                                    lang_data.append({
                                        'dream_id': row['call_id'],
                                        'dream_text': row['dream'],
                                        'session_id': row['session_id'],
                                        'language': language,
                                        'language_code': row['language_code'],
                                        'script': row['script'],
                                        'timestamp': row['timestamp']
                                    })
                    except Exception as e:
                        print(f"  Error loading {dreams_file}: {e}")
            
            self.dream_data[language] = lang_data
            print(f"  Loaded {len(lang_data)} dreams for {language}")
        
        total_dreams = sum(len(data) for data in self.dream_data.values())
        print(f"Total dreams loaded: {total_dreams}")

    def analyze_categories(self, dream_text, language):
        """Analyze a single dream across all cultural categories"""
        text_lower = dream_text.lower()
        results = {}
        
        # 1. Characters and Social Roles
        results['characters'] = {}
        for category, markers in self.characters.items():
            count = sum(1 for marker in markers if marker in text_lower)
            results['characters'][category] = count
        
        # 2. Social Interactions
        results['interactions'] = {}
        for category, markers in self.interactions.items():
            count = sum(1 for marker in markers if marker in text_lower)
            results['interactions'][category] = count
        
        # 3. Settings and Locales
        results['settings'] = {}
        for category, markers in self.settings.items():
            count = sum(1 for marker in markers if marker in text_lower)
            results['settings'][category] = count
        
        # 4. Activities and Behaviors
        results['activities'] = {}
        for category, markers in self.activities.items():
            count = sum(1 for marker in markers if marker in text_lower)
            results['activities'][category] = count
        
        # 5. Themes and Motifs
        results['themes'] = {}
        for category, markers in self.themes.items():
            count = sum(1 for marker in markers if marker in text_lower)
            results['themes'][category] = count
        
        # 6. Emotions and Subjective States
        results['emotions'] = {}
        for category, markers in self.emotions.items():
            count = sum(1 for marker in markers if marker in text_lower)
            results['emotions'][category] = count
        
        # 7. Dream Self and Agency
        results['agency'] = {}
        for category, markers in self.agency_markers.items():
            count = sum(1 for marker in markers if marker in text_lower)
            results['agency'][category] = count
        
        # Calculate word count and dream length
        results['metrics'] = {
            'word_count': len(dream_text.split()),
            'char_count': len(dream_text),
            'sentence_count': len([s for s in dream_text.split('.') if s.strip()])
        }
        
        return results

    def run_cultural_analysis(self):
        """Run comprehensive cultural analysis on all dream data"""
        print("\n=== Running Cultural Dream Analysis ===")
        
        for language in self.languages:
            if language not in self.dream_data or not self.dream_data[language]:
                continue
                
            print(f"\nAnalyzing {language} dreams...")
            lang_results = {
                'language': language,
                'total_dreams': len(self.dream_data[language]),
                'categories': {
                    'characters': defaultdict(int),
                    'interactions': defaultdict(int),
                    'settings': defaultdict(int),
                    'activities': defaultdict(int),
                    'themes': defaultdict(int),
                    'emotions': defaultdict(int),
                    'agency': defaultdict(int)
                },
                'metrics': {
                    'total_words': 0,
                    'total_chars': 0,
                    'total_sentences': 0,
                    'avg_words': 0,
                    'avg_chars': 0,
                    'avg_sentences': 0
                },
                'individual_dreams': []
            }
            
            # Analyze each dream
            for dream in self.dream_data[language]:
                dream_analysis = self.analyze_categories(dream['dream_text'], language)
                lang_results['individual_dreams'].append({
                    'dream_id': dream['dream_id'],
                    'analysis': dream_analysis
                })
                
                # Aggregate results
                for category in ['characters', 'interactions', 'settings', 'activities', 'themes', 'emotions', 'agency']:
                    for subcategory, count in dream_analysis[category].items():
                        lang_results['categories'][category][subcategory] += count
                
                # Aggregate metrics
                lang_results['metrics']['total_words'] += dream_analysis['metrics']['word_count']
                lang_results['metrics']['total_chars'] += dream_analysis['metrics']['char_count']
                lang_results['metrics']['total_sentences'] += dream_analysis['metrics']['sentence_count']
            
            # Calculate averages
            total_dreams = lang_results['total_dreams']
            if total_dreams > 0:
                lang_results['metrics']['avg_words'] = lang_results['metrics']['total_words'] / total_dreams
                lang_results['metrics']['avg_chars'] = lang_results['metrics']['total_chars'] / total_dreams
                lang_results['metrics']['avg_sentences'] = lang_results['metrics']['total_sentences'] / total_dreams
            
            self.analysis_results[language] = lang_results
        
        print("Cultural analysis complete!")

    def generate_cultural_report(self):
        """Generate comprehensive cultural analysis report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"cultural_dream_analysis_report_{timestamp}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Cultural Dream Analysis Report\n")
            f.write("## Hall-Van de Castle (HVdC) + Gottschalk-Gleser + Cultural Scripts Analysis\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overview
            f.write("## Overview\n\n")
            f.write("This analysis applies the Hall-Van de Castle system combined with Gottschalk-Gleser emotional analysis and cultural scripts theory to examine GPT-4o generated dreams across five languages.\n\n")
            
            # Summary Statistics
            f.write("## Summary by Language\n\n")
            f.write("| Language | Dreams | Avg Words | Avg Characters | Script System |\n")
            f.write("|----------|--------|-----------|----------------|---------------|\n")
            
            for language in self.languages:
                if language in self.analysis_results:
                    results = self.analysis_results[language]
                    script = self.dream_data[language][0]['script'] if self.dream_data[language] else 'Unknown'
                    f.write(f"| {language.title()} | {results['total_dreams']} | {results['metrics']['avg_words']:.1f} | {results['metrics']['avg_chars']:.1f} | {script} |\n")
            f.write("\n")
            
            # Detailed Analysis by Category
            categories = [
                ('characters', 'Characters and Social Roles (HVdC)'),
                ('interactions', 'Social Interactions (HVdC + Cultural Scripts)'),
                ('settings', 'Settings and Locales (HVdC + Cultural Scripts)'),
                ('activities', 'Activities and Behaviors (HVdC + Cultural Scripts)'),
                ('themes', 'Themes and Motifs (HVdC + Semiotic Analysis)'),
                ('emotions', 'Emotions and Subjective States (Gottschalk-Gleser)'),
                ('agency', 'Dream Self and Agency (HVdC + Linguistic Anthropology)')
            ]
            
            for category_key, category_title in categories:
                f.write(f"## {category_title}\n\n")
                
                # Create comparison table
                all_subcategories = set()
                for lang_results in self.analysis_results.values():
                    all_subcategories.update(lang_results['categories'][category_key].keys())
                
                if all_subcategories:
                    f.write("| Subcategory | " + " | ".join(lang.title() for lang in self.languages if lang in self.analysis_results) + " |\n")
                    f.write("|-------------|" + "|".join(["--------"] * len([l for l in self.languages if l in self.analysis_results])) + "|\n")
                    
                    for subcategory in sorted(all_subcategories):
                        row = f"| {subcategory.replace('_', ' ').title()} | "
                        values = []
                        for language in self.languages:
                            if language in self.analysis_results:
                                count = self.analysis_results[language]['categories'][category_key].get(subcategory, 0)
                                total_dreams = self.analysis_results[language]['total_dreams']
                                percentage = (count / total_dreams * 100) if total_dreams > 0 else 0
                                values.append(f"{count} ({percentage:.1f}%)")
                        row += " | ".join(values) + " |\n"
                        f.write(row)
                f.write("\n")
            
            # Cross-Cultural Insights
            f.write("## Cross-Cultural Insights\n\n")
            
            # Dominant settings by language
            f.write("### Dominant Settings by Language\n")
            for language in self.languages:
                if language in self.analysis_results:
                    settings = self.analysis_results[language]['categories']['settings']
                    if settings:
                        top_setting = max(settings.items(), key=lambda x: x[1])
                        f.write(f"- **{language.title()}**: {top_setting[0].replace('_', ' ').title()} ({top_setting[1]} occurrences)\n")
            f.write("\n")
            
            # Emotional patterns
            f.write("### Emotional Patterns\n")
            for language in self.languages:
                if language in self.analysis_results:
                    emotions = self.analysis_results[language]['categories']['emotions']
                    if emotions:
                        top_emotion = max(emotions.items(), key=lambda x: x[1])
                        f.write(f"- **{language.title()}**: {top_emotion[0].title()} ({top_emotion[1]} occurrences)\n")
            f.write("\n")
            
            # Agency patterns
            f.write("### Agency Patterns\n")
            for language in self.languages:
                if language in self.analysis_results:
                    agency = self.analysis_results[language]['categories']['agency']
                    if agency:
                        top_agency = max(agency.items(), key=lambda x: x[1])
                        f.write(f"- **{language.title()}**: {top_agency[0].replace('_', ' ').title()} ({top_agency[1]} occurrences)\n")
            f.write("\n")
            
            # Cultural Script Analysis
            f.write("## Cultural Script Analysis\n\n")
            f.write("### Public vs Private Sphere Orientation\n")
            for language in self.languages:
                if language in self.analysis_results:
                    settings = self.analysis_results[language]['categories']['settings']
                    home_score = settings.get('home', 0)
                    public_score = settings.get('public', 0)
                    total = home_score + public_score
                    if total > 0:
                        private_ratio = home_score / total * 100
                        public_ratio = public_score / total * 100
                        f.write(f"- **{language.title()}**: {private_ratio:.1f}% Private, {public_ratio:.1f}% Public\n")
            f.write("\n")
            
            f.write("### Sacred vs Secular Content\n")
            for language in self.languages:
                if language in self.analysis_results:
                    settings = self.analysis_results[language]['categories']['settings']
                    sacred_score = settings.get('sacred', 0)
                    secular_score = settings.get('secular', 0)
                    total = sacred_score + secular_score
                    if total > 0:
                        sacred_ratio = sacred_score / total * 100
                        secular_ratio = secular_score / total * 100
                        f.write(f"- **{language.title()}**: {sacred_ratio:.1f}% Sacred, {secular_ratio:.1f}% Secular\n")
            f.write("\n")
            
            # Recommendations
            f.write("## Research Recommendations\n\n")
            f.write("1. **Script System Effects**: Compare dream imagery patterns across Latin, Cyrillic, and Hebrew scripts\n")
            f.write("2. **Cultural Memory**: Analyze mythological vs modern themes by language background\n")
            f.write("3. **Emotional Expression**: Examine cultural norms around emotional display in dream narratives\n")
            f.write("4. **Agency Patterns**: Investigate active vs passive agency across individualistic vs collectivistic cultures\n")
            f.write("5. **Spatial Metaphors**: Study nature vs built environment preferences by geographic origin\n\n")
            
            f.write("## Data Export\n\n")
            f.write(f"- Raw analysis data: cultural_analysis_data_{timestamp}.json\n")
            f.write(f"- Dream-by-dream breakdown: individual_dream_analysis_{timestamp}.csv\n")
        
        # Export raw data
        json_file = f"cultural_analysis_data_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)
        
        # Export individual dream analysis
        csv_file = f"individual_dream_analysis_{timestamp}.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['language', 'dream_id', 'word_count', 'char_count'] + \
                        [f"{cat}_{subcat}" for cat in ['characters', 'interactions', 'settings', 'activities', 'themes', 'emotions', 'agency']
                         for subcat in getattr(self, cat).keys()]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for language, lang_results in self.analysis_results.items():
                for dream_data in lang_results['individual_dreams']:
                    row = {
                        'language': language,
                        'dream_id': dream_data['dream_id'],
                        'word_count': dream_data['analysis']['metrics']['word_count'],
                        'char_count': dream_data['analysis']['metrics']['char_count']
                    }
                    
                    # Add category scores
                    for category in ['characters', 'interactions', 'settings', 'activities', 'themes', 'emotions', 'agency']:
                        for subcategory, count in dream_data['analysis'][category].items():
                            row[f"{category}_{subcategory}"] = count
                    
                    writer.writerow(row)
        
        print(f"\nCultural analysis report generated:")
        print(f"  - Report: {report_file}")
        print(f"  - Raw data: {json_file}")
        print(f"  - CSV export: {csv_file}")
        
        return report_file

    def run_full_analysis(self):
        """Run complete cultural dream analysis pipeline"""
        self.load_dream_data()
        self.run_cultural_analysis()
        report_file = self.generate_cultural_report()
        return report_file

def main():
    """Main execution function"""
    print("=== Cultural Dream Analysis Tool ===")
    print("Hall-Van de Castle + Gottschalk-Gleser + Cultural Scripts Theory")
    print("=" * 60)
    
    analyzer = CulturalDreamAnalyzer()
    report_file = analyzer.run_full_analysis()
    
    print(f"\nAnalysis complete! Report saved as: {report_file}")

if __name__ == "__main__":
    main()
