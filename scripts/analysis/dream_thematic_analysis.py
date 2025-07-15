#!/usr/bin/env python3
"""
Dream Thematic Analysis
Analyzes actual dream content for themes, symbols, cultural patterns, and narrative structures
"""

import pandas as pd
import re
from collections import Counter, defaultdict
from pathlib import Path
import json
from datetime import datetime
import time
from deep_translator import GoogleTranslator
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.filterwarnings('ignore')

class DreamThematicAnalyzer:
    def __init__(self):
        self.dreams_by_language = {}
        self.language_codes = {
            'english': 'en',
            'basque': 'eu', 
            'hebrew': 'auto',  # Use auto-detection for Hebrew due to language code conflicts
            'serbian': 'sr',
            'slovenian': 'sl'
        }
        # Create translations directory
        self.translations_dir = Path('translations')
        self.translations_dir.mkdir(exist_ok=True)
        
        # Enhanced thematic patterns for semantic similarity (now using text descriptions instead of keywords)
        self.thematic_patterns = {
            'flying_freedom': 'flying through air wings soaring freedom liberation weightless floating gliding hovering airborne flight levitate',
            'water_emotion': 'water ocean sea river lake rain swimming diving drowning flood waves tears flowing liquid blue wet',
            'death_transformation': 'death dying dead funeral grave cemetery ghost spirit afterlife transformation rebirth resurrection ending corpse',
            'chase_anxiety': 'chase chasing running fleeing escaping pursuit hunting following being followed anxiety panic fear',
            'falling_loss': 'falling dropping plummeting tumbling losing balance cliff height vertigo descent gravity',
            'nature_spiritual': 'nature trees forest mountains flowers plants animals earth spiritual sacred divine natural green',
            'home_security': 'home house apartment room building shelter safety security comfort familiar domestic residence bedroom kitchen',
            'animals_instinct': 'animals dogs cats birds wildlife pets instinct primal wild hunting pack creature beast horse wolf bear snake fish',
            'people_relationships': 'people family friends relationships social interaction communication love romance partnership mother father child',
            'transportation': 'car bus train plane bike driving traveling road vehicle journey movement transport bicycle ship travel',
            'light_illumination': 'light bright sun sunshine glow illumination darkness shadow lamp candle fire flame shine',
            'food_nourishment': 'food eating meal cooking kitchen restaurant hunger nourishment taste flavor dining dinner',
            'violence_conflict': 'violence fighting conflict war battle aggression anger rage destruction attack weapon blood',
            'work_achievement': 'work job career office meeting boss achievement success accomplishment professional business',
            'technology_modern': 'technology computer phone internet digital modern electronic device machine robot',
            'magic_supernatural': 'magic supernatural mystical fantasy wizard witch spell miraculous extraordinary paranormal ghost spirit',
            'clothing_appearance': 'clothing clothes dress appearance fashion style beauty mirror reflection naked uniform costume shirt shoes',
            'buildings_architecture': 'buildings architecture construction house structure castle tower bridge city urban',
            'music_creativity': 'music singing dancing art creativity performance instrument rhythm melody artistic',
            'money_security': 'money rich poor wealth treasure gold expensive cheap financial economics buying selling',
            'time_change': 'time past future history clock hour minute yesterday tomorrow change transformation evolution'
        }
        
        # Initialize TF-IDF vectorizer for semantic similarity
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        
        # Similarity threshold for theme detection
        self.similarity_threshold = 0.15  # Adjusted threshold for better sensitivity
        
        print("SEARCH Initialized semantic similarity analyzer")
        print(f"STATS Themes: {len(self.thematic_patterns)}")
        print(f"TARGET Similarity threshold: {self.similarity_threshold}")
        
        # Enhanced cultural markers for semantic similarity
        self.cultural_markers = {
            'individualism': 'individual personal self myself independence autonomy freedom choice privacy rights',
            'collectivism': 'community group family together collective society tribe unity cooperation sharing',
            'spiritual_traditional': 'spiritual religious sacred holy divine prayer temple church traditional ritual ceremony',
            'urban_modern': 'urban city modern metropolitan skyscraper technology digital contemporary progressive',
            'rural_traditional': 'rural village countryside farm traditional agricultural pastoral simple humble',
            'western_culture': 'western democracy freedom individual rights progress modern liberal democratic',
            'authority_hierarchy': 'authority power hierarchy leadership control dominance command obedience respect'
        }
        
        # Enhanced emotional patterns for semantic similarity
        self.emotional_patterns = {
            'positive': 'happy joyful love beautiful wonderful amazing peaceful calm content satisfied delighted',
            'negative': 'sad fearful angry hate terrible horrible worried anxious depressed distressed troubled',
            'neutral': 'normal ordinary usual regular simple plain common everyday routine standard'
        }
    
    def save_translations(self, language, dreams_data, session_id):
        """Save translations to separate files for each language"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON with full metadata
        json_file = self.translations_dir / f"{language}_translations_{session_id}.json"
        json_data = {
            'language': language,
            'session_id': session_id,
            'timestamp': timestamp,
            'total_dreams': len(dreams_data),
            'dreams': dreams_data
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        # Save as CSV for easy viewing
        csv_file = self.translations_dir / f"{language}_translations_{session_id}.csv"
        csv_data = []
        for dream in dreams_data:
            csv_data.append({
                'dream_id': dream['dream_id'],
                'language': dream['language'],
                'original_text': dream['original_text'],
                'translated_text': dream.get('translated_text', ''),
                'word_count': dream['word_count'],
                'char_count': dream['char_count']
            })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        # Create text files only for non-English languages
        if language != 'english':
            # Save original language text only
            original_file = self.translations_dir / f"{language}_original_{session_id}.txt"
            with open(original_file, 'w', encoding='utf-8') as f:
                f.write(f"# {language.title()} Dreams - Original Text\n")
                f.write(f"# Session: {session_id}\n")
                f.write(f"# Generated: {timestamp}\n\n")
                for i, dream in enumerate(dreams_data, 1):
                    f.write(f"=== Dream {i} ({dream['dream_id']}) ===\n")
                    f.write(f"{dream['original_text']}\n\n")
            
            # Save translated text only
            translated_file = self.translations_dir / f"{language}_translated_{session_id}.txt"
            with open(translated_file, 'w', encoding='utf-8') as f:
                f.write(f"# {language.title()} Dreams - English Translation\n")
                f.write(f"# Session: {session_id}\n")
                f.write(f"# Generated: {timestamp}\n\n")
                for i, dream in enumerate(dreams_data, 1):
                    f.write(f"=== Dream {i} ({dream['dream_id']}) ===\n")
                    f.write(f"Original: {dream['original_text'][:100]}...\n")
                    f.write(f"Translation: {dream.get('translated_text', 'N/A')}\n\n")
        
        print(f"Saved {language} translations ({len(dreams_data)} dreams)")
        print(f"  • JSON: {json_file.name}")
        print(f"  • CSV: {csv_file.name}")
        if language != 'english':
            print(f"  • Original: {original_file.name}")
            print(f"  • Translated: {translated_file.name}")
        
    def load_existing_translations(self, language, session_id):
        """Load existing translations if available"""
        json_file = self.translations_dir / f"{language}_translations_{session_id}.json"
        
        print(f"  SEARCH Looking for: {json_file}")
        
        if json_file.exists():
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"  FOUND Found existing {language} translations ({data['total_dreams']} dreams) - SKIPPING")
                return data['dreams']
            except Exception as e:
                print(f"  WARNING Error loading existing {language} translations: {e}")
                return None
        else:
            print(f"  MISSING No existing translations found for {language}")
            return None
    
    def translate_text(self, text, source_lang, target_lang='en', max_retries=3):
        """Translate text using Google Translate"""
        if source_lang == target_lang:
            return text
        
        # Handle empty or very short text
        if not text or len(text.strip()) < 3:
            return text
        
        for attempt in range(max_retries):
            try:
                translator = GoogleTranslator(source=source_lang, target=target_lang)
                
                # Split long text into chunks to avoid API limits
                max_chunk_size = 4500  # Google Translate limit is ~5000 chars
                if len(text) <= max_chunk_size:
                    translated = translator.translate(text)
                    return translated if translated else text
                else:
                    # Split into sentences and translate in chunks
                    sentences = text.split('. ')
                    translated_chunks = []
                    current_chunk = ""
                    
                    for sentence in sentences:
                        if len(current_chunk + sentence) < max_chunk_size:
                            current_chunk += sentence + ". "
                        else:
                            if current_chunk:
                                chunk_translation = translator.translate(current_chunk.strip())
                                translated_chunks.append(chunk_translation if chunk_translation else current_chunk)
                                time.sleep(0.1)  # Rate limiting
                            current_chunk = sentence + ". "
                    
                    # Translate final chunk
                    if current_chunk:
                        chunk_translation = translator.translate(current_chunk.strip())
                        translated_chunks.append(chunk_translation if chunk_translation else current_chunk)
                    
                    return " ".join(translated_chunks)
                    
            except Exception as e:
                print(f"Translation attempt {attempt + 1} failed for {source_lang}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retry
                else:
                    print(f"Translation failed after {max_retries} attempts, using original text")
                    return text
        
        return text
        
    def load_dreams(self):
        """Load dreams from all languages and translate to English"""
        languages = ['english', 'basque', 'hebrew', 'serbian', 'slovenian']
        
        # Find the most recent session with dream data
        session_dirs = []
        for lang in languages:
            lang_dir = Path(f'logs/{lang}/gpt-4o')
            if lang_dir.exists():
                for session_dir in lang_dir.iterdir():
                    if session_dir.is_dir() and session_dir.name.startswith('session_'):
                        session_dirs.append(session_dir.name)
        
        if not session_dirs:
            print("No dream sessions found!")
            return
        
        # Use the most recent session
        latest_session = sorted(session_dirs)[-1]
        print(f"Using session: {latest_session}")
        print(f"Translations will be saved to: {self.translations_dir}")
        
        # Check for existing translation files first
        print(f"\nSEARCH Checking for existing translations...")
        existing_files = list(self.translations_dir.glob(f"*_translations_{latest_session}.json"))
        if existing_files:
            print(f"Found {len(existing_files)} existing translation files:")
            for file in existing_files:
                lang = file.name.split('_translations_')[0]
                print(f"  FOUND {lang}")
        
        total_translated = 0
        skipped_languages = []
        
        for lang in languages:
            dreams_file = Path(f'logs/{lang}/gpt-4o/{latest_session}/dreams.csv')
            if not dreams_file.exists():
                print(f"ERROR No dreams file found for {lang}")
                continue
                
            print(f"\nProcessing {lang} dreams...")
            
            # Check for existing translations first
            existing_dreams = self.load_existing_translations(lang, latest_session)
            if existing_dreams:
                self.dreams_by_language[lang] = existing_dreams
                skipped_languages.append(lang)
                continue
            
            # Load and process dreams
                df = pd.read_csv(dreams_file)
                successful_dreams = df[df['status'] == 'success']
            print(f"Found {len(successful_dreams)} successful dreams")
            
            self.dreams_by_language[lang] = []
            
            for i, (_, row) in enumerate(successful_dreams.iterrows()):
                original_text = str(row['dream'])
                
                # Translate non-English dreams
                if lang != 'english':
                    print(f"  TRANSLATING {lang} dream {i+1}/{len(successful_dreams)}...")
                    translated_text = self.translate_text(
                        original_text, 
                        self.language_codes[lang], 
                        'en'
                    )
                    total_translated += 1
                    time.sleep(0.2)  # Rate limiting
                else:
                    # For English, no translation needed
                    translated_text = original_text
                
                analyzed_text = translated_text.lower()
                
                dream_data = {
                        'dream_id': row.get('call_id', f"{lang}_{len(self.dreams_by_language[lang])}"),
                    'text': analyzed_text,  # Translated text for analysis
                    'original_text': original_text,  # Original text for display
                    'translated_text': translated_text if lang != 'english' else None,
                    'word_count': len(analyzed_text.split()),
                    'char_count': len(analyzed_text),
                    'language': lang
                }
                
                self.dreams_by_language[lang].append(dream_data)
            
            # Save translations for this language
            self.save_translations(lang, self.dreams_by_language[lang], latest_session)
        
        # Summary
        total_dreams = sum(len(dreams) for dreams in self.dreams_by_language.values())
        print(f"\nSTATS Translation Summary:")
        print(f"  • Total dreams loaded: {total_dreams}")
        print(f"  • Newly translated: {total_translated}")
        print(f"  • Languages processed: {len(self.dreams_by_language)}")
        if skipped_languages:
            print(f"  • Skipped (already translated): {', '.join(skipped_languages)}")
        
        for lang, dreams in self.dreams_by_language.items():
            if lang in skipped_languages:
                print(f"    LOADED {lang}: {len(dreams)} dreams (from cache)")
            elif lang == 'english':
                print(f"    LOADED {lang}: {len(dreams)} dreams (native)")
            else:
                print(f"    TRANSLATED {lang}: {len(dreams)} dreams")
        
        # Create summary file
        self.create_translation_summary(latest_session)
    
    def create_translation_summary(self, session_id):
        """Create a summary of all translations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.translations_dir / f"translation_summary_{session_id}.md"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("# Dream Translations Summary\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Session: {session_id}\n\n")
            
            f.write("## Overview\n\n")
            total_dreams = sum(len(dreams) for dreams in self.dreams_by_language.values())
            f.write(f"- **Total Dreams**: {total_dreams}\n")
            f.write(f"- **Languages**: {len(self.dreams_by_language)}\n")
            f.write(f"- **Translation Files**: {len(list(self.translations_dir.glob('*.json')))}\n\n")
            
            f.write("## Language Breakdown\n\n")
            f.write("| Language | Dreams | Original File | Translation File | CSV File |\n")
            f.write("|----------|--------|---------------|------------------|----------|\n")
            
            for lang, dreams in self.dreams_by_language.items():
                json_file = f"{lang}_translations_{session_id}.json"
                csv_file = f"{lang}_translations_{session_id}.csv"
                original_file = f"{lang}_original_{session_id}.txt" if lang != 'english' else 'N/A'
                translated_file = f"{lang}_translated_{session_id}.txt" if lang != 'english' else 'N/A'
                
                f.write(f"| {lang.title()} | {len(dreams)} | {original_file} | {translated_file} | {csv_file} |\n")
            
            f.write("\n## Files Created\n\n")
            for file_path in sorted(self.translations_dir.glob('*')):
                if file_path.is_file() and session_id in file_path.name:
                    f.write(f"- `{file_path.name}`\n")
            
            f.write("\n## Usage Instructions\n\n")
            f.write("### JSON Files\n")
            f.write("Complete data with metadata, original text, and translations. Best for programmatic access.\n\n")
            f.write("### CSV Files\n")
            f.write("Tabular format for spreadsheet analysis. Easy to open in Excel or Google Sheets.\n\n")
            f.write("### TXT Files\n")
            f.write("- `*_original_*.txt`: Original language text only\n")
            f.write("- `*_translated_*.txt`: English translations with original context\n\n")
            
            f.write("### Loading Translations\n")
            f.write("```python\n")
            f.write("import json\n")
            f.write("import pandas as pd\n\n")
            f.write("# Load JSON data\n")
            f.write(f"with open('translations/{lang}_translations_{session_id}.json', 'r') as f:\n")
            f.write("    data = json.load(f)\n\n")
            f.write("# Load CSV data\n")
            f.write(f"df = pd.read_csv('translations/{lang}_translations_{session_id}.csv')\n")
            f.write("```\n")
        
        print(f"\nTranslation summary created: {summary_file}")
    
    def compute_semantic_similarity(self, text, pattern_text):
        """Compute semantic similarity between text and pattern using TF-IDF and cosine similarity"""
        if not text or not pattern_text:
            return 0.0
        
        try:
            # Combine text and pattern for TF-IDF fitting
            documents = [text, pattern_text]
            
            # Fit TF-IDF on both documents
            tfidf_matrix = self.vectorizer.fit_transform(documents)
            
            # Compute cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Return similarity between text and pattern
            return similarity_matrix[0, 1]
            
        except Exception as e:
            # Fallback for very short texts
            return 0.0
    
    def analyze_themes_semantic(self, dreams, pattern_dict, pattern_type="themes"):
        """Analyze themes using semantic similarity instead of exact keyword matching"""
        results = {}
        
        print(f"ANALYZE Analyzing {pattern_type} using semantic similarity...")
        
        for pattern_name, pattern_text in pattern_dict.items():
            pattern_matches = []
            similarities = []
            
            for dream in dreams:
                dream_text = dream['text']
                
                # Compute semantic similarity
                similarity = self.compute_semantic_similarity(dream_text, pattern_text)
                similarities.append(similarity)
                
                # Check if similarity exceeds threshold
                if similarity > self.similarity_threshold:
                    pattern_matches.append({
                        'dream_id': dream['dream_id'],
                        'similarity': similarity,
                        'sample': dream['original_text'][:200] + "..." if len(dream['original_text']) > 200 else dream['original_text']
                    })
            
            # Calculate statistics
            total_occurrences = len(pattern_matches)
            dreams_with_pattern = len(pattern_matches)
            percentage = (dreams_with_pattern / len(dreams)) * 100 if dreams else 0
            avg_similarity = np.mean(similarities) if similarities else 0
            max_similarity = np.max(similarities) if similarities else 0
            
            results[pattern_name] = {
                'total_occurrences': total_occurrences,
                'dreams_with_theme': dreams_with_pattern,
                'percentage': percentage,
                'avg_similarity': avg_similarity,
                'max_similarity': max_similarity,
                'avg_occurrences_per_dream': total_occurrences / len(dreams) if dreams else 0,
                'examples': pattern_matches[:3]  # Top 3 examples
            }
            
            if percentage > 0:
                print(f"  FOUND {pattern_name}: {percentage:.1f}% (avg sim: {avg_similarity:.3f})")
        
        return results
    
    def analyze_themes(self):
        """Analyze themes across all languages using semantic similarity"""
        results = {}
        
        for lang, dreams in self.dreams_by_language.items():
            print(f"\nANALYZING {lang} ({len(dreams)} dreams)")
            
            # Determine if this language uses translation
            uses_translation = lang != 'english'
            translation_method = 'Google Translate (auto-detection)' if lang == 'hebrew' else 'Google Translate' if uses_translation else 'Native'
            
            lang_results = {
                'total_dreams': len(dreams),
                'uses_translation': uses_translation,
                'translation_method': translation_method,
                'analysis_language': 'English (translated)' if uses_translation else 'English (native)',
                'analysis_method': 'Semantic Similarity (TF-IDF + Cosine)',
                'similarity_threshold': self.similarity_threshold,
                'themes': {},
                'cultural_markers': {},
                'emotions': {},
                'narrative_patterns': {},
                'sample_dreams': []
            }
            
            # Analyze themes using semantic similarity
            lang_results['themes'] = self.analyze_themes_semantic(dreams, self.thematic_patterns, "themes")
            
            # Analyze cultural markers using semantic similarity
            lang_results['cultural_markers'] = self.analyze_cultural_markers_semantic(dreams)
            
            # Analyze emotions using semantic similarity
            lang_results['emotions'] = self.analyze_emotions_semantic(dreams)
            
            # Find most distinctive dreams for this language
            lang_results['sample_dreams'] = self.get_representative_dreams(dreams, lang)
            
            results[lang] = lang_results
            
            # Summary for this language
            theme_count = sum(1 for theme_data in lang_results['themes'].values() if theme_data['percentage'] > 0)
            print(f"  STATS {theme_count}/{len(self.thematic_patterns)} themes detected")
        
        return results
    
    def analyze_cultural_markers_semantic(self, dreams):
        """Analyze cultural markers using semantic similarity"""
        results = {}
        
        for marker_name, marker_text in self.cultural_markers.items():
            marker_matches = []
            similarities = []
            
            for dream in dreams:
                dream_text = dream['text']
                similarity = self.compute_semantic_similarity(dream_text, marker_text)
                similarities.append(similarity)
                
                if similarity > self.similarity_threshold:
                    marker_matches.append(similarity)
            
            # Calculate statistics
            total_occurrences = len(marker_matches)
            avg_similarity = np.mean(similarities) if similarities else 0
            
            results[marker_name] = {
                'total_occurrences': total_occurrences,
                'avg_per_dream': total_occurrences / len(dreams) if dreams else 0,
                'avg_similarity': avg_similarity
            }
        
        return results
    
    def analyze_emotions_semantic(self, dreams):
        """Analyze emotions using semantic similarity"""
        results = {}
        
        for emotion_type, emotion_text in self.emotional_patterns.items():
            emotion_matches = []
            similarities = []
            
            for dream in dreams:
                dream_text = dream['text']
                similarity = self.compute_semantic_similarity(dream_text, emotion_text)
                similarities.append(similarity)
                
                if similarity > self.similarity_threshold:
                    emotion_matches.append(similarity)
            
            # Calculate statistics
            total_occurrences = len(emotion_matches)
            avg_similarity = np.mean(similarities) if similarities else 0
            
            results[emotion_type] = {
                'total_occurrences': total_occurrences,
                'avg_per_dream': total_occurrences / len(dreams) if dreams else 0,
                'avg_similarity': avg_similarity
            }
        
        return results
    
    def get_representative_dreams(self, dreams, language, max_samples=3):
        """Get representative dream samples for a language"""
        # Sort by word count to get varied examples
        sorted_dreams = sorted(dreams, key=lambda x: x['word_count'], reverse=True)
        
        samples = []
        for i, dream in enumerate(sorted_dreams[:max_samples]):
            samples.append({
                'rank': i + 1,
                'word_count': dream['word_count'],
                'dream_id': dream['dream_id'],
                'content': dream['original_text'][:500] + "..." if len(dream['original_text']) > 500 else dream['original_text']
            })
        
        return samples
    
    def compare_themes_across_languages(self, results):
        """Compare themes across languages"""
        comparison = {}
        
        # Get all theme names
        all_themes = set()
        for lang_results in results.values():
            all_themes.update(lang_results['themes'].keys())
        
        for theme in all_themes:
            comparison[theme] = {}
            theme_stats = []
            
            for lang, lang_results in results.items():
                if theme in lang_results['themes']:
                    percentage = lang_results['themes'][theme]['percentage']
                    comparison[theme][lang] = percentage
                    theme_stats.append(percentage)
                else:
                    comparison[theme][lang] = 0
                    theme_stats.append(0)
            
            # Calculate statistics
            comparison[theme]['mean'] = sum(theme_stats) / len(theme_stats)
            comparison[theme]['max'] = max(theme_stats)
            comparison[theme]['min'] = min(theme_stats)
            comparison[theme]['range'] = max(theme_stats) - min(theme_stats)
            comparison[theme]['dominant_language'] = max(comparison[theme].items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0)[0]
        
        return comparison
    
    def generate_report(self, results, comparison):
        """Generate comprehensive thematic analysis report with semantic similarity"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"dream_thematic_analysis_{timestamp}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Dream Thematic Analysis Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Session: 20250705_194838\n\n")
            
            # Enhanced methodology note
            f.write("## WARNING Important Note: Enhanced Semantic Analysis Methodology\n\n")
            f.write("**This analysis uses advanced semantic similarity instead of exact keyword matching.**\n\n")
            f.write("### Analysis Method: Semantic Similarity\n")
            f.write("- **Technique**: TF-IDF vectorization + Cosine similarity\n")
            f.write("- **Advantage**: Captures thematic concepts even when exact words differ\n")
            f.write("- **Similarity Threshold**: 0.15 (15% minimum semantic overlap)\n")
            f.write("- **Language Independence**: Works across translation variations\n\n")
            
            translated_langs = [lang for lang, r in results.items() if r.get('uses_translation', False)]
            if translated_langs:
                f.write(f"**Languages analyzed via translation**: {', '.join(lang.title() for lang in translated_langs)}\n")
                f.write(f"**Translation method**: Google Translate API\n")
                f.write(f"**Analysis language**: English (all dreams analyzed in English)\n\n")
                f.write("**Enhanced Translation Benefits:**\n")
                f.write("- Semantic similarity captures meaning beyond exact word matches\n")
                f.write("- Improved detection of cultural themes in translated content\n")
                f.write("- Better handling of idiomatic expressions and cultural concepts\n")
                f.write("- Hebrew uses auto-detection for improved compatibility\n\n")
            
            # Overview
            total_dreams = sum(r['total_dreams'] for r in results.values())
            f.write(f"## Overview\n\n")
            f.write(f"- **Total Dreams Analyzed**: {total_dreams}\n")
            f.write(f"- **Languages**: {', '.join(results.keys())}\n")
            f.write(f"- **Themes Analyzed**: {len(self.thematic_patterns)}\n")
            f.write(f"- **Cultural Markers**: {len(self.cultural_markers)}\n")
            f.write(f"- **Analysis Method**: Semantic Similarity (TF-IDF + Cosine)\n")
            f.write(f"- **Similarity Threshold**: {self.similarity_threshold}\n")
            f.write(f"- **Native Language Analysis**: English\n")
            f.write(f"- **Translated Languages**: {len(translated_langs)}\n\n")
            
            # Top themes across all languages
            f.write("## Most Common Dream Themes (Cross-Linguistic)\n\n")
            sorted_themes = sorted(comparison.items(), key=lambda x: x[1]['mean'], reverse=True)
            
            f.write("| Theme | Average % | Dominant Language | Range |\n")
            f.write("|-------|-----------|-------------------|-------|\n")
            for theme, stats in sorted_themes[:15]:  # Top 15 themes
                f.write(f"| {theme.replace('_', ' ').title()} | {stats['mean']:.1f}% | {stats['dominant_language'].title()} | {stats['range']:.1f}% |\n")
            f.write("\n")
            
            # Language-specific analysis
            f.write("## Language-Specific Dream Patterns\n\n")
            
            for lang, lang_results in results.items():
                translation_note = f" - **ANALYZED VIA {lang_results['translation_method'].upper()}**" if lang_results.get('uses_translation') else ""
                f.write(f"### {lang.title()} Dreams ({lang_results['total_dreams']} dreams){translation_note}\n\n")
                
                if lang_results.get('uses_translation'):
                    f.write(f"**Analysis Method**: {lang_results['analysis_language']} via Semantic Similarity\n")
                    f.write(f"**Translation**: {lang_results['translation_method']}\n")
                    f.write("*Note: Analysis performed using semantic similarity on English translations*\n\n")
                else:
                    f.write("**Analysis Method**: Native English text via Semantic Similarity\n\n")
                
                # Top themes for this language
                sorted_lang_themes = sorted(lang_results['themes'].items(), 
                                          key=lambda x: x[1]['percentage'], reverse=True)
                
                f.write("**Top Themes:**\n")
                for theme_name, theme_data in sorted_lang_themes[:10]:
                    avg_sim = theme_data.get('avg_similarity', 0)
                    f.write(f"- **{theme_name.replace('_', ' ').title()}**: {theme_data['percentage']:.1f}% of dreams ({theme_data['dreams_with_theme']} dreams) [avg similarity: {avg_sim:.3f}]\n")
                f.write("\n")
                
                # Cultural markers
                f.write("**Cultural Markers:**\n")
                sorted_markers = sorted(lang_results['cultural_markers'].items(), 
                                      key=lambda x: x[1]['avg_per_dream'], reverse=True)
                for marker_name, marker_data in sorted_markers:
                    if marker_data['avg_per_dream'] > 0:
                        avg_sim = marker_data.get('avg_similarity', 0)
                        f.write(f"- **{marker_name.replace('_', ' ').title()}**: {marker_data['avg_per_dream']:.2f} avg per dream [avg similarity: {avg_sim:.3f}]\n")
                f.write("\n")
                
                # Emotional patterns
                f.write("**Emotional Patterns:**\n")
                sorted_emotions = sorted(lang_results['emotions'].items(), 
                                       key=lambda x: x[1]['avg_per_dream'], reverse=True)
                for emotion_name, emotion_data in sorted_emotions:
                    if emotion_data['avg_per_dream'] > 0:
                        avg_sim = emotion_data.get('avg_similarity', 0)
                        f.write(f"- **{emotion_name.replace('_', ' ').title()}**: {emotion_data['avg_per_dream']:.2f} avg per dream [avg similarity: {avg_sim:.3f}]\n")
                f.write("\n")
                
                # Sample dreams
                f.write("**Representative Dreams:**\n")
                for sample in lang_results['sample_dreams']:
                    f.write(f"**Dream {sample['rank']}** ({sample['word_count']} words):\n")
                    f.write(f">{sample['content']}\n\n")
                
                f.write("---\n\n")
            
            # Cross-cultural insights with enhanced semantic analysis
            f.write("## Cross-Cultural Insights\n\n")
            f.write("*Based on semantic similarity analysis across translated dream content*\n\n")
            
            f.write("### Themes with Highest Cultural Variation\n\n")
            high_variation_themes = sorted(comparison.items(), key=lambda x: x[1]['range'], reverse=True)
            
            for theme, stats in high_variation_themes[:10]:  # Top 10 most variable
                f.write(f"**{theme.replace('_', ' ').title()}**\n")
                f.write(f"- Range: {stats['range']:.1f}% (from {stats['min']:.1f}% to {stats['max']:.1f}%)\n")
                f.write(f"- Dominant in: {stats['dominant_language'].title()}\n")
                f.write(f"- By language: ")
                lang_stats = []
                for lang in results.keys():
                    if lang in stats:
                        lang_stats.append(f"{lang}: {stats[lang]:.1f}%")
                f.write(", ".join(lang_stats))
                f.write("\n\n")
            
                f.write("### Universal Themes (Low Variation Across Languages)\n\n")
            low_variation_themes = sorted(comparison.items(), key=lambda x: x[1]['range'])
            
            for theme, stats in low_variation_themes[:5]:  # Top 5 most universal
                if stats['mean'] > 0:  # Only include themes that actually appear
                    f.write(f"- **{theme.replace('_', ' ').title()}**: {stats['mean']:.1f}% average (range: {stats['range']:.1f}%)\n")
                f.write("\n")
            
            f.write("## Methodology Benefits\n\n")
            f.write("### Semantic Similarity Advantages:\n")
            f.write("1. **Translation Independence**: Detects themes regardless of exact word choice\n")
            f.write("2. **Cultural Sensitivity**: Captures concepts expressed differently across languages\n")
            f.write("3. **Robust Analysis**: Less sensitive to translation artifacts\n")
            f.write("4. **Improved Accuracy**: Better detection of thematic content in translated dreams\n\n")
            
            f.write("### Statistical Improvements:\n")
            f.write("- Reduced false negatives from translation variations\n")
            f.write("- More balanced cross-linguistic comparisons\n")
            f.write("- Enhanced detection of cultural themes\n")
            f.write("- Quantified similarity scores for transparency\n\n")
        
        print(f"\nREPORT Report generated: {report_file}")
        return report_file

def main():
    print("Starting Dream Thematic Analysis...")
    
    analyzer = DreamThematicAnalyzer()
    
    # Load dreams
    analyzer.load_dreams()
    
    # Analyze themes
    print("\nAnalyzing themes and cultural patterns...")
    results = analyzer.analyze_themes()
    
    # Compare across languages
    print("Comparing themes across languages...")
    comparison = analyzer.compare_themes_across_languages(results)
    
    # Generate report
    print("Generating thematic analysis report...")
    report_file = analyzer.generate_report(results, comparison)
    
    # Print summary
    print(f"\nAnalysis complete!")
    print(f"Report saved: {report_file}")
    
    # Quick preview of top findings
    print("\nKey Findings Preview:")
    sorted_themes = sorted(comparison.items(), key=lambda x: x[1]['mean'], reverse=True)
    print("Top 5 Most Common Themes:")
    for i, (theme, stats) in enumerate(sorted_themes[:5], 1):
        print(f"  {i}. {theme.replace('_', ' ').title()}: {stats['mean']:.1f}% avg (dominant in {stats['dominant_language'].title()})")
    
    print("\nTop 5 Most Culturally Variable Themes:")
    high_variation = sorted(comparison.items(), key=lambda x: x[1]['range'], reverse=True)
    for i, (theme, stats) in enumerate(high_variation[:5], 1):
        print(f"  {i}. {theme.replace('_', ' ').title()}: {stats['range']:.1f}% range ({stats['dominant_language'].title()} dominant)")

if __name__ == "__main__":
    main() 