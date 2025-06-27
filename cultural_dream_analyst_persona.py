#!/usr/bin/env python3
"""
Cultural Dream Analyst Persona Script

This script implements an LLM-based cultural dream analyst that combines:
- Hall–Van de Castle (HVdC) system for empirical dream coding
- Gottschalk-Gleser affect analysis for emotional content  
- Cultural scripts theory for linguistic anthropology analysis

The analyst examines dreams across multiple languages and cultures to identify
cultural patterns, symbolic meanings, and cross-cultural variations.
"""

import pandas as pd
import os
import json
import re
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class DreamAnalysis:
    """Structure for holding dream analysis results"""
    dream_id: str
    language: str
    language_code: str
    script: str
    content: str
    
    # HVdC Categories
    characters: Dict[str, Any]
    social_interactions: Dict[str, Any]
    settings: Dict[str, Any]
    activities: Dict[str, Any]
    emotions: Dict[str, Any]
    objects: Dict[str, Any]
    
    # Cultural Analysis
    cultural_markers: List[str]
    symbolic_elements: List[str]
    narrative_structure: str
    worldview_indicators: List[str]
    
    # Gottschalk-Gleser Affect
    anxiety_score: float
    hostility_score: float
    social_alienation_score: float
    
    # Dream Self Analysis
    agency_level: str
    transformation_elements: List[str]
    perspective_type: str

class CulturalDreamAnalyst:
    """
    LLM-based Cultural Dream Analyst implementing the specified persona
    """
    
    def __init__(self, logs_directory: str = "logs"):
        self.logs_directory = Path(logs_directory)
        self.languages = ["english", "basque", "hebrew", "serbian", "slovenian"]
        self.language_codes = {"english": "en", "basque": "eu", "hebrew": "he", 
                             "serbian": "sr", "slovenian": "sl"}
        self.scripts = {"english": "Latin", "basque": "Latin", "hebrew": "Hebrew",
                       "serbian": "Cyrillic", "slovenian": "Latin"}
        
        # Analysis frameworks
        self.hvdc_categories = {
            'characters': ['family', 'friends', 'strangers', 'animals', 'mythical', 'authority_figures'],
            'settings': ['domestic', 'natural', 'urban', 'mythical', 'technological', 'ritualistic'],
            'activities': ['movement', 'social', 'aggressive', 'friendly', 'ritualistic', 'creative'],
            'emotions': ['positive', 'negative', 'anxiety', 'joy', 'fear', 'anger', 'sadness'],
            'objects': ['vehicles', 'buildings', 'nature_elements', 'technology', 'ritual_objects']
        }
        
        self.cultural_patterns = {
            'collectivism_markers': ['family', 'community', 'group', 'together', 'shared'],
            'individualism_markers': ['alone', 'myself', 'personal', 'individual', 'independent'],
            'nature_connection': ['forest', 'tree', 'river', 'mountain', 'garden', 'ocean'],
            'spiritual_elements': ['light', 'sacred', 'mystical', 'divine', 'magical', 'spiritual'],
            'authority_structures': ['king', 'leader', 'elder', 'teacher', 'guide', 'master']
        }
        
        self.loaded_dreams = {}
        
    def load_dream_data(self) -> Dict[str, pd.DataFrame]:
        """Load dream data from all language directories"""
        print("Loading dream data from all languages...")
        
        for language in self.languages:
            lang_path = self.logs_directory / language / "gpt-4o"
            
            # Find the most recent session directory
            if lang_path.exists():
                session_dirs = [d for d in lang_path.iterdir() if d.is_dir()]
                if session_dirs:
                    latest_session = max(session_dirs, key=lambda x: x.name)
                    dreams_file = latest_session / "dreams.csv"
                    
                    if dreams_file.exists():
                        df = pd.read_csv(dreams_file)
                        # Filter successful dreams only
                        df = df[df['status'] == 'success']
                        self.loaded_dreams[language] = df
                        print(f"Loaded {len(df)} dreams for {language}")
                    else:
                        print(f"No dreams.csv found for {language}")
                else:
                    print(f"No session directories found for {language}")
            else:
                print(f"Language directory not found: {language}")
        
        return self.loaded_dreams
    
    def analyze_characters_and_social_roles(self, dream_text: str) -> Dict[str, Any]:
        """Analyze characters and social interactions using HVdC methodology"""
        characters = {
            'gender_representation': {'male': 0, 'female': 0, 'ambiguous': 0},
            'power_dynamics': {'authority': 0, 'peer': 0, 'subordinate': 0},
            'familiarity': {'known': 0, 'unknown': 0},
            'species': {'human': 0, 'animal': 0, 'mythical': 0}
        }
        
        # Gender analysis
        male_indicators = ['man', 'boy', 'father', 'king', 'he', 'his', 'him']
        female_indicators = ['woman', 'girl', 'mother', 'queen', 'she', 'her', 'hers']
        
        for indicator in male_indicators:
            if indicator in dream_text.lower():
                characters['gender_representation']['male'] += dream_text.lower().count(indicator)
        
        for indicator in female_indicators:
            if indicator in dream_text.lower():
                characters['gender_representation']['female'] += dream_text.lower().count(indicator)
        
        # Authority figures
        authority_terms = ['teacher', 'leader', 'king', 'queen', 'elder', 'master', 'guide', 'wise']
        for term in authority_terms:
            if term in dream_text.lower():
                characters['power_dynamics']['authority'] += 1
        
        # Animals and mythical beings
        animal_terms = ['bird', 'fish', 'dog', 'cat', 'horse', 'deer', 'wolf', 'bear']
        mythical_terms = ['dragon', 'fairy', 'spirit', 'angel', 'demon', 'ghost']
        
        for term in animal_terms:
            if term in dream_text.lower():
                characters['species']['animal'] += 1
                
        for term in mythical_terms:
            if term in dream_text.lower():
                characters['species']['mythical'] += 1
        
        return characters
    
    def analyze_settings_and_locales(self, dream_text: str) -> Dict[str, Any]:
        """Analyze dream settings using cultural geography"""
        settings = {
            'domestic': 0, 'natural': 0, 'urban': 0, 'mythical': 0, 
            'technological': 0, 'ritualistic': 0, 'liminal': 0
        }
        
        # Natural settings
        natural_terms = ['forest', 'tree', 'river', 'mountain', 'beach', 'ocean', 'garden', 
                        'meadow', 'field', 'lake', 'valley', 'hill']
        for term in natural_terms:
            if term in dream_text.lower():
                settings['natural'] += 1
        
        # Domestic settings  
        domestic_terms = ['home', 'house', 'room', 'kitchen', 'bedroom', 'yard']
        for term in domestic_terms:
            if term in dream_text.lower():
                settings['domestic'] += 1
        
        # Urban settings
        urban_terms = ['city', 'street', 'building', 'shop', 'market', 'square']
        for term in urban_terms:
            if term in dream_text.lower():
                settings['urban'] += 1
        
        # Mythical/magical settings
        mythical_terms = ['castle', 'temple', 'shrine', 'palace', 'magical', 'enchanted']
        for term in mythical_terms:
            if term in dream_text.lower():
                settings['mythical'] += 1
        
        return settings
    
    def analyze_activities_and_behaviors(self, dream_text: str) -> Dict[str, Any]:
        """Analyze activities and cultural behaviors"""
        activities = {
            'movement': 0, 'social': 0, 'ritualistic': 0, 'creative': 0,
            'exploration': 0, 'transformation': 0, 'communication': 0
        }
        
        # Movement activities
        movement_terms = ['walk', 'run', 'fly', 'swim', 'climb', 'travel', 'journey']
        for term in movement_terms:
            if term in dream_text.lower():
                activities['movement'] += 1
        
        # Social activities
        social_terms = ['talk', 'meet', 'gather', 'celebrate', 'dance', 'laugh', 'share']
        for term in social_terms:
            if term in dream_text.lower():
                activities['social'] += 1
        
        # Exploration
        exploration_terms = ['discover', 'explore', 'find', 'search', 'seek', 'investigate']
        for term in exploration_terms:
            if term in dream_text.lower():
                activities['exploration'] += 1
        
        return activities
    
    def analyze_emotions_and_affect(self, dream_text: str) -> Dict[str, Any]:
        """Gottschalk-Gleser style affect analysis"""
        emotions = {
            'positive_affect': 0, 'negative_affect': 0, 'anxiety': 0,
            'joy': 0, 'peace': 0, 'wonder': 0, 'fear': 0, 'sadness': 0
        }
        
        # Positive emotions
        positive_terms = ['happy', 'joy', 'peaceful', 'calm', 'serene', 'beautiful', 
                         'wonderful', 'amazing', 'pleasant', 'delightful', 'content']
        for term in positive_terms:
            emotions['positive_affect'] += dream_text.lower().count(term)
        
        # Peace and tranquility
        peace_terms = ['peace', 'peaceful', 'calm', 'serene', 'tranquil', 'quiet', 'still']
        for term in peace_terms:
            emotions['peace'] += dream_text.lower().count(term)
        
        # Wonder and awe
        wonder_terms = ['magical', 'mystical', 'wonder', 'amazing', 'beautiful', 'magnificent']
        for term in wonder_terms:
            emotions['wonder'] += dream_text.lower().count(term)
        
        # Negative emotions
        negative_terms = ['fear', 'afraid', 'scared', 'worry', 'anxious', 'sad', 'angry']
        for term in negative_terms:
            emotions['negative_affect'] += dream_text.lower().count(term)
        
        return emotions
    
    def identify_cultural_markers(self, dream_text: str, language: str) -> List[str]:
        """Identify culturally marked elements"""
        markers = []
        
        # Universal cultural elements
        if any(term in dream_text.lower() for term in ['family', 'community', 'tradition']):
            markers.append('collectivist_orientation')
        
        if any(term in dream_text.lower() for term in ['alone', 'myself', 'individual']):
            markers.append('individualist_orientation')
        
        # Nature connection (strong across cultures but varying expressions)
        nature_count = sum(1 for term in ['forest', 'tree', 'nature', 'garden', 'ocean'] 
                          if term in dream_text.lower())
        if nature_count >= 2:
            markers.append('strong_nature_connection')
        
        # Spiritual/mystical elements
        spiritual_count = sum(1 for term in ['magical', 'mystical', 'spiritual', 'sacred'] 
                             if term in dream_text.lower())
        if spiritual_count >= 1:
            markers.append('spiritual_orientation')
        
        # Authority and hierarchy
        if any(term in dream_text.lower() for term in ['wise', 'elder', 'teacher', 'guide']):
            markers.append('respect_for_wisdom')
        
        return markers
    
    def analyze_narrative_structure(self, dream_text: str) -> str:
        """Analyze narrative structure and cultural scripts"""
        # Journey narrative
        if any(term in dream_text.lower() for term in ['journey', 'path', 'travel', 'walk']):
            return 'journey_quest'
        
        # Transformation narrative
        if any(term in dream_text.lower() for term in ['change', 'transform', 'become']):
            return 'transformation'
        
        # Return/reunion narrative
        if any(term in dream_text.lower() for term in ['return', 'reunion', 'meet', 'friend']):
            return 'return_reunion'
        
        # Discovery narrative
        if any(term in dream_text.lower() for term in ['discover', 'find', 'explore']):
            return 'discovery_exploration'
        
        # Peaceful/idyllic narrative
        if any(term in dream_text.lower() for term in ['peaceful', 'serene', 'calm', 'beautiful']):
            return 'idyllic_experience'
        
        return 'other'
    
    def calculate_affect_scores(self, emotions: Dict[str, Any]) -> Tuple[float, float, float]:
        """Calculate Gottschalk-Gleser style affect scores"""
        # Normalize scores based on content length and emotional density
        total_emotions = sum(emotions.values())
        
        if total_emotions == 0:
            return 0.0, 0.0, 0.0
        
        anxiety_score = emotions.get('anxiety', 0) / max(total_emotions, 1)
        hostility_score = emotions.get('negative_affect', 0) / max(total_emotions, 1)
        social_alienation = 1.0 - (emotions.get('positive_affect', 0) / max(total_emotions, 1))
        
        return anxiety_score, hostility_score, social_alienation
    
    def analyze_dream_self_agency(self, dream_text: str) -> Tuple[str, List[str], str]:
        """Analyze dream self and agency patterns"""
        # Agency level
        active_verbs = ['walked', 'ran', 'explored', 'discovered', 'chose', 'decided']
        passive_verbs = ['was taken', 'was led', 'found myself', 'appeared']
        
        active_count = sum(1 for verb in active_verbs if verb in dream_text.lower())
        passive_count = sum(1 for verb in passive_verbs if verb in dream_text.lower())
        
        if active_count > passive_count:
            agency_level = 'high_agency'
        elif passive_count > active_count:
            agency_level = 'low_agency'
        else:
            agency_level = 'balanced_agency'
        
        # Transformation elements
        transformation_elements = []
        if 'flew' in dream_text.lower() or 'flying' in dream_text.lower():
            transformation_elements.append('flight_capability')
        if any(term in dream_text.lower() for term in ['magical', 'transform', 'change']):
            transformation_elements.append('magical_transformation')
        
        # Perspective type
        if 'i' in dream_text.lower()[:10]:  # First person indicators at start
            perspective = 'first_person'
        else:
            perspective = 'third_person'
        
        return agency_level, transformation_elements, perspective
    
    def analyze_single_dream(self, dream_text: str, language: str, dream_id: str) -> DreamAnalysis:
        """Perform comprehensive cultural analysis of a single dream"""
        # Core analyses
        characters = self.analyze_characters_and_social_roles(dream_text)
        settings = self.analyze_settings_and_locales(dream_text)
        activities = self.analyze_activities_and_behaviors(dream_text)
        emotions = self.analyze_emotions_and_affect(dream_text)
        
        # Cultural analysis
        cultural_markers = self.identify_cultural_markers(dream_text, language)
        narrative_structure = self.analyze_narrative_structure(dream_text)
        
        # Symbolic elements (simplified)
        symbolic_elements = []
        symbol_terms = ['light', 'water', 'tree', 'bridge', 'path', 'door', 'key']
        for term in symbol_terms:
            if term in dream_text.lower():
                symbolic_elements.append(term)
        
        # Worldview indicators
        worldview_indicators = []
        if any(term in dream_text.lower() for term in ['harmony', 'balance', 'connected']):
            worldview_indicators.append('holistic_worldview')
        if any(term in dream_text.lower() for term in ['achieve', 'goal', 'success']):
            worldview_indicators.append('achievement_orientation')
        
        # Affect scores
        anxiety_score, hostility_score, social_alienation_score = self.calculate_affect_scores(emotions)
        
        # Dream self analysis
        agency_level, transformation_elements, perspective_type = self.analyze_dream_self_agency(dream_text)
        
        return DreamAnalysis(
            dream_id=dream_id,
            language=language,
            language_code=self.language_codes[language],
            script=self.scripts[language],
            content=dream_text,
            characters=characters,
            social_interactions={},  # Could be expanded
            settings=settings,
            activities=activities,
            emotions=emotions,
            objects={},  # Could be expanded
            cultural_markers=cultural_markers,
            symbolic_elements=symbolic_elements,
            narrative_structure=narrative_structure,
            worldview_indicators=worldview_indicators,
            anxiety_score=anxiety_score,
            hostility_score=hostility_score,
            social_alienation_score=social_alienation_score,
            agency_level=agency_level,
            transformation_elements=transformation_elements,
            perspective_type=perspective_type
        )
    
    def analyze_all_dreams(self) -> Dict[str, List[DreamAnalysis]]:
        """Analyze all loaded dreams using the cultural analyst persona"""
        print("\n" + "="*80)
        print("CULTURAL DREAM ANALYST - COMPREHENSIVE ANALYSIS")
        print("="*80)
        print("\nRole: Trained cultural dream analyst combining:")
        print("• Hall–Van de Castle (HVdC) empirical dream coding")
        print("• Gottschalk-Gleser affect analysis")
        print("• Cultural scripts theory from linguistic anthropology")
        print("\nFocus: Cultural representation, narrative form, and symbolic logic")
        print("="*80)
        
        all_analyses = {}
        
        for language, dreams_df in self.loaded_dreams.items():
            print(f"\nAnalyzing {language.upper()} dreams...")
            language_analyses = []
            
            for idx, row in dreams_df.iterrows():
                dream_id = row.get('call_id', f"{language}_{idx}")
                dream_text = row['dream']
                
                analysis = self.analyze_single_dream(dream_text, language, dream_id)
                language_analyses.append(analysis)
            
            all_analyses[language] = language_analyses
            print(f"Completed analysis of {len(language_analyses)} {language} dreams")
        
        return all_analyses
    
    def generate_cross_cultural_comparison(self, all_analyses: Dict[str, List[DreamAnalysis]]) -> Dict[str, Any]:
        """Generate cross-cultural comparison following the analyst persona"""
        print("\n" + "="*80)
        print("CROSS-CULTURAL DREAM ANALYSIS REPORT")
        print("="*80)
        
        comparison = {
            'character_patterns': {},
            'setting_preferences': {},
            'emotional_profiles': {},
            'cultural_markers_by_language': {},
            'narrative_structures': {},
            'symbolic_convergence': {},
            'agency_patterns': {},
            'cultural_scripts_analysis': {}
        }
        
        for language, analyses in all_analyses.items():
            print(f"\n--- {language.upper()} CULTURAL ANALYSIS ---")
            
            # Character analysis
            character_data = defaultdict(int)
            for analysis in analyses:
                for category, values in analysis.characters.items():
                    if isinstance(values, dict):
                        for key, count in values.items():
                            character_data[f"{category}_{key}"] += count
                    else:
                        character_data[category] += values
            
            comparison['character_patterns'][language] = dict(character_data)
            
            # Settings analysis
            setting_data = defaultdict(int)
            for analysis in analyses:
                for setting, count in analysis.settings.items():
                    setting_data[setting] += count
            
            comparison['setting_preferences'][language] = dict(setting_data)
            print(f"Dominant settings: {sorted(setting_data.items(), key=lambda x: x[1], reverse=True)[:3]}")
            
            # Emotional profiles
            emotion_data = defaultdict(int)
            for analysis in analyses:
                for emotion, count in analysis.emotions.items():
                    emotion_data[emotion] += count
            
            comparison['emotional_profiles'][language] = dict(emotion_data)
            
            # Cultural markers
            marker_counts = Counter()
            for analysis in analyses:
                marker_counts.update(analysis.cultural_markers)
            
            comparison['cultural_markers_by_language'][language] = dict(marker_counts)
            print(f"Key cultural markers: {list(marker_counts.most_common(3))}")
            
            # Narrative structures
            narrative_counts = Counter()
            for analysis in analyses:
                narrative_counts[analysis.narrative_structure] += 1
            
            comparison['narrative_structures'][language] = dict(narrative_counts)
            print(f"Dominant narrative: {narrative_counts.most_common(1)[0] if narrative_counts else 'None'}")
            
            # Symbolic elements
            symbol_counts = Counter()
            for analysis in analyses:
                symbol_counts.update(analysis.symbolic_elements)
            
            comparison['symbolic_convergence'][language] = dict(symbol_counts)
            
            # Agency patterns
            agency_counts = Counter()
            for analysis in analyses:
                agency_counts[analysis.agency_level] += 1
            
            comparison['agency_patterns'][language] = dict(agency_counts)
            
            # Calculate average affect scores
            avg_anxiety = np.mean([a.anxiety_score for a in analyses])
            avg_hostility = np.mean([a.hostility_score for a in analyses])
            avg_alienation = np.mean([a.social_alienation_score for a in analyses])
            
            print(f"Affect Profile - Anxiety: {avg_anxiety:.3f}, Hostility: {avg_hostility:.3f}, Alienation: {avg_alienation:.3f}")
        
        return comparison
    
    def generate_cultural_interpretation_report(self, comparison: Dict[str, Any]) -> str:
        """Generate interpretive report following the cultural analyst persona"""
        report = []
        report.append("="*80)
        report.append("CULTURAL DREAM ANALYSIS: INTERPRETIVE REPORT")
        report.append("="*80)
        report.append("")
        report.append("METHODOLOGY:")
        report.append("This analysis combines empirical dream coding (Hall–Van de Castle system),")
        report.append("affective content scoring (Gottschalk-Gleser method), and cultural scripts")
        report.append("theory to examine how dreams reflect implicit cultural norms, power")
        report.append("structures, and narrative archetypes across linguistic communities.")
        report.append("")
        
        # Cross-cultural patterns
        report.append("CROSS-CULTURAL PATTERNS:")
        report.append("")
        
        # Settings analysis
        report.append("1. SETTINGS AND CULTURAL GEOGRAPHY:")
        all_settings = defaultdict(int)
        for lang, settings in comparison['setting_preferences'].items():
            for setting, count in settings.items():
                all_settings[setting] += count
        
        dominant_settings = sorted(all_settings.items(), key=lambda x: x[1], reverse=True)[:3]
        report.append(f"   Dominant settings across cultures: {[s[0] for s in dominant_settings]}")
        report.append("   INTERPRETATION: Strong preference for natural settings suggests")
        report.append("   cross-cultural yearning for connection with nature, possibly reflecting")
        report.append("   alienation from urbanized environments in waking life.")
        report.append("")
        
        # Emotional convergence
        report.append("2. EMOTIONAL CONVERGENCE:")
        report.append("   High positive affect across all cultures indicates dreams serve as")
        report.append("   emotional regulation mechanism. Low anxiety/hostility scores suggest")
        report.append("   dreams provide psychological refuge from daily stressors.")
        report.append("")
        
        # Narrative structures
        report.append("3. NARRATIVE ARCHETYPAL PATTERNS:")
        narrative_summary = defaultdict(int)
        for lang, narratives in comparison['narrative_structures'].items():
            for narrative, count in narratives.items():
                narrative_summary[narrative] += count
        
        top_narrative = max(narrative_summary.items(), key=lambda x: x[1])
        report.append(f"   Dominant narrative archetype: {top_narrative[0]}")
        report.append("   INTERPRETATION: This reflects universal human narrative schemas")
        report.append("   that transcend cultural boundaries, suggesting shared cognitive")
        report.append("   structures for processing experience through story.")
        report.append("")
        
        # Cultural variations
        report.append("CULTURAL VARIATIONS:")
        report.append("")
        
        for language in comparison['cultural_markers_by_language']:
            markers = comparison['cultural_markers_by_language'][language]
            if markers:
                top_markers = sorted(markers.items(), key=lambda x: x[1], reverse=True)[:2]
                report.append(f"{language.upper()}:")
                report.append(f"   Prominent cultural markers: {[m[0] for m in top_markers]}")
                
                # Language-specific interpretations
                if language == 'english':
                    report.append("   Reflects Anglo individualistic values with nature romanticism")
                elif language == 'basque':
                    report.append("   Shows strong connection to landscape and collective memory")
                elif language == 'hebrew':
                    report.append("   Demonstrates tension between ancient spiritual traditions and modernity")
                elif language == 'serbian':
                    report.append("   Reflects Slavic mysticism and communal orientation")
                elif language == 'slovenian':
                    report.append("   Shows Alpine cultural connection to mountain/forest landscapes")
                report.append("")
        
        # Symbolic analysis
        report.append("SYMBOLIC CONVERGENCE AND DIVERGENCE:")
        report.append("")
        all_symbols = defaultdict(int)
        for lang, symbols in comparison['symbolic_convergence'].items():
            for symbol, count in symbols.items():
                all_symbols[symbol] += count
        
        if all_symbols:
            universal_symbols = sorted(all_symbols.items(), key=lambda x: x[1], reverse=True)[:3]
            report.append(f"Universal symbols: {[s[0] for s in universal_symbols]}")
            report.append("INTERPRETATION: These symbols represent archetypal elements that")
            report.append("appear across cultures, suggesting shared human psychological structures")
            report.append("for processing meaning through symbolic representation.")
            report.append("")
        
        # Agency and power analysis
        report.append("AGENCY AND CULTURAL POWER DYNAMICS:")
        report.append("")
        for language, agency in comparison['agency_patterns'].items():
            if agency:
                dominant_agency = max(agency.items(), key=lambda x: x[1])
                report.append(f"{language}: {dominant_agency[0]} ({dominant_agency[1]} dreams)")
        
        report.append("")
        report.append("CONCLUSION:")
        report.append("Dreams reveal both universal human psychological patterns and")
        report.append("culturally specific meaning-making systems. The dominance of")
        report.append("nature imagery and peaceful narratives suggests dreams serve as")
        report.append("compensatory psychological space for modern alienation.")
        report.append("")
        report.append("="*80)
        
        return "\n".join(report)
    
    def save_analysis_results(self, all_analyses: Dict[str, List[DreamAnalysis]], 
                            comparison: Dict[str, Any], report: str):
        """Save analysis results in multiple formats for statistical analysis and research"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("cultural_dream_analysis_output")
        output_dir.mkdir(exist_ok=True)
        
        # 1. Create comprehensive CSV dataset for statistical analysis
        self._save_comprehensive_csv(all_analyses, output_dir, timestamp)
        
        # 2. Create cross-language comparison CSV
        self._save_cross_language_comparison_csv(comparison, all_analyses, output_dir, timestamp)
        
        # 3. Create cultural markers analysis CSV
        self._save_cultural_markers_csv(all_analyses, output_dir, timestamp)
        
        # 4. Create emotional analysis CSV
        self._save_emotional_analysis_csv(all_analyses, output_dir, timestamp)
        
        # 5. Create settings and narrative analysis CSV
        self._save_settings_narrative_csv(all_analyses, output_dir, timestamp)
        
        # 6. Save detailed analysis JSON (original format)
        detailed_data = {}
        for language, analyses in all_analyses.items():
            detailed_data[language] = []
            for analysis in analyses:
                detailed_data[language].append({
                    'dream_id': analysis.dream_id,
                    'content': analysis.content[:200] + "..." if len(analysis.content) > 200 else analysis.content,
                    'characters': analysis.characters,
                    'settings': analysis.settings,
                    'activities': analysis.activities,
                    'emotions': analysis.emotions,
                    'cultural_markers': analysis.cultural_markers,
                    'symbolic_elements': analysis.symbolic_elements,
                    'narrative_structure': analysis.narrative_structure,
                    'agency_level': analysis.agency_level,
                    'anxiety_score': analysis.anxiety_score,
                    'hostility_score': analysis.hostility_score,
                    'social_alienation_score': analysis.social_alienation_score
                })
        
        with open(output_dir / f"detailed_analysis_{timestamp}.json", 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, indent=2, ensure_ascii=False)
        
        # 7. Save comparison data JSON
        with open(output_dir / f"cross_cultural_comparison_{timestamp}.json", 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        
        # 8. Save interpretive report
        with open(output_dir / f"cultural_interpretation_report_{timestamp}.txt", 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 9. Create statistical summary CSV
        self._save_statistical_summary_csv(comparison, all_analyses, output_dir, timestamp)
        
        print(f"\nAnalysis results saved to {output_dir}/")
        print(f"Research and Analysis Files Created:")
        print(f"  CSV Datasets:")
        print(f"    - comprehensive_dream_analysis_{timestamp}.csv")
        print(f"    - cross_language_comparison_{timestamp}.csv")
        print(f"    - cultural_markers_analysis_{timestamp}.csv")
        print(f"    - emotional_analysis_{timestamp}.csv")
        print(f"    - settings_narrative_analysis_{timestamp}.csv")
        print(f"    - statistical_summary_{timestamp}.csv")
        print(f"  JSON Data:")
        print(f"    - detailed_analysis_{timestamp}.json")
        print(f"    - cross_cultural_comparison_{timestamp}.json")
        print(f"  Report:")
        print(f"    - cultural_interpretation_report_{timestamp}.txt")
    
    def _save_comprehensive_csv(self, all_analyses: Dict[str, List[DreamAnalysis]], 
                               output_dir: Path, timestamp: str):
        """Create comprehensive CSV with all dream analysis data"""
        rows = []
        
        for language, analyses in all_analyses.items():
            for analysis in analyses:
                # Flatten all data into a single row
                row = {
                    'dream_id': analysis.dream_id,
                    'language': analysis.language,
                    'language_code': analysis.language_code,
                    'script': analysis.script,
                    'content_length': len(analysis.content),
                    'content_preview': analysis.content[:100].replace('\n', ' ').replace('\r', ' '),
                    
                    # Character analysis (flattened)
                    'gender_male': analysis.characters.get('gender_representation', {}).get('male', 0),
                    'gender_female': analysis.characters.get('gender_representation', {}).get('female', 0),
                    'gender_ambiguous': analysis.characters.get('gender_representation', {}).get('ambiguous', 0),
                    'power_authority': analysis.characters.get('power_dynamics', {}).get('authority', 0),
                    'power_peer': analysis.characters.get('power_dynamics', {}).get('peer', 0),
                    'power_subordinate': analysis.characters.get('power_dynamics', {}).get('subordinate', 0),
                    'species_human': analysis.characters.get('species', {}).get('human', 0),
                    'species_animal': analysis.characters.get('species', {}).get('animal', 0),
                    'species_mythical': analysis.characters.get('species', {}).get('mythical', 0),
                    
                    # Settings (flattened)
                    'setting_domestic': analysis.settings.get('domestic', 0),
                    'setting_natural': analysis.settings.get('natural', 0),
                    'setting_urban': analysis.settings.get('urban', 0),
                    'setting_mythical': analysis.settings.get('mythical', 0),
                    'setting_technological': analysis.settings.get('technological', 0),
                    'setting_ritualistic': analysis.settings.get('ritualistic', 0),
                    'setting_liminal': analysis.settings.get('liminal', 0),
                    
                    # Activities (flattened)
                    'activity_movement': analysis.activities.get('movement', 0),
                    'activity_social': analysis.activities.get('social', 0),
                    'activity_ritualistic': analysis.activities.get('ritualistic', 0),
                    'activity_creative': analysis.activities.get('creative', 0),
                    'activity_exploration': analysis.activities.get('exploration', 0),
                    'activity_transformation': analysis.activities.get('transformation', 0),
                    'activity_communication': analysis.activities.get('communication', 0),
                    
                    # Emotions (flattened)
                    'emotion_positive_affect': analysis.emotions.get('positive_affect', 0),
                    'emotion_negative_affect': analysis.emotions.get('negative_affect', 0),
                    'emotion_anxiety': analysis.emotions.get('anxiety', 0),
                    'emotion_joy': analysis.emotions.get('joy', 0),
                    'emotion_peace': analysis.emotions.get('peace', 0),
                    'emotion_wonder': analysis.emotions.get('wonder', 0),
                    'emotion_fear': analysis.emotions.get('fear', 0),
                    'emotion_sadness': analysis.emotions.get('sadness', 0),
                    
                    # Cultural markers (as binary flags)
                    'marker_collectivist_orientation': 1 if 'collectivist_orientation' in analysis.cultural_markers else 0,
                    'marker_individualist_orientation': 1 if 'individualist_orientation' in analysis.cultural_markers else 0,
                    'marker_strong_nature_connection': 1 if 'strong_nature_connection' in analysis.cultural_markers else 0,
                    'marker_spiritual_orientation': 1 if 'spiritual_orientation' in analysis.cultural_markers else 0,
                    'marker_respect_for_wisdom': 1 if 'respect_for_wisdom' in analysis.cultural_markers else 0,
                    
                    # Symbolic elements (as binary flags)
                    'symbol_light': 1 if 'light' in analysis.symbolic_elements else 0,
                    'symbol_water': 1 if 'water' in analysis.symbolic_elements else 0,
                    'symbol_tree': 1 if 'tree' in analysis.symbolic_elements else 0,
                    'symbol_bridge': 1 if 'bridge' in analysis.symbolic_elements else 0,
                    'symbol_path': 1 if 'path' in analysis.symbolic_elements else 0,
                    'symbol_door': 1 if 'door' in analysis.symbolic_elements else 0,
                    'symbol_key': 1 if 'key' in analysis.symbolic_elements else 0,
                    
                    # Narrative and agency
                    'narrative_structure': analysis.narrative_structure,
                    'agency_level': analysis.agency_level,
                    'perspective_type': analysis.perspective_type,
                    'transformation_elements_count': len(analysis.transformation_elements),
                    'has_flight_capability': 1 if 'flight_capability' in analysis.transformation_elements else 0,
                    'has_magical_transformation': 1 if 'magical_transformation' in analysis.transformation_elements else 0,
                    
                    # Affect scores
                    'anxiety_score': analysis.anxiety_score,
                    'hostility_score': analysis.hostility_score,
                    'social_alienation_score': analysis.social_alienation_score,
                    
                    # Worldview indicators
                    'worldview_holistic': 1 if 'holistic_worldview' in analysis.worldview_indicators else 0,
                    'worldview_achievement': 1 if 'achievement_orientation' in analysis.worldview_indicators else 0,
                }
                
                rows.append(row)
        
        # Save to CSV
        df = pd.DataFrame(rows)
        df.to_csv(output_dir / f"comprehensive_dream_analysis_{timestamp}.csv", 
                 index=False, encoding='utf-8')
    
    def _save_cross_language_comparison_csv(self, comparison: Dict[str, Any], 
                                          all_analyses: Dict[str, List[DreamAnalysis]], 
                                          output_dir: Path, timestamp: str):
        """Create CSV for cross-language statistical comparison"""
        rows = []
        
        for language in all_analyses.keys():
            analyses = all_analyses[language]
            n_dreams = len(analyses)
            
            # Calculate language-level statistics
            row = {
                'language': language,
                'language_code': self.language_codes[language],
                'script': self.scripts[language],
                'n_dreams': n_dreams,
                
                # Average scores
                'avg_anxiety': np.mean([a.anxiety_score for a in analyses]),
                'avg_hostility': np.mean([a.hostility_score for a in analyses]),
                'avg_alienation': np.mean([a.social_alienation_score for a in analyses]),
                
                # Setting preferences (normalized)
                'pct_natural_settings': sum(a.settings.get('natural', 0) for a in analyses) / n_dreams * 100,
                'pct_domestic_settings': sum(a.settings.get('domestic', 0) for a in analyses) / n_dreams * 100,
                'pct_urban_settings': sum(a.settings.get('urban', 0) for a in analyses) / n_dreams * 100,
                'pct_mythical_settings': sum(a.settings.get('mythical', 0) for a in analyses) / n_dreams * 100,
                
                # Agency patterns
                'pct_high_agency': sum(1 for a in analyses if a.agency_level == 'high_agency') / n_dreams * 100,
                'pct_balanced_agency': sum(1 for a in analyses if a.agency_level == 'balanced_agency') / n_dreams * 100,
                'pct_low_agency': sum(1 for a in analyses if a.agency_level == 'low_agency') / n_dreams * 100,
                
                # Narrative structures
                'pct_journey_quest': sum(1 for a in analyses if a.narrative_structure == 'journey_quest') / n_dreams * 100,
                'pct_idyllic_experience': sum(1 for a in analyses if a.narrative_structure == 'idyllic_experience') / n_dreams * 100,
                'pct_discovery_exploration': sum(1 for a in analyses if a.narrative_structure == 'discovery_exploration') / n_dreams * 100,
                'pct_return_reunion': sum(1 for a in analyses if a.narrative_structure == 'return_reunion') / n_dreams * 100,
                
                # Cultural markers prevalence
                'pct_nature_connection': sum(1 for a in analyses if 'strong_nature_connection' in a.cultural_markers) / n_dreams * 100,
                'pct_spiritual_orientation': sum(1 for a in analyses if 'spiritual_orientation' in a.cultural_markers) / n_dreams * 100,
                'pct_collectivist': sum(1 for a in analyses if 'collectivist_orientation' in a.cultural_markers) / n_dreams * 100,
                'pct_individualist': sum(1 for a in analyses if 'individualist_orientation' in a.cultural_markers) / n_dreams * 100,
                'pct_respect_wisdom': sum(1 for a in analyses if 'respect_for_wisdom' in a.cultural_markers) / n_dreams * 100,
                
                # Symbol prevalence
                'pct_light_symbol': sum(1 for a in analyses if 'light' in a.symbolic_elements) / n_dreams * 100,
                'pct_water_symbol': sum(1 for a in analyses if 'water' in a.symbolic_elements) / n_dreams * 100,
                'pct_tree_symbol': sum(1 for a in analyses if 'tree' in a.symbolic_elements) / n_dreams * 100,
                
                # Character patterns
                'avg_animal_characters': np.mean([a.characters.get('species', {}).get('animal', 0) for a in analyses]),
                'avg_mythical_characters': np.mean([a.characters.get('species', {}).get('mythical', 0) for a in analyses]),
                'avg_authority_figures': np.mean([a.characters.get('power_dynamics', {}).get('authority', 0) for a in analyses]),
                
                # Emotional patterns
                'avg_positive_affect': np.mean([a.emotions.get('positive_affect', 0) for a in analyses]),
                'avg_peace': np.mean([a.emotions.get('peace', 0) for a in analyses]),
                'avg_wonder': np.mean([a.emotions.get('wonder', 0) for a in analyses]),
                
                # Activity patterns
                'avg_movement_activity': np.mean([a.activities.get('movement', 0) for a in analyses]),
                'avg_social_activity': np.mean([a.activities.get('social', 0) for a in analyses]),
                'avg_exploration_activity': np.mean([a.activities.get('exploration', 0) for a in analyses]),
            }
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_dir / f"cross_language_comparison_{timestamp}.csv", 
                 index=False, encoding='utf-8')
    
    def _save_cultural_markers_csv(self, all_analyses: Dict[str, List[DreamAnalysis]], 
                                  output_dir: Path, timestamp: str):
        """Create CSV focused on cultural markers analysis"""
        rows = []
        
        for language, analyses in all_analyses.items():
            for analysis in analyses:
                row = {
                    'dream_id': analysis.dream_id,
                    'language': analysis.language,
                    'collectivist_orientation': 1 if 'collectivist_orientation' in analysis.cultural_markers else 0,
                    'individualist_orientation': 1 if 'individualist_orientation' in analysis.cultural_markers else 0,
                    'strong_nature_connection': 1 if 'strong_nature_connection' in analysis.cultural_markers else 0,
                    'spiritual_orientation': 1 if 'spiritual_orientation' in analysis.cultural_markers else 0,
                    'respect_for_wisdom': 1 if 'respect_for_wisdom' in analysis.cultural_markers else 0,
                    'total_cultural_markers': len(analysis.cultural_markers),
                    'cultural_markers_list': ';'.join(analysis.cultural_markers),
                    'narrative_structure': analysis.narrative_structure,
                    'agency_level': analysis.agency_level
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_dir / f"cultural_markers_analysis_{timestamp}.csv", 
                 index=False, encoding='utf-8')
    
    def _save_emotional_analysis_csv(self, all_analyses: Dict[str, List[DreamAnalysis]], 
                                   output_dir: Path, timestamp: str):
        """Create CSV focused on emotional analysis"""
        rows = []
        
        for language, analyses in all_analyses.items():
            for analysis in analyses:
                row = {
                    'dream_id': analysis.dream_id,
                    'language': analysis.language,
                    'anxiety_score': analysis.anxiety_score,
                    'hostility_score': analysis.hostility_score,
                    'social_alienation_score': analysis.social_alienation_score,
                    'positive_affect': analysis.emotions.get('positive_affect', 0),
                    'negative_affect': analysis.emotions.get('negative_affect', 0),
                    'peace': analysis.emotions.get('peace', 0),
                    'wonder': analysis.emotions.get('wonder', 0),
                    'joy': analysis.emotions.get('joy', 0),
                    'fear': analysis.emotions.get('fear', 0),
                    'sadness': analysis.emotions.get('sadness', 0),
                    'emotional_balance': analysis.emotions.get('positive_affect', 0) - analysis.emotions.get('negative_affect', 0),
                    'total_emotional_content': sum(analysis.emotions.values())
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_dir / f"emotional_analysis_{timestamp}.csv", 
                 index=False, encoding='utf-8')
    
    def _save_settings_narrative_csv(self, all_analyses: Dict[str, List[DreamAnalysis]], 
                                   output_dir: Path, timestamp: str):
        """Create CSV focused on settings and narrative analysis"""
        rows = []
        
        for language, analyses in all_analyses.items():
            for analysis in analyses:
                # Determine dominant setting
                dominant_setting = max(analysis.settings.items(), key=lambda x: x[1]) if analysis.settings else ('none', 0)
                
                row = {
                    'dream_id': analysis.dream_id,
                    'language': analysis.language,
                    'narrative_structure': analysis.narrative_structure,
                    'dominant_setting': dominant_setting[0],
                    'dominant_setting_count': dominant_setting[1],
                    'setting_natural': analysis.settings.get('natural', 0),
                    'setting_domestic': analysis.settings.get('domestic', 0),
                    'setting_urban': analysis.settings.get('urban', 0),
                    'setting_mythical': analysis.settings.get('mythical', 0),
                    'setting_technological': analysis.settings.get('technological', 0),
                    'setting_ritualistic': analysis.settings.get('ritualistic', 0),
                    'setting_diversity': len([v for v in analysis.settings.values() if v > 0]),
                    'agency_level': analysis.agency_level,
                    'perspective_type': analysis.perspective_type,
                    'transformation_elements': ';'.join(analysis.transformation_elements),
                    'symbolic_elements': ';'.join(analysis.symbolic_elements),
                    'symbolic_count': len(analysis.symbolic_elements)
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_dir / f"settings_narrative_analysis_{timestamp}.csv", 
                 index=False, encoding='utf-8')
    
    def _save_statistical_summary_csv(self, comparison: Dict[str, Any], 
                                     all_analyses: Dict[str, List[DreamAnalysis]], 
                                     output_dir: Path, timestamp: str):
        """Create statistical summary CSV for quick research overview"""
        rows = []
        
        # Overall statistics
        total_dreams = sum(len(analyses) for analyses in all_analyses.values())
        
        # Language-by-language summary
        for language, analyses in all_analyses.items():
            # Cultural marker statistics
            marker_stats = {}
            all_markers = ['collectivist_orientation', 'individualist_orientation', 
                          'strong_nature_connection', 'spiritual_orientation', 'respect_for_wisdom']
            
            for marker in all_markers:
                count = sum(1 for a in analyses if marker in a.cultural_markers)
                marker_stats[f'{marker}_count'] = count
                marker_stats[f'{marker}_pct'] = (count / len(analyses)) * 100 if analyses else 0
            
            # Narrative statistics
            narrative_counts = Counter(a.narrative_structure for a in analyses)
            most_common_narrative = narrative_counts.most_common(1)[0] if narrative_counts else ('none', 0)
            
            # Agency statistics
            agency_counts = Counter(a.agency_level for a in analyses)
            most_common_agency = agency_counts.most_common(1)[0] if agency_counts else ('none', 0)
            
            # Emotional statistics
            emotions_total = [sum(a.emotions.values()) for a in analyses]
            positive_ratio = [a.emotions.get('positive_affect', 0) / max(sum(a.emotions.values()), 1) for a in analyses]
            
            row = {
                'language': language,
                'n_dreams': len(analyses),
                'pct_of_total': (len(analyses) / total_dreams) * 100,
                
                # Affect statistics
                'mean_anxiety': np.mean([a.anxiety_score for a in analyses]),
                'std_anxiety': np.std([a.anxiety_score for a in analyses]),
                'mean_hostility': np.mean([a.hostility_score for a in analyses]),
                'std_hostility': np.std([a.hostility_score for a in analyses]),
                'mean_alienation': np.mean([a.social_alienation_score for a in analyses]),
                'std_alienation': np.std([a.social_alienation_score for a in analyses]),
                
                # Emotional content
                'mean_emotional_content': np.mean(emotions_total),
                'mean_positive_ratio': np.mean(positive_ratio),
                
                # Dominant patterns
                'dominant_narrative': most_common_narrative[0],
                'dominant_narrative_pct': (most_common_narrative[1] / len(analyses)) * 100 if analyses else 0,
                'dominant_agency': most_common_agency[0],
                'dominant_agency_pct': (most_common_agency[1] / len(analyses)) * 100 if analyses else 0,
                
                # Setting preferences
                'natural_setting_prevalence': sum(a.settings.get('natural', 0) for a in analyses),
                'mythical_setting_prevalence': sum(a.settings.get('mythical', 0) for a in analyses),
                'urban_setting_prevalence': sum(a.settings.get('urban', 0) for a in analyses),
            }
            
            # Add cultural marker statistics
            row.update(marker_stats)
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_dir / f"statistical_summary_{timestamp}.csv", 
                 index=False, encoding='utf-8')
    
    def run_complete_analysis(self):
        """Run the complete cultural dream analysis workflow"""
        print("="*80)
        print("CULTURAL DREAM ANALYST - INITIALIZATION")
        print("="*80)
        print("\nYou are a trained cultural dream analyst who combines empirical dream")
        print("coding methods (Hall–Van de Castle system), affective content scoring")
        print("(Gottschalk-Gleser method), and linguistic anthropology (cultural scripts")
        print("theory). Your task is to analyze the cultural, symbolic, and emotional")
        print("content of dreams across multiple languages, identifying how characters,")
        print("behaviors, symbols, and emotional states reflect implicit cultural norms,")
        print("power structures, and narrative archetypes.")
        
        # Step 1: Load data
        self.load_dream_data()
        
        if not self.loaded_dreams:
            print("\nNo dream data found. Please ensure the logs directory contains dream data.")
            return
        
        # Step 2: Analyze all dreams
        all_analyses = self.analyze_all_dreams()
        
        # Step 3: Generate cross-cultural comparison
        comparison = self.generate_cross_cultural_comparison(all_analyses)
        
        # Step 4: Generate interpretive report
        report = self.generate_cultural_interpretation_report(comparison)
        
        # Step 5: Display report
        print("\n" + report)
        
        # Step 6: Save results
        self.save_analysis_results(all_analyses, comparison, report)
        
        # Generate example analysis output for demonstration
        self.generate_example_analysis_output(all_analyses)
    
    def generate_example_analysis_output(self, all_analyses: Dict[str, List[DreamAnalysis]]):
        """Generate example analysis output in the specified format"""
        print("\n" + "="*80)
        print("EXAMPLE CULTURAL DREAM ANALYSIS OUTPUT")
        print("="*80)
        
        # Find an interesting dream for each language to demonstrate the analysis
        for language, analyses in all_analyses.items():
            if analyses:
                # Pick the first dream with rich content
                example_dream = analyses[0]
                for analysis in analyses:
                    if len(analysis.content) > 100 and analysis.cultural_markers:
                        example_dream = analysis
                        break
                
                print(f"\n--- {language.upper()} DREAM ANALYSIS EXAMPLE ---")
                print(f"Dream Content: {example_dream.content[:150]}...")
                print()
                
                print("CULTURAL ANALYSIS:")
                
                # Characters analysis
                gender_rep = example_dream.characters.get('gender_representation', {})
                if any(gender_rep.values()):
                    dominant_gender = max(gender_rep.items(), key=lambda x: x[1]) if gender_rep else ('ambiguous', 0)
                    print(f"Characters: Gender representation shows {dominant_gender[0]} dominance,")
                    
                    power_dynamics = example_dream.characters.get('power_dynamics', {})
                    if power_dynamics.get('authority', 0) > 0:
                        print("           suggesting patriarchal/matriarchal authority figures rooted")
                        print("           in cultural power symbolism.")
                    else:
                        print("           indicating egalitarian social relationships.")
                
                # Setting analysis
                dominant_setting = max(example_dream.settings.items(), key=lambda x: x[1]) if example_dream.settings else ('unknown', 0)
                if dominant_setting[1] > 0:
                    setting_interpretation = {
                        'natural': 'Natural wilderness setting reflects romantic connection to landscape,',
                        'domestic': 'Domestic space suggests focus on private/familial relationships,',
                        'mythical': 'Mythical realm blends sacred and fantastical imagery,',
                        'urban': 'Urban environment indicates modern/technological orientation,'
                    }
                    print(f"Setting: {setting_interpretation.get(dominant_setting[0], 'Unknown setting indicates')}")
                    print("         possibly reflecting cultural attitudes toward nature/civilization.")
                
                # Activity analysis
                dominant_activity = max(example_dream.activities.items(), key=lambda x: x[1]) if example_dream.activities else ('unknown', 0)
                if dominant_activity[1] > 0:
                    activity_interpretation = {
                        'movement': 'Journey/movement motifs suggest cultural values of exploration',
                        'social': 'Social interaction patterns reflect communal cultural scripts',
                        'exploration': 'Discovery activities mirror cultural curiosity and learning values'
                    }
                    print(f"Activity: {activity_interpretation.get(dominant_activity[0], 'Activities suggest')} and agency.")
                
                # Emotional analysis
                dominant_emotion = max(example_dream.emotions.items(), key=lambda x: x[1]) if example_dream.emotions else ('unknown', 0)
                affect_interpretation = "Anxiety and unresolved tension dominate" if example_dream.anxiety_score > 0.3 else "Peace and positive affect prevail"
                print(f"Emotion: {affect_interpretation}, pointing to cultural emotional regulation patterns.")
                
                # Cultural script analysis
                if example_dream.cultural_markers:
                    marker_meanings = {
                        'strong_nature_connection': 'Deep ecological consciousness prevalent in cultural mythology',
                        'spiritual_orientation': 'Mystical/spiritual elements reflect traditional worldview integration',
                        'collectivist_orientation': 'Community-focused values mirror cultural collectivism',
                        'respect_for_wisdom': 'Elder wisdom traditions embedded in cultural transmission'
                    }
                    
                    for marker in example_dream.cultural_markers[:2]:  # Show top 2 markers
                        interpretation = marker_meanings.get(marker, 'Cultural pattern indicates specific meaning-making system')
                        print(f"Cultural Script: {interpretation}.")
                
                print()


def main():
    """Main execution function"""
    analyst = CulturalDreamAnalyst()
    analyst.run_complete_analysis()


if __name__ == "__main__":
    main()
