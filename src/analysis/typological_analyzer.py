#!/usr/bin/env python3
"""
Typological Linguistic Analysis for Dream Narratives

This module provides a comprehensive analysis framework that explores relationships
between linguistic typological features (WALS) and narrative patterns in dream texts.
The analysis is purely exploratory and data-driven, without theoretical preconceptions.

Main Components:
- WALS feature definitions for 5 languages
- Narrative dimension scoring (LLM-based and heuristic fallback)
- Typological distance calculations
- Correlation analysis and pattern discovery
- Comprehensive visualization and reporting

Usage:
    from src.analysis.typological_analyzer import TypologicalAnalyzer
    
    analyzer = TypologicalAnalyzer(llm_interface=llm_interface)
    results = analyzer.analyze_dreams(dreams_by_language)
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import asyncio
import logging
from datetime import datetime
from collections import defaultdict
import itertools
from scipy.stats import pearsonr, spearmanr
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Import LLM interface
try:
    from ..models.llm_interface import LLMInterface, GenerationConfig
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    LLMInterface = None
    GenerationConfig = None

@dataclass
class WALSFeatures:
    """WALS (World Atlas of Language Structures) features for a language."""
    tense_aspect: str
    alignment: str
    subject_expression: str
    modality: str
    evidentiality: str
    word_order: str
    case_marking: str
    definiteness: str
    gender: str
    number: str
    negation: str
    voice: str

@dataclass
class NarrativeScores:
    """Narrative dimension scores for a dream text."""
    dreamer_agency: float  # 0-1 scale
    other_agents: float    # 0-1 scale
    interaction: float     # 0-1 scale
    emotion: float         # 0-1 scale
    temporal_coherence: float  # 0-1 scale
    cultural_motifs: float # 0-1 scale
    narrative_complexity: float  # 0-1 scale

@dataclass
class TypologicalDistance:
    """Typological distance between two languages."""
    language_pair: Tuple[str, str]
    distance: float
    feature_differences: Dict[str, bool]
    shared_features: int
    total_features: int

@dataclass
class AnalysisResult:
    """Complete analysis result for a dream."""
    dream_id: str
    language: str
    dream_text: str
    narrative_scores: NarrativeScores
    wals_features: WALSFeatures
    analysis_method: str  # 'llm' or 'heuristic'
    timestamp: str

class TypologicalAnalyzer:
    """Main typological linguistic analysis engine."""
    
    def __init__(self, llm_interface: Optional[LLMInterface] = None):
        self.llm_interface = llm_interface
        self.wals_data = self._initialize_wals_data()
        self.narrative_prompt = self._create_narrative_prompt()
        
    def _initialize_wals_data(self) -> Dict[str, WALSFeatures]:
        """Initialize WALS features for all target languages."""
        return {
            'english': WALSFeatures(
                tense_aspect='past_present_future',
                alignment='nominative_accusative',
                subject_expression='obligatory_pronouns',
                modality='modal_verbs',
                evidentiality='no_evidentials',
                word_order='SVO',
                case_marking='minimal_case',
                definiteness='definite_indefinite',
                gender='no_gender',
                number='singular_plural',
                negation='standard_negation',
                voice='active_passive'
            ),
            'basque': WALSFeatures(
                tense_aspect='complex_aspectual',
                alignment='ergative_absolutive',
                subject_expression='pro_drop',
                modality='modal_particles',
                evidentiality='no_evidentials',
                word_order='SOV',
                case_marking='rich_case',
                definiteness='definite_indefinite',
                gender='no_gender',
                number='singular_plural',
                negation='standard_negation',
                voice='active_passive'
            ),
            'serbian': WALSFeatures(
                tense_aspect='rich_aspectual',
                alignment='nominative_accusative',
                subject_expression='pro_drop',
                modality='modal_particles',
                evidentiality='no_evidentials',
                word_order='SVO_flexible',
                case_marking='rich_case',
                definiteness='no_articles',
                gender='masculine_feminine_neuter',
                number='singular_plural',
                negation='standard_negation',
                voice='active_passive'
            ),
            'hebrew': WALSFeatures(
                tense_aspect='triconsonantal_binyan',
                alignment='nominative_accusative',
                subject_expression='pro_drop',
                modality='modal_particles',
                evidentiality='reportative_evidential',
                word_order='SVO',
                case_marking='minimal_case',
                definiteness='definite_indefinite',
                gender='masculine_feminine',
                number='singular_plural_dual',
                negation='standard_negation',
                voice='active_passive_middle'
            ),
            'slovenian': WALSFeatures(
                tense_aspect='rich_aspectual',
                alignment='nominative_accusative',
                subject_expression='pro_drop',
                modality='modal_particles',
                evidentiality='no_evidentials',
                word_order='SVO_flexible',
                case_marking='rich_case',
                definiteness='no_articles',
                gender='masculine_feminine_neuter',
                number='singular_dual_plural',
                negation='standard_negation',
                voice='active_passive'
            )
        }
    
    def _create_narrative_prompt(self) -> str:
        """Create LLM prompt for narrative dimension scoring."""
        return """You are a narrative analyst that scores dream texts on specific dimensions. 
        Your task is to analyze dream narratives and return precise numerical scores.

        Always respond with ONLY a valid JSON object containing exactly these 7 fields:
        - dreamer_agency: float (0.0-1.0)
        - other_agents: float (0.0-1.0) 
        - interaction: float (0.0-1.0)
        - emotion: float (0.0-1.0)
        - temporal_coherence: float (0.0-1.0)
        - cultural_motifs: float (0.0-1.0)
        - narrative_complexity: float (0.0-1.0)

        Do not include any explanations, markdown formatting, or extra text. 
        Return only the JSON object."""
    
    def calculate_typological_distance(self, lang1: str, lang2: str) -> TypologicalDistance:
        """Calculate typological distance between two languages."""
        features1 = self.wals_data[lang1]
        features2 = self.wals_data[lang2]
        
        differences = {}
        shared_count = 0
        total_count = 0
        
        for field in WALSFeatures.__dataclass_fields__:
            val1 = getattr(features1, field)
            val2 = getattr(features2, field)
            differences[field] = val1 != val2
            if val1 == val2:
                shared_count += 1
            total_count += 1
        
        distance = sum(differences.values()) / total_count
        
        return TypologicalDistance(
            language_pair=(lang1, lang2),
            distance=distance,
            feature_differences=differences,
            shared_features=shared_count,
            total_features=total_count
        )
    
    def calculate_all_distances(self, languages: List[str]) -> Dict[Tuple[str, str], TypologicalDistance]:
        """Calculate typological distances for all language pairs."""
        distances = {}
        for lang1, lang2 in itertools.combinations(languages, 2):
            distances[(lang1, lang2)] = self.calculate_typological_distance(lang1, lang2)
        return distances
    
    async def score_narrative_llm(self, dream_text: str, language: str) -> Optional[NarrativeScores]:
        """Score narrative dimensions using LLM."""
        if not self.llm_interface:
            return None
        
        config = GenerationConfig(
            model='gpt-4o',
            temperature=0.1,  # Low temperature for consistent scoring
            max_tokens=300,  # Increased for JSON response
            top_p=0.9
        )
        
        # More explicit user prompt for JSON scoring
        user_prompt = f"""Score this {language} dream text on narrative dimensions (0.0-1.0 scale):

Dream text: {dream_text}

Return ONLY a JSON object with these exact fields:
- dreamer_agency: (0.0-1.0) How much agency/control the dreamer has
- other_agents: (0.0-1.0) Presence of other characters
- interaction: (0.0-1.0) Level of social interaction
- emotion: (0.0-1.0) Emotional intensity
- temporal_coherence: (0.0-1.0) How coherent the timeline is
- cultural_motifs: (0.0-1.0) Culture-specific elements
- narrative_complexity: (0.0-1.0) Story structure complexity

Example format:
{{"dreamer_agency": 0.7, "other_agents": 0.3, "interaction": 0.1, "emotion": 0.5, "temporal_coherence": 0.8, "cultural_motifs": 0.2, "narrative_complexity": 0.6}}"""
        
        try:
            response = await self.llm_interface.generate_dream(
                user_prompt, 
                config, 
                self.narrative_prompt
            )
            
            # Clean the response - remove any markdown formatting
            response_clean = response.strip()
            if response_clean.startswith('```json'):
                response_clean = response_clean[7:]
            if response_clean.endswith('```'):
                response_clean = response_clean[:-3]
            response_clean = response_clean.strip()
            
            # Log the actual response for debugging
            logging.debug(f"LLM response for {language}: {response_clean[:200]}...")
            
            # Check for empty response
            if not response_clean:
                logging.warning(f"Empty response from LLM for {language} dream scoring")
                return None
            
            # Parse JSON response
            try:
                scores_data = json.loads(response_clean)
            except json.JSONDecodeError as e:
                logging.warning(f"JSON parsing failed for {language}: {e}. Response: {response_clean[:100]}...")
                # Try to extract JSON from response if it's embedded in text
                import re
                scores_data = None
                
                # First, try to find a complete JSON object
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_clean)
                if json_match:
                    try:
                        scores_data = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        pass
                
                # If that didn't work, try to find JSON with required fields
                if scores_data is None:
                    json_pattern = r'\{[^}]*(?:dreamer_agency|other_agents|interaction|emotion|temporal_coherence|cultural_motifs|narrative_complexity)[^}]*\}'
                    json_match = re.search(json_pattern, response_clean)
                    if json_match:
                        try:
                            scores_data = json.loads(json_match.group())
                        except json.JSONDecodeError:
                            pass
                
                # If we still don't have valid JSON, give up
                if scores_data is None:
                    logging.warning(f"Could not extract valid JSON from {language} response")
                    return None
            
            # Validate all required fields are present
            required_fields = ['dreamer_agency', 'other_agents', 'interaction', 'emotion', 
                             'temporal_coherence', 'cultural_motifs', 'narrative_complexity']
            
            if not all(field in scores_data for field in required_fields):
                missing_fields = [field for field in required_fields if field not in scores_data]
                logging.warning(f"Missing fields in LLM response for {language}: {missing_fields}")
                return None
            
            # Validate score ranges (0.0-1.0)
            for field, value in scores_data.items():
                if field in required_fields:
                    if not isinstance(value, (int, float)) or value < 0 or value > 1:
                        logging.warning(f"Invalid score for {field} in {language}: {value}")
                        return None
            
            # Create NarrativeScores with validated data
            return NarrativeScores(
                dreamer_agency=float(scores_data['dreamer_agency']),
                other_agents=float(scores_data['other_agents']),
                interaction=float(scores_data['interaction']),
                emotion=float(scores_data['emotion']),
                temporal_coherence=float(scores_data['temporal_coherence']),
                cultural_motifs=float(scores_data['cultural_motifs']),
                narrative_complexity=float(scores_data['narrative_complexity'])
            )
                
        except Exception as e:
            logging.error(f"Error in LLM narrative scoring for {language}: {e}")
            return None
    
    def score_narrative_heuristic(self, dream_text: str, language: str) -> NarrativeScores:
        """Fallback heuristic scoring when LLM is unavailable."""
        # Simple heuristic based on text analysis
        text_lower = dream_text.lower()
        words = text_lower.split()
        word_count = len(words)
        
        # Dreamer agency indicators
        agency_words = ['i', 'me', 'my', 'myself', 'decided', 'chose', 'ran', 'walked', 'said', 'took']
        agency_score = min(1.0, sum(1 for word in agency_words if word in text_lower) / 10)
        
        # Other agents indicators
        other_words = ['he', 'she', 'they', 'people', 'person', 'man', 'woman', 'friend', 'family']
        other_score = min(1.0, sum(1 for word in other_words if word in text_lower) / 8)
        
        # Interaction indicators
        interaction_words = ['talked', 'spoke', 'conversation', 'together', 'meeting', 'group']
        interaction_score = min(1.0, sum(1 for word in interaction_words if word in text_lower) / 6)
        
        # Emotion indicators
        emotion_words = ['happy', 'sad', 'angry', 'scared', 'afraid', 'excited', 'worried', 'love']
        emotion_score = min(1.0, sum(1 for word in emotion_words if word in text_lower) / 8)
        
        # Temporal coherence (inverse of temporal jump words)
        temporal_words = ['suddenly', 'then', 'next', 'after', 'before', 'meanwhile']
        temporal_score = max(0.0, 1.0 - (sum(1 for word in temporal_words if word in text_lower) / 10))
        
        # Cultural motifs (placeholder - would need language-specific cultural markers)
        cultural_score = 0.3  # Default moderate score
        
        # Narrative complexity (based on sentence structure and length)
        complexity_score = min(1.0, word_count / 500)  # Longer texts assumed more complex
        
        return NarrativeScores(
            dreamer_agency=agency_score,
            other_agents=other_score,
            interaction=interaction_score,
            emotion=emotion_score,
            temporal_coherence=temporal_score,
            cultural_motifs=cultural_score,
            narrative_complexity=complexity_score
        )
    
    async def analyze_dream(self, dream_id: str, dream_text: str, language: str) -> AnalysisResult:
        """Analyze a single dream for typological features."""
        # Try LLM scoring first, fallback to heuristic
        narrative_scores = await self.score_narrative_llm(dream_text, language)
        method = 'llm'
        
        if narrative_scores is None:
            narrative_scores = self.score_narrative_heuristic(dream_text, language)
            method = 'heuristic'
        
        return AnalysisResult(
            dream_id=dream_id,
            language=language,
            dream_text=dream_text[:200] + "..." if len(dream_text) > 200 else dream_text,
            narrative_scores=narrative_scores,
            wals_features=self.wals_data[language],
            analysis_method=method,
            timestamp=datetime.now().isoformat()
        )
    
    async def analyze_dreams(self, dreams_by_language: Dict[str, List[Dict]], 
                           max_dreams_per_language: int = 50) -> Dict[str, Any]:
        """Analyze dreams from all languages and compute correlations."""
        results = []
        total_dreams = sum(min(len(dreams), max_dreams_per_language) 
                          for dreams in dreams_by_language.values())
        
        analyzed_count = 0
        
        # Analyze each dream
        for language, dreams in dreams_by_language.items():
            if language not in self.wals_data:
                logging.warning(f"No WALS data for language: {language}")
                continue
                
            for dream in dreams[:max_dreams_per_language]:
                dream_id = dream.get('dream_id', f"{language}_{analyzed_count}")
                dream_text = dream.get('dream_text', '')
                
                if dream_text:
                    result = await self.analyze_dream(dream_id, dream_text, language)
                    results.append(result)
                    analyzed_count += 1
        
        # Calculate correlations and patterns
        correlations = self._calculate_correlations(results)
        language_distances = self.calculate_all_distances(list(dreams_by_language.keys()))
        clusters = self._perform_clustering(results)
        
        return {
            'results': results,
            'total_analyzed': analyzed_count,
            'correlations': correlations,
            'language_distances': language_distances,
            'clusters': clusters,
            'summary_stats': self._generate_summary_stats(results),
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_correlations(self, results: List[AnalysisResult]) -> Dict[str, Any]:
        """Calculate correlations between WALS features and narrative dimensions."""
        # Initialize empty correlations structure
        correlations = {
            'language_narrative_means': {},
            'feature_patterns': {},
            'cross_linguistic_patterns': {}
        }
        
        # Handle empty results
        if not results:
            return correlations
            
        # Convert to DataFrame for easier analysis
        data = []
        for result in results:
            row = {
                'language': result.language,
                'analysis_method': result.analysis_method
            }
            # Add WALS features
            for field in WALSFeatures.__dataclass_fields__:
                row[f'wals_{field}'] = getattr(result.wals_features, field)
            # Add narrative scores
            for field in NarrativeScores.__dataclass_fields__:
                row[f'narrative_{field}'] = getattr(result.narrative_scores, field)
            data.append(row)
        
        # Handle case where data is empty
        if not data:
            return correlations
            
        df = pd.DataFrame(data)
        
        # Language-level narrative means
        for language in df['language'].unique():
            lang_data = df[df['language'] == language]
            means = {}
            for field in NarrativeScores.__dataclass_fields__:
                means[field] = lang_data[f'narrative_{field}'].mean()
            correlations['language_narrative_means'][language] = means
        
        # Feature patterns (exploratory analysis)
        correlations['feature_patterns'] = self._discover_feature_patterns(df)
        
        return correlations
    
    def _discover_feature_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Discover patterns between WALS features and narrative dimensions."""
        patterns = {}
        
        # Group by language and calculate means
        lang_means = df.groupby('language').agg({
            col: 'mean' for col in df.columns if col.startswith('narrative_')
        }).reset_index()
        
        # Add WALS features for each language
        for idx, row in lang_means.iterrows():
            language = row['language']
            if language in self.wals_data:
                wals_features = self.wals_data[language]
                for field in WALSFeatures.__dataclass_fields__:
                    lang_means.loc[idx, f'wals_{field}'] = getattr(wals_features, field)
        
        # Find correlations between WALS features and narrative dimensions
        wals_cols = [col for col in lang_means.columns if col.startswith('wals_')]
        narrative_cols = [col for col in lang_means.columns if col.startswith('narrative_')]
        
        patterns['correlations'] = {}
        patterns['language_profiles'] = {}
        
        # Language profiles
        for idx, row in lang_means.iterrows():
            language = row['language']
            profile = {
                'wals_features': {col.replace('wals_', ''): row[col] for col in wals_cols},
                'narrative_means': {col.replace('narrative_', ''): row[col] for col in narrative_cols}
            }
            patterns['language_profiles'][language] = profile
        
        return patterns
    
    def _perform_clustering(self, results: List[AnalysisResult]) -> Dict[str, Any]:
        """Perform clustering analysis on languages based on narrative patterns."""
        # Handle empty results
        if not results:
            return {
                'languages': [],
                'centroids': {},
                'linkage_matrix': None,
                'distance_matrix': None,
                'clustering_status': 'no_data',
                'status_message': 'No dreams available for clustering'
            }
            
        # Group by language and calculate means
        lang_means = {}
        for result in results:
            if result.language not in lang_means:
                lang_means[result.language] = []
            
            scores = [getattr(result.narrative_scores, field) 
                     for field in NarrativeScores.__dataclass_fields__]
            lang_means[result.language].append(scores)
        
        # Calculate language centroids
        centroids = {}
        for language, scores_list in lang_means.items():
            if scores_list:  # Make sure we have scores
                centroids[language] = np.mean(scores_list, axis=0)
        
        # Perform hierarchical clustering
        languages = list(centroids.keys())
        
        if len(languages) < 2:
            return {
                'languages': languages,
                'centroids': centroids,
                'linkage_matrix': None,
                'distance_matrix': None,
                'clustering_status': 'insufficient_languages',
                'status_message': f'Need at least 2 languages for clustering (found {len(languages)})'
            }
        
        try:
            centroid_matrix = np.array([centroids[lang] for lang in languages])
            
            # Ensure we have valid data for clustering
            if centroid_matrix.size == 0:
                raise ValueError("Empty centroid matrix")
            
            if np.any(np.isnan(centroid_matrix)):
                raise ValueError("NaN values in centroid data")
            
            # Check for identical centroids (would cause clustering issues)
            unique_centroids = np.unique(centroid_matrix, axis=0)
            if len(unique_centroids) < 2:
                raise ValueError("All languages have identical narrative patterns")
            
            # Check for near-identical centroids (within tolerance)
            from scipy.spatial.distance import pdist
            pairwise_distances = pdist(centroid_matrix)
            min_distance = np.min(pairwise_distances) if len(pairwise_distances) > 0 else 0
            
            if min_distance < 1e-10:  # Very small tolerance
                raise ValueError("Languages have nearly identical narrative patterns")
            
            linkage_matrix = linkage(centroid_matrix, method='ward')
            
            # Validate linkage matrix dimensions
            if linkage_matrix.shape[0] != len(languages) - 1 or linkage_matrix.shape[1] != 4:
                raise ValueError(f"Invalid linkage matrix shape: {linkage_matrix.shape}")
            
            # Check for valid linkage values
            if np.any(np.isnan(linkage_matrix)) or np.any(np.isinf(linkage_matrix)):
                raise ValueError("Linkage matrix contains invalid values")
            
            return {
                'languages': languages,
                'centroids': centroids,
                'linkage_matrix': linkage_matrix.tolist(),
                'distance_matrix': pdist(centroid_matrix).tolist(),
                'clustering_status': 'success',
                'status_message': f'Successfully clustered {len(languages)} languages'
            }
            
        except Exception as e:
            error_msg = str(e)
            logging.warning(f"Clustering failed: {error_msg}")
            
            # Provide more specific status based on error type
            if "identical" in error_msg.lower():
                status = 'identical_patterns'
                message = f'Languages have identical narrative patterns - try analyzing more diverse dreams'
            elif "near" in error_msg.lower():
                status = 'similar_patterns'  
                message = f'Languages have very similar narrative patterns - clustering not meaningful'
            elif "empty" in error_msg.lower() or "nan" in error_msg.lower():
                status = 'invalid_data'
                message = f'Invalid clustering data: {error_msg}'
            else:
                status = 'failed'
                message = f'Clustering failed: {error_msg}'
            
            return {
                'languages': languages,
                'centroids': centroids,
                'linkage_matrix': None,
                'distance_matrix': None,
                'clustering_status': status,
                'status_message': message
            }
    
    def _generate_summary_stats(self, results: List[AnalysisResult]) -> Dict[str, Any]:
        """Generate summary statistics for the analysis."""
        total_dreams = len(results)
        languages = list(set(r.language for r in results)) if results else []
        
        stats = {
            'total_dreams': total_dreams,
            'languages': languages,
            'language_counts': {lang: sum(1 for r in results if r.language == lang) 
                              for lang in languages},
            'analysis_methods': {
                'llm': sum(1 for r in results if r.analysis_method == 'llm'),
                'heuristic': sum(1 for r in results if r.analysis_method == 'heuristic')
            },
            'narrative_dimension_means': {},
            'wals_feature_distribution': {}
        }
        
        # Only calculate stats if we have results
        if results:
            # Calculate overall narrative dimension means
            for field in NarrativeScores.__dataclass_fields__:
                scores = [getattr(r.narrative_scores, field) for r in results]
                if scores:  # Make sure we have scores
                    stats['narrative_dimension_means'][field] = {
                        'mean': np.mean(scores),
                        'std': np.std(scores),
                        'min': np.min(scores),
                        'max': np.max(scores)
                    }
            
            # WALS feature distribution
            for field in WALSFeatures.__dataclass_fields__:
                values = [getattr(r.wals_features, field) for r in results]
                if values:  # Make sure we have values
                    stats['wals_feature_distribution'][field] = {
                        'values': list(set(values)),
                        'counts': {val: values.count(val) for val in set(values)}
                    }
        
        return stats
    
    def create_visualizations(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive visualizations for the analysis."""
        visualizations = {}
        
        # 1. Language-Narrative Correlation Heatmap
        if 'correlations' in analysis_results:
            correlations = analysis_results['correlations']
            if 'language_narrative_means' in correlations:
                lang_means = correlations['language_narrative_means']
                
                # Create DataFrame for heatmap
                df_heatmap = pd.DataFrame(lang_means).T
                
                fig_heatmap = px.imshow(
                    df_heatmap.values,
                    x=df_heatmap.columns,
                    y=df_heatmap.index,
                    color_continuous_scale='RdBu_r',
                    title='Narrative Dimension Means by Language',
                    labels={'x': 'Narrative Dimension', 'y': 'Language', 'color': 'Mean Score'}
                )
                visualizations['narrative_heatmap'] = fig_heatmap
        
        # 2. Typological Distance Matrix
        if 'language_distances' in analysis_results:
            distances = analysis_results['language_distances']
            languages = list(set([pair[0] for pair in distances.keys()] + 
                               [pair[1] for pair in distances.keys()]))
            
            # Create distance matrix
            dist_matrix = np.zeros((len(languages), len(languages)))
            for i, lang1 in enumerate(languages):
                for j, lang2 in enumerate(languages):
                    if i != j:
                        pair = (lang1, lang2) if (lang1, lang2) in distances else (lang2, lang1)
                        if pair in distances:
                            dist_matrix[i, j] = distances[pair].distance
            
            fig_dist = px.imshow(
                dist_matrix,
                x=languages,
                y=languages,
                color_continuous_scale='Viridis',
                title='Typological Distance Matrix (WALS Features)',
                labels={'x': 'Language', 'y': 'Language', 'color': 'Distance'}
            )
            visualizations['distance_matrix'] = fig_dist
        
        # 3. Language Clustering Dendrogram
        if 'clusters' in analysis_results:
            clusters = analysis_results['clusters']
            clustering_status = clusters.get('clustering_status', 'unknown')
            status_message = clusters.get('status_message', 'Clustering status unknown')
            
            if clustering_status == 'success' and clusters['linkage_matrix'] is not None:
                try:
                    linkage_matrix = np.array(clusters['linkage_matrix'])
                    languages = clusters['languages']
                    
                    # Comprehensive validation for plotly dendrogram requirements
                    if len(languages) < 2:
                        raise ValueError("Need at least 2 languages for dendrogram")
                    
                    # Check linkage matrix dimensions and content
                    expected_rows = len(languages) - 1
                    if linkage_matrix.shape[0] != expected_rows:
                        raise ValueError(f"Linkage matrix has {linkage_matrix.shape[0]} rows, expected {expected_rows}")
                    
                    if linkage_matrix.shape[1] != 4:
                        raise ValueError(f"Linkage matrix has {linkage_matrix.shape[1]} columns, expected 4")
                    
                    # Check for valid linkage values
                    if np.any(np.isnan(linkage_matrix)) or np.any(np.isinf(linkage_matrix)):
                        raise ValueError("Linkage matrix contains invalid values (NaN or Inf)")
                    
                    # Check that indices in linkage matrix are valid
                    max_index = linkage_matrix[:, :2].max()
                    if max_index >= len(languages) + len(linkage_matrix):
                        raise ValueError("Linkage matrix contains invalid cluster indices")
                    
                    # Additional check: ensure we have enough unique data points
                    if 'centroids' in clusters:
                        centroids = clusters['centroids']
                        unique_centroids = np.unique(np.array([centroids[lang] for lang in languages]), axis=0)
                        if len(unique_centroids) < 2:
                            raise ValueError("All languages have identical narrative patterns")
                    
                    # Suppress specific scipy warnings during dendrogram creation
                    import warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message=".*Dimensions of Z and labels.*")
                        warnings.filterwarnings("ignore", message=".*observations cannot be determined.*")
                        
                        fig_dendro = ff.create_dendrogram(
                            linkage_matrix,
                            labels=languages
                        )
                        fig_dendro.update_layout(
                            title='Language Clustering Based on Narrative Patterns',
                            annotations=[
                                dict(
                                    x=0.5, y=1.05, xref='paper', yref='paper',
                                    text=f"✅ {status_message}",
                                    showarrow=False, font=dict(size=12, color='green')
                                )
                            ]
                        )
                        visualizations['dendrogram'] = fig_dendro
                        
                except Exception as e:
                    # Create informative bar chart with detailed error
                    fig_dendro = go.Figure()
                    if clusters['languages']:
                        fig_dendro.add_trace(go.Bar(
                            x=clusters['languages'],
                            y=[1] * len(clusters['languages']),
                            name='Languages',
                            marker_color='lightcoral'
                        ))
                    
                    # Determine error type for better messaging
                    error_msg = str(e)
                    if "identical narrative patterns" in error_msg:
                        icon = "🔄"
                        user_msg = "Languages have identical narrative patterns"
                    elif "invalid cluster indices" in error_msg or "Dimensions" in error_msg:
                        icon = "🔧"
                        user_msg = "Clustering calculation issue - try with more dreams"
                    elif "Need at least 2 languages" in error_msg:
                        icon = "🔢"
                        user_msg = "Need more languages for clustering"
                    else:
                        icon = "⚠️"
                        user_msg = f"Visualization error: {error_msg}"
                    
                    fig_dendro.update_layout(
                        title='Language Clustering (Visualization Issue)',
                        xaxis_title='Language',
                        yaxis_title='Count',
                        annotations=[
                            dict(
                                x=0.5, y=1.05, xref='paper', yref='paper',
                                text=f"{icon} {user_msg}",
                                showarrow=False, font=dict(size=12, color='orange')
                            )
                        ]
                    )
                    visualizations['dendrogram'] = fig_dendro
            else:
                # Create informative bar chart with status message
                fig_dendro = go.Figure()
                if clusters['languages']:
                    color = 'lightblue' if clustering_status == 'insufficient_languages' else 'lightcoral'
                    fig_dendro.add_trace(go.Bar(
                        x=clusters['languages'],
                        y=[1] * len(clusters['languages']),
                        name='Languages',
                        marker_color=color
                    ))
                    
                    title_suffix = {
                        'no_data': '(No Data)',
                        'insufficient_languages': '(Need More Languages)',
                        'failed': '(Analysis Failed)',
                        'identical_patterns': '(Identical Patterns)',
                        'similar_patterns': '(Very Similar Patterns)',
                        'invalid_data': '(Invalid Data)',
                        'unknown': '(Status Unknown)'
                    }.get(clustering_status, '(Unavailable)')
                    
                    icon = {
                        'no_data': '📊',
                        'insufficient_languages': '🔢',
                        'failed': '❌',
                        'identical_patterns': '🔄',
                        'similar_patterns': '↔️',
                        'invalid_data': '⚠️',
                        'unknown': '❓'
                    }.get(clustering_status, '⚠️')
                    
                    fig_dendro.update_layout(
                        title=f'Language Clustering {title_suffix}',
                        xaxis_title='Language',
                        yaxis_title='Count',
                        annotations=[
                            dict(
                                x=0.5, y=1.05, xref='paper', yref='paper',
                                text=f"{icon} {status_message}",
                                showarrow=False, font=dict(size=12, color='steelblue')
                            )
                        ]
                    )
                else:
                    fig_dendro.update_layout(
                        title='Language Clustering (No Data)',
                        annotations=[
                            dict(
                                x=0.5, y=0.5, xref='paper', yref='paper',
                                text="📊 No languages available for analysis",
                                showarrow=False, font=dict(size=14, color='gray')
                            )
                        ]
                    )
                visualizations['dendrogram'] = fig_dendro
        
        # 4. Radar Chart for Language Profiles
        if 'correlations' in analysis_results:
            correlations = analysis_results['correlations']
            if 'feature_patterns' in correlations and 'language_profiles' in correlations['feature_patterns']:
                profiles = correlations['feature_patterns']['language_profiles']
                
                fig_radar = go.Figure()
                
                dimensions = list(NarrativeScores.__dataclass_fields__.keys())
                
                for language, profile in profiles.items():
                    try:
                        values = [profile['narrative_means'].get(dim, 0.0) for dim in dimensions]
                        values.append(values[0])  # Close the radar chart
                        
                        fig_radar.add_trace(go.Scatterpolar(
                            r=values,
                            theta=dimensions + [dimensions[0]],
                            fill='toself',
                            name=language.title()
                        ))
                    except Exception as e:
                        print(f"Warning: Could not add radar trace for {language}: {e}")
                        continue
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1])
                    ),
                    title='Language Profiles: Narrative Dimensions',
                    showlegend=True
                )
                visualizations['radar_chart'] = fig_radar
        
        return visualizations
    
    def export_results(self, analysis_results: Dict[str, Any], output_dir: Path) -> Dict[str, Path]:
        """Export analysis results in multiple formats."""
        output_dir.mkdir(parents=True, exist_ok=True)
        exported_files = {}
        
        # 1. Complete results as JSON
        json_file = output_dir / 'typological_analysis_results.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            # Convert results to serializable format
            serializable_results = self._make_serializable(analysis_results)
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        exported_files['json'] = json_file
        
        # 2. Language-Narrative means as CSV
        if 'correlations' in analysis_results:
            correlations = analysis_results['correlations']
            if 'language_narrative_means' in correlations:
                lang_means = correlations['language_narrative_means']
                df_means = pd.DataFrame(lang_means).T
                csv_file = output_dir / 'language_narrative_means.csv'
                df_means.to_csv(csv_file)
                exported_files['narrative_means_csv'] = csv_file
        
        # 3. WALS features as CSV
        wals_df = pd.DataFrame([
            {'language': lang, **asdict(features)}
            for lang, features in self.wals_data.items()
        ])
        wals_file = output_dir / 'wals_features.csv'
        wals_df.to_csv(wals_file, index=False)
        exported_files['wals_csv'] = wals_file
        
        # 4. Typological distances as CSV
        if 'language_distances' in analysis_results:
            distances = analysis_results['language_distances']
            dist_data = []
            for pair, dist_obj in distances.items():
                dist_data.append({
                    'language_1': pair[0],
                    'language_2': pair[1],
                    'distance': dist_obj.distance,
                    'shared_features': dist_obj.shared_features,
                    'total_features': dist_obj.total_features
                })
            dist_df = pd.DataFrame(dist_data)
            dist_file = output_dir / 'typological_distances.csv'
            dist_df.to_csv(dist_file, index=False)
            exported_files['distances_csv'] = dist_file
        
        # 5. Summary report as markdown
        report_file = output_dir / 'typological_analysis_report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(self._generate_markdown_report(analysis_results))
        exported_files['report_md'] = report_file
        
        return exported_files
    
    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            # Handle dictionary keys that are tuples
            serializable_dict = {}
            for k, v in obj.items():
                if isinstance(k, tuple):
                    # Convert tuple keys to strings
                    serializable_key = '-'.join(str(item) for item in k)
                    serializable_dict[serializable_key] = self._make_serializable(v)
                else:
                    serializable_dict[k] = self._make_serializable(v)
            return serializable_dict
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (WALSFeatures, NarrativeScores, AnalysisResult, TypologicalDistance)):
            return asdict(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    def _generate_markdown_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate a comprehensive markdown report."""
        report = f"""# Typological Linguistic Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report presents a comprehensive analysis of the relationship between linguistic typological features (WALS) and narrative patterns in dream texts across {len(analysis_results.get('summary_stats', {}).get('languages', []))} languages.

### Key Findings

- **Total Dreams Analyzed:** {analysis_results.get('total_analyzed', 0)}
- **Languages:** {', '.join(analysis_results.get('summary_stats', {}).get('languages', []))}
- **Analysis Methods:** {analysis_results.get('summary_stats', {}).get('analysis_methods', {})}

## Methodology

### WALS Features Analyzed
- **Tense/Aspect:** Temporal marking systems
- **Alignment:** Grammatical alignment patterns
- **Subject Expression:** Pronoun dropping patterns
- **Modality:** Modal expression systems
- **Evidentiality:** Evidential marking
- **Word Order:** Basic constituent order
- **Case Marking:** Case system complexity
- **Definiteness:** Article systems
- **Gender:** Gender marking systems
- **Number:** Number marking systems
- **Negation:** Negation strategies
- **Voice:** Voice systems

### Narrative Dimensions Scored
- **Dreamer Agency:** Control and agency of the dreamer
- **Other Agents:** Presence of other characters
- **Interaction:** Social interaction levels
- **Emotion:** Emotional intensity
- **Temporal Coherence:** Timeline coherence
- **Cultural Motifs:** Culture-specific elements
- **Narrative Complexity:** Story structure complexity

## Results

### Language Distance Analysis
"""
        
        if 'language_distances' in analysis_results:
            distances = analysis_results['language_distances']
            report += "\n**Typological Distances (WALS Features):**\n"
            for pair, dist_obj in distances.items():
                report += f"- {pair[0].title()} ↔ {pair[1].title()}: {dist_obj.distance:.3f} " \
                         f"({dist_obj.shared_features}/{dist_obj.total_features} shared features)\n"
        
        if 'summary_stats' in analysis_results:
            stats = analysis_results['summary_stats']
            report += "\n### Narrative Dimension Statistics\n"
            for dim, values in stats.get('narrative_dimension_means', {}).items():
                report += f"- **{dim.replace('_', ' ').title()}:** " \
                         f"Mean={values['mean']:.3f}, SD={values['std']:.3f}, " \
                         f"Range=[{values['min']:.3f}, {values['max']:.3f}]\n"
        
        report += """
## Interpretation

This analysis takes a purely exploratory approach, discovering patterns in the data without 
theoretical preconceptions. The results reveal empirical relationships between linguistic 
structure and narrative patterns that warrant further investigation.

### Statistical Approach

All correlations and patterns are reported as observed in the data. The analysis does not 
make causal claims but presents associations that may inform future research directions.

## Technical Notes

- Analysis conducted using both LLM-based and heuristic narrative scoring
- WALS features based on established typological categories
- Clustering performed using hierarchical methods
- All numeric results represent observed patterns in the sample data

---

*Generated by Typological Linguistic Analysis Engine*
"""
        
        return report 