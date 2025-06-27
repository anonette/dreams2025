"""
Simple LLM-based discourse analyzer for dream narratives.
Focus: Thematic and discourse analysis using structured prompts.
"""

import json
from typing import List, Dict, Any


class LLMDiscourseAnalyzer:
    """Simple LLM-based analyzer for dream discourse and thematic analysis."""
    
    def __init__(self, llm_interface, model_name: str = 'gpt-4o'):
        """Initialize with LLM interface."""
        self.llm = llm_interface
        self.model = model_name
    
    def analyze_dream_themes(self, dreams: List[str], language: str = 'english') -> Dict[str, Any]:
        """Analyze thematic content of dreams using LLM."""
        # Combine dreams for analysis
        dreams_text = "\n---\n".join(dreams[:10])  # Limit to 10 dreams for token management
        
        prompt = f"""You are a dream researcher analyzing {language} dream narratives for thematic content.

DREAMS TO ANALYZE:
{dreams_text}

Please identify the main themes, motives, and symbolic elements in these dreams. Focus on:
1. Recurring themes and motives
2. Emotional content
3. Symbolic elements
4. Cultural or linguistic patterns (if relevant for {language})

Provide your analysis in this JSON format:
{{
    "main_themes": [
        {{
            "theme": "theme_name",
            "frequency": "number_or_description",
            "significance": "brief_explanation"
        }}
    ],
    "dream_motives": [
        {{
            "motive": "motive_name",
            "examples": ["example1", "example2"],
            "psychological_meaning": "brief_interpretation"
        }}
    ],
    "emotional_patterns": [
        {{
            "emotion": "emotion_name",
            "intensity": "low/medium/high",
            "context": "when_it_appears"
        }}
    ],
    "symbolic_elements": [
        {{
            "symbol": "symbol_name",
            "frequency": "how_often",
            "interpretation": "possible_meaning"
        }}
    ],
    "cultural_markers": ["marker1", "marker2"],
    "overall_analysis": "brief_summary_of_findings"
}}

Return only valid JSON."""

        try:
            response = self.llm.generate_text(prompt, self.model)
            # Try to parse JSON from response
            if '```json' in response:
                json_text = response.split('```json')[1].split('```')[0].strip()
            else:
                json_text = response.strip()
            
            return json.loads(json_text)
        except Exception as e:
            return {
                "error": f"Analysis failed: {str(e)}",
                "main_themes": [],
                "dream_motives": [],
                "emotional_patterns": [],
                "symbolic_elements": [],
                "cultural_markers": [],
                "overall_analysis": "Analysis could not be completed"
            }
    
    def analyze_discourse_patterns(self, dreams: List[str], language: str = 'english') -> Dict[str, Any]:
        """Analyze discourse patterns in dream narratives."""
        dreams_text = "\n---\n".join(dreams[:8])  # Limit for token management
        
        prompt = f"""You are a discourse analyst studying {language} dream narratives.

DREAM NARRATIVES:
{dreams_text}

Analyze the discourse patterns in these dreams. Focus on:
1. Narrative structure and flow
2. Temporal patterns (past/present/future references)
3. Agency and perspective (who acts, who observes)
4. Coherence and fragmentation
5. Language-specific discourse markers

Provide analysis in JSON format:
{{
    "narrative_structure": {{
        "coherence_level": "high/medium/low",
        "typical_flow": "description_of_common_pattern",
        "fragmentation": "degree_and_type"
    }},
    "temporal_patterns": {{
        "dominant_tense": "past/present/future/mixed",
        "time_references": ["type1", "type2"],
        "temporal_coherence": "description"
    }},
    "agency_patterns": {{
        "dreamer_agency": "high/medium/low",
        "passive_experiences": "frequency_and_type",
        "perspective": "first_person/third_person/mixed"
    }},
    "discourse_markers": [
        {{
            "marker": "linguistic_marker",
            "function": "what_it_indicates",
            "frequency": "how_often"
        }}
    ],
    "language_specific": [
        {{
            "feature": "language_specific_feature",
            "significance": "what_it_reveals"
        }}
    ],
    "overall_discourse": "summary_of_discourse_characteristics"
}}

Return only valid JSON."""

        try:
            response = self.llm.generate_text(prompt, self.model)
            if '```json' in response:
                json_text = response.split('```json')[1].split('```')[0].strip()
            else:
                json_text = response.strip()
            
            return json.loads(json_text)
        except Exception as e:
            return {
                "error": f"Discourse analysis failed: {str(e)}",
                "narrative_structure": {},
                "temporal_patterns": {},
                "agency_patterns": {},
                "discourse_markers": [],
                "language_specific": [],
                "overall_discourse": "Analysis could not be completed"
            }
    
    def cluster_by_discourse(self, dreams: List[str], n_clusters: int = 5) -> Dict[str, Any]:
        """Cluster dreams based on discourse analysis."""
        # Limit dreams for analysis
        sample_dreams = dreams[:15] if len(dreams) > 15 else dreams
        
        dreams_with_ids = []
        for i, dream in enumerate(sample_dreams):
            dreams_with_ids.append(f"Dream {i+1}: {dream}")
        
        dreams_text = "\n\n".join(dreams_with_ids)
        
        prompt = f"""You are a researcher clustering dream narratives based on discourse analysis.

DREAMS TO CLUSTER:
{dreams_text}

Analyze these dreams and group them into {n_clusters} clusters based on discourse patterns, narrative structure, and thematic content.

Consider:
1. Narrative coherence and structure
2. Temporal flow and agency patterns
3. Thematic similarity
4. Emotional tone and discourse markers

Provide clustering in JSON format:
{{
    "clusters": [
        {{
            "cluster_id": 1,
            "cluster_theme": "descriptive_name",
            "discourse_pattern": "main_discourse_characteristic",
            "dream_ids": [1, 5, 8],
            "key_features": ["feature1", "feature2"],
            "sample_analysis": "brief_explanation_of_this_cluster"
        }}
    ],
    "clustering_rationale": "explanation_of_clustering_approach",
    "discourse_insights": "what_the_clustering_reveals_about_discourse"
}}

Return only valid JSON."""

        try:
            response = self.llm.generate_text(prompt, self.model)
            if '```json' in response:
                json_text = response.split('```json')[1].split('```')[0].strip()
            else:
                json_text = response.strip()
            
            result = json.loads(json_text)
            
            # Add actual dreams to clusters
            for cluster in result.get('clusters', []):
                dream_ids = cluster.get('dream_ids', [])
                cluster['dreams'] = [
                    sample_dreams[i-1] for i in dream_ids 
                    if i-1 < len(sample_dreams)
                ]
            
            return result
        except Exception as e:
            return {
                "error": f"Clustering failed: {str(e)}",
                "clusters": [],
                "clustering_rationale": "Analysis could not be completed",
                "discourse_insights": "Analysis could not be completed"
            }
    
    def compare_cross_linguistic(self, dreams_by_language: Dict[str, List[str]]) -> Dict[str, Any]:
        """Compare discourse patterns across languages."""
        # Sample dreams from each language
        sample_data = {}
        for lang, dreams in dreams_by_language.items():
            sample_data[lang] = dreams[:5] if len(dreams) > 5 else dreams
        
        # Format for prompt
        comparison_text = ""
        for lang, dreams in sample_data.items():
            comparison_text += f"\n{lang.upper()} DREAMS:\n"
            for i, dream in enumerate(dreams):
                comparison_text += f"{i+1}. {dream}\n"
        
        prompt = f"""You are a cross-linguistic researcher comparing dream discourse across languages.

DREAM SAMPLES BY LANGUAGE:
{comparison_text}

Compare the discourse patterns, themes, and cultural elements across these languages.

Analyze:
1. Common themes and motives across languages
2. Language-specific discourse patterns
3. Cultural differences in dream content
4. Narrative structure differences
5. Emotional expression patterns

Provide analysis in JSON format:
{{
    "common_themes": [
        {{
            "theme": "universal_theme",
            "languages": ["lang1", "lang2"],
            "significance": "why_its_universal"
        }}
    ],
    "language_specific": {{
        "english": ["specific_feature1", "specific_feature2"],
        "basque": ["specific_feature1", "specific_feature2"],
        "serbian": ["specific_feature1", "specific_feature2"],
        "hebrew": ["specific_feature1", "specific_feature2"],
        "slovenian": ["specific_feature1", "specific_feature2"]
    }},
    "discourse_differences": [
        {{
            "language": "language_name",
            "discourse_pattern": "unique_discourse_characteristic",
            "cultural_significance": "what_it_reveals"
        }}
    ],
    "cultural_insights": [
        {{
            "culture": "culture_name",
            "insight": "cultural_finding",
            "evidence": "supporting_examples"
        }}
    ],
    "overall_comparison": "summary_of_cross_linguistic_findings"
}}

Return only valid JSON."""

        try:
            response = self.llm.generate_text(prompt, self.model)
            if '```json' in response:
                json_text = response.split('```json')[1].split('```')[0].strip()
            else:
                json_text = response.strip()
            
            return json.loads(json_text)
        except Exception as e:
            return {
                "error": f"Cross-linguistic analysis failed: {str(e)}",
                "common_themes": [],
                "language_specific": {},
                "discourse_differences": [],
                "cultural_insights": [],
                "overall_comparison": "Analysis could not be completed"
            }
