"""
Main dream analysis interface - simplified and clean.
Provides access to NLP and LLM-based analysis for thematic and discourse analysis.
"""

from typing import List, Dict, Any, Optional
from .nlp_analyzer import NLPDreamAnalyzer
from .llm_discourse_analyzer import LLMDiscourseAnalyzer


class DreamAnalysis:
    """Main interface for dream analysis - combines NLP and LLM approaches."""
    
    def __init__(self, llm_interface=None, model_name: str = 'gpt-4o'):
        """
        Initialize dream analysis with optional LLM support.
        
        Args:
            llm_interface: LLM interface for discourse analysis (optional)
            model_name: Model to use for LLM analysis
        """
        # Always available: NLP analyzer
        self.nlp_analyzer = NLPDreamAnalyzer()
        
        # Optional: LLM analyzer for discourse analysis
        self.llm_analyzer = None
        if llm_interface:
            try:
                self.llm_analyzer = LLMDiscourseAnalyzer(llm_interface, model_name)
            except Exception as e:
                print(f"Warning: LLM analyzer initialization failed: {e}")
    
    def analyze_basic_themes(self, dreams: List[str]) -> Dict[str, Any]:
        """Basic thematic analysis using NLP."""
        return self.nlp_analyzer.analyze_dream_motives(dreams)
    
    def cluster_dreams_nlp(self, dreams: List[str], n_clusters: int = 5) -> Dict[str, Any]:
        """Cluster dreams using NLP approach."""
        return self.nlp_analyzer.cluster_dreams(dreams, n_clusters)
    
    def compare_languages_nlp(self, dreams_by_language: Dict[str, List[str]]) -> Dict[str, Any]:
        """Compare dream motives across languages using NLP."""
        return self.nlp_analyzer.compare_languages(dreams_by_language)
    
    def analyze_discourse_themes(self, dreams: List[str], language: str = 'english') -> Dict[str, Any]:
        """Advanced thematic analysis using LLM discourse analysis."""
        if not self.llm_analyzer:
            return {"error": "LLM analyzer not available. Use analyze_basic_themes() instead."}
        
        return self.llm_analyzer.analyze_dream_themes(dreams, language)
    
    def analyze_discourse_patterns(self, dreams: List[str], language: str = 'english') -> Dict[str, Any]:
        """Analyze discourse patterns using LLM."""
        if not self.llm_analyzer:
            return {"error": "LLM analyzer not available."}
        
        return self.llm_analyzer.analyze_discourse_patterns(dreams, language)
    
    def cluster_dreams_discourse(self, dreams: List[str], n_clusters: int = 5) -> Dict[str, Any]:
        """Cluster dreams using discourse analysis."""
        if not self.llm_analyzer:
            return {"error": "LLM analyzer not available. Use cluster_dreams_nlp() instead."}
        
        return self.llm_analyzer.cluster_by_discourse(dreams, n_clusters)
    
    def compare_cross_linguistic(self, dreams_by_language: Dict[str, List[str]]) -> Dict[str, Any]:
        """Compare dreams across languages using discourse analysis."""
        if not self.llm_analyzer:
            return {"error": "LLM analyzer not available. Use compare_languages_nlp() instead."}
        
        return self.llm_analyzer.compare_cross_linguistic(dreams_by_language)
    
    def full_analysis(self, dreams_by_language: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Perform complete analysis combining both approaches.
        
        Args:
            dreams_by_language: Dictionary mapping language names to lists of dreams
            
        Returns:
            Comprehensive analysis results
        """
        results = {
            'analysis_type': 'comprehensive',
            'nlp_analysis': {},
            'llm_analysis': {},
            'summary': {}
        }
        
        # NLP Analysis (always available)
        try:
            results['nlp_analysis'] = {
                'language_comparison': self.compare_languages_nlp(dreams_by_language),
                'clustering': {}
            }
            
            # Cluster dreams from each language
            for language, dreams in dreams_by_language.items():
                if dreams:
                    clustering = self.cluster_dreams_nlp(dreams, min(5, len(dreams)))
                    results['nlp_analysis']['clustering'][language] = clustering
        except Exception as e:
            results['nlp_analysis']['error'] = str(e)
        
        # LLM Analysis (if available)
        if self.llm_analyzer:
            try:
                results['llm_analysis'] = {
                    'cross_linguistic': self.compare_cross_linguistic(dreams_by_language),
                    'discourse_by_language': {}
                }
                
                # Analyze discourse patterns for each language
                for language, dreams in dreams_by_language.items():
                    if dreams:
                        themes = self.analyze_discourse_themes(dreams, language)
                        patterns = self.analyze_discourse_patterns(dreams, language)
                        clustering = self.cluster_dreams_discourse(dreams, min(5, len(dreams)))
                        
                        results['llm_analysis']['discourse_by_language'][language] = {
                            'themes': themes,
                            'patterns': patterns,
                            'clustering': clustering
                        }
            except Exception as e:
                results['llm_analysis']['error'] = str(e)
        else:
            results['llm_analysis']['message'] = "LLM analysis not available"
        
        # Generate summary
        results['summary'] = self._generate_summary(results, dreams_by_language)
        
        return results
    
    def _generate_summary(self, results: Dict[str, Any], dreams_by_language: Dict[str, List[str]]) -> Dict[str, Any]:
        """Generate analysis summary."""
        total_dreams = sum(len(dreams) for dreams in dreams_by_language.values())
        languages = list(dreams_by_language.keys())
        
        summary = {
            'total_dreams': total_dreams,
            'languages_analyzed': languages,
            'analysis_methods': ['NLP'],
            'key_findings': []
        }
        
        if self.llm_analyzer:
            summary['analysis_methods'].append('LLM_Discourse')
        
        # Extract key findings from NLP analysis
        nlp_analysis = results.get('nlp_analysis', {})
        if 'language_comparison' in nlp_analysis:
            lang_comp = nlp_analysis['language_comparison']
            for lang, data in lang_comp.items():
                if 'top_motives' in data and data['top_motives']:
                    top_motive = data['top_motives'][0][0]
                    summary['key_findings'].append(f"{lang}: top motive '{top_motive}'")
        
        return summary


# Convenience function for quick analysis
def analyze_dreams(dreams_by_language: Dict[str, List[str]], 
                  llm_interface=None, 
                  analysis_type: str = 'basic') -> Dict[str, Any]:
    """
    Quick analysis function.
    
    Args:
        dreams_by_language: Dictionary mapping languages to dream lists
        llm_interface: Optional LLM interface for advanced analysis
        analysis_type: 'basic' (NLP only) or 'full' (NLP + LLM)
    
    Returns:
        Analysis results
    """
    analyzer = DreamAnalysis(llm_interface)
    
    if analysis_type == 'basic':
        return {
            'language_comparison': analyzer.compare_languages_nlp(dreams_by_language),
            'clustering': {
                lang: analyzer.cluster_dreams_nlp(dreams) 
                for lang, dreams in dreams_by_language.items() if dreams
            }
        }
    else:
        return analyzer.full_analysis(dreams_by_language)
