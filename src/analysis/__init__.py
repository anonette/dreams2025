"""
Simplified Dream Analysis Package

This package provides two main approaches for dream analysis:
1. NLP-based analysis: Basic thematic analysis and clustering using TF-IDF and k-means
2. LLM-based discourse analysis: Advanced thematic and discourse analysis using structured prompts

Main Components:
- NLPDreamAnalyzer: Simple NLP-based thematic analysis
- LLMDiscourseAnalyzer: LLM-based discourse and thematic analysis
- DreamAnalysis: Main interface combining both approaches

Usage:
    from src.analysis import DreamAnalysis, analyze_dreams
    
    # Basic analysis (NLP only)
    results = analyze_dreams(dreams_by_language, analysis_type='basic')
    
    # Full analysis (NLP + LLM)
    results = analyze_dreams(dreams_by_language, llm_interface, analysis_type='full')
"""

from .nlp_analyzer import NLPDreamAnalyzer
from .llm_discourse_analyzer import LLMDiscourseAnalyzer
from .dream_analysis import DreamAnalysis, analyze_dreams

__all__ = [
    'NLPDreamAnalyzer',
    'LLMDiscourseAnalyzer', 
    'DreamAnalysis',
    'analyze_dreams'
]

__version__ = "2.0.0"
__description__ = "Simplified dream analysis focused on thematic and discourse analysis"
