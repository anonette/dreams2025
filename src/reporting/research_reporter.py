"""
Research-grade reporting system for cross-linguistic dream analysis.
Generates structured reports suitable for academic papers and data sharing.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import logging
import zipfile
import shutil
from collections import defaultdict, Counter
import scipy.stats as stats
from dataclasses import dataclass, asdict
import yaml
import asyncio

# Import typological analyzer
try:
    from ..analysis.typological_analyzer import TypologicalAnalyzer
    TYPOLOGICAL_AVAILABLE = True
except ImportError:
    TYPOLOGICAL_AVAILABLE = False
    TypologicalAnalyzer = None

# Import LLM interface
try:
    from ..models.llm_interface import LLMInterface
    LLM_INTERFACE_AVAILABLE = True
except ImportError:
    LLM_INTERFACE_AVAILABLE = False
    LLMInterface = None

@dataclass
class ResearchMetadata:
    """Metadata for research reports."""
    study_id: str
    title: str
    authors: List[str]
    institution: str
    date_generated: str
    languages_analyzed: List[str]
    models_used: List[str]
    total_dreams: int
    analysis_methods: List[str]
    keywords: List[str]
    abstract: str = ""
    
class ResearchReporter:
    """Generate research-grade reports and data packages."""
    
    def __init__(self, logs_dir: str = "logs", output_dir: str = "research_reports"):
        self.logs_dir = Path(logs_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Report structure
        self.report_sections = [
            "abstract",
            "introduction", 
            "methodology",
            "results",
            "discussion",
            "conclusion",
            "references",
            "appendices"
        ]
        
    def generate_research_report(self, 
                               session_ids: List[str],
                               metadata: ResearchMetadata,
                               include_data_package: bool = True,
                               include_typological_analysis: bool = True,
                               api_keys: Optional[Dict[str, str]] = None,
                               max_dreams_per_language: int = 50) -> Dict[str, Any]:
        """Generate a complete research report with data package."""
        
        report_id = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        report_dir = self.output_dir / report_id
        report_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Generating research report: {report_id}")
        
        # Load and analyze data
        data = self._load_session_data(session_ids)
        analysis_results = self._perform_comprehensive_analysis(data)
        
        # Add typological analysis if requested
        typological_results = None
        if include_typological_analysis and TYPOLOGICAL_AVAILABLE:
            self.logger.info("Running typological linguistic analysis...")
            typological_results = self._run_typological_analysis(data, api_keys, max_dreams_per_language)
            if typological_results:
                analysis_results['typological_analysis'] = typological_results
        
        # Generate report sections
        report = {
            'metadata': asdict(metadata),
            'sections': self._generate_report_sections(data, analysis_results, metadata),
            'tables': self._generate_research_tables(data, analysis_results),
            'figures': self._generate_research_figures(data, analysis_results, report_dir),
            'statistics': self._generate_statistical_summary(data, analysis_results),
            'data_summary': self._generate_data_summary(data)
        }
        
        # Save structured report
        self._save_structured_report(report, report_dir, report_id)
        
        # Generate LaTeX/academic format
        self._generate_latex_report(report, report_dir, report_id)
        
        # Generate markdown report
        self._generate_markdown_report(report, report_dir, report_id)
        
        # Create data package
        if include_data_package:
            self._create_data_package(data, analysis_results, report_dir, report_id)
        
        # Generate sharing package
        sharing_package = self._create_sharing_package(report_dir, report_id, metadata)
        
        self.logger.info(f"Research report generated: {report_dir}")
        
        return {
            'report_id': report_id,
            'report_dir': str(report_dir),
            'sharing_package': sharing_package,
            'files_generated': list(report_dir.glob('*')),
            'summary': report['statistics']
        }
    
    def _load_session_data(self, session_ids: List[str]) -> Dict[str, Any]:
        """Load data from multiple sessions."""
        data = {
            'api_calls': [],
            'dreams': [],
            'session_metadata': [],
            'temporal_stats': [],
            'rejected_dreams': []
        }
        
        for session_id in session_ids:
            # Find session files across all languages
            for lang_dir in self.logs_dir.iterdir():
                if lang_dir.is_dir() and not lang_dir.name.startswith('batch'):
                    session_path = lang_dir / "gpt-4o" / f"session_{session_id}"
                    if session_path.exists():
                        self._load_session_files(session_path, data, lang_dir.name, session_id)
        
        return data
    
    def _load_session_files(self, session_path: Path, data: Dict, language: str, session_id: str):
        """Load files from a specific session."""
        # API calls
        api_calls_file = session_path / "api_calls.csv"
        if api_calls_file.exists():
            df = pd.read_csv(api_calls_file)
            df['session_id'] = session_id
            df['language'] = language
            data['api_calls'].append(df)
        
        # Dreams
        dreams_file = session_path / "dreams.csv"
        if dreams_file.exists():
            df = pd.read_csv(dreams_file)
            df['session_id'] = session_id
            df['language'] = language
            data['dreams'].append(df)
        
        # Session metadata
        session_data_file = session_path / "session_data.json"
        if session_data_file.exists():
            with open(session_data_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
                session_data['session_id'] = session_id
                session_data['language'] = language
                data['session_metadata'].append(session_data)
        
        # Temporal statistics
        temporal_file = session_path / "temporal_statistics.json"
        if temporal_file.exists():
            with open(temporal_file, 'r', encoding='utf-8') as f:
                temporal_data = json.load(f)
                temporal_data['session_id'] = session_id
                temporal_data['language'] = language
                data['temporal_stats'].append(temporal_data)
        
        # Rejected dreams
        rejected_file = session_path / "rejected_dreams.csv"
        if rejected_file.exists():
            df = pd.read_csv(rejected_file)
            df['session_id'] = session_id
            df['language'] = language
            data['rejected_dreams'].append(df)
    
    def _perform_comprehensive_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis."""
        # Combine dataframes
        api_calls_df = pd.concat(data['api_calls'], ignore_index=True) if data['api_calls'] else pd.DataFrame()
        dreams_df = pd.concat(data['dreams'], ignore_index=True) if data['dreams'] else pd.DataFrame()
        
        analysis = {}
        
        if not api_calls_df.empty:
            # Basic statistics
            analysis['descriptive_stats'] = self._calculate_descriptive_stats(api_calls_df, dreams_df)
            
            # Success rate analysis
            analysis['success_rates'] = self._analyze_success_rates(api_calls_df)
            
            # Duration analysis
            analysis['duration_analysis'] = self._analyze_durations(api_calls_df)
            
            # Language comparison
            analysis['language_comparison'] = self._compare_languages(api_calls_df, dreams_df)
            
            # Content analysis
            if not dreams_df.empty:
                analysis['content_analysis'] = self._analyze_dream_content(dreams_df)
            
            # Temporal patterns
            analysis['temporal_patterns'] = self._analyze_temporal_patterns(api_calls_df)
            
            # Statistical tests
            analysis['statistical_tests'] = self._perform_statistical_tests(api_calls_df, dreams_df)
        
        return analysis
    
    def _calculate_descriptive_stats(self, api_calls_df: pd.DataFrame, dreams_df: pd.DataFrame) -> Dict:
        """Calculate descriptive statistics."""
        stats = {}
        
        # Overall statistics
        stats['total_api_calls'] = len(api_calls_df)
        stats['unique_languages'] = api_calls_df['language'].nunique()
        stats['unique_sessions'] = api_calls_df['session_id'].nunique()
        stats['date_range'] = {
            'start': str(api_calls_df['timestamp'].min()),
            'end': str(api_calls_df['timestamp'].max())
        }
        
        # Success rates
        stats['overall_success_rate'] = (api_calls_df['status'] == 'success').mean()
        stats['success_by_language'] = api_calls_df.groupby('language')['status'].apply(
            lambda x: (x == 'success').mean()
        ).to_dict()
        
        # Duration statistics
        if 'duration_seconds' in api_calls_df.columns:
            stats['duration_stats'] = {
                'mean': api_calls_df['duration_seconds'].mean(),
                'median': api_calls_df['duration_seconds'].median(),
                'std': api_calls_df['duration_seconds'].std(),
                'min': api_calls_df['duration_seconds'].min(),
                'max': api_calls_df['duration_seconds'].max()
            }
        
        # Dream content statistics
        if not dreams_df.empty and 'dream' in dreams_df.columns:
            dreams_df['dream_length'] = dreams_df['dream'].str.len()
            dreams_df['word_count'] = dreams_df['dream'].str.split().str.len()
            
            stats['dream_stats'] = {
                'total_dreams': len(dreams_df),
                'avg_length_chars': dreams_df['dream_length'].mean(),
                'avg_word_count': dreams_df['word_count'].mean(),
                'length_by_language': dreams_df.groupby('language')['dream_length'].mean().to_dict(),
                'words_by_language': dreams_df.groupby('language')['word_count'].mean().to_dict()
            }
        
        return stats
    
    def _analyze_success_rates(self, api_calls_df: pd.DataFrame) -> Dict:
        """Analyze success rates across different dimensions."""
        analysis = {}
        
        # Overall success rate
        analysis['overall'] = (api_calls_df['status'] == 'success').mean()
        
        # By language
        analysis['by_language'] = api_calls_df.groupby('language')['status'].apply(
            lambda x: (x == 'success').mean()
        ).to_dict()
        
        # By session
        analysis['by_session'] = api_calls_df.groupby('session_id')['status'].apply(
            lambda x: (x == 'success').mean()
        ).to_dict()
        
        # Confidence intervals
        n = len(api_calls_df)
        p = analysis['overall']
        se = np.sqrt(p * (1 - p) / n)
        analysis['confidence_interval_95'] = {
            'lower': p - 1.96 * se,
            'upper': p + 1.96 * se
        }
        
        return analysis
    
    def _analyze_durations(self, api_calls_df: pd.DataFrame) -> Dict:
        """Analyze API call durations."""
        if 'duration_seconds' not in api_calls_df.columns:
            return {}
        
        analysis = {}
        
        # Overall duration statistics
        durations = api_calls_df['duration_seconds']
        analysis['descriptive'] = {
            'mean': durations.mean(),
            'median': durations.median(),
            'std': durations.std(),
            'min': durations.min(),
            'max': durations.max(),
            'q25': durations.quantile(0.25),
            'q75': durations.quantile(0.75)
        }
        
        # By language
        by_lang_stats = api_calls_df.groupby('language')['duration_seconds'].agg([
            'mean', 'median', 'std', 'min', 'max'
        ])
        analysis['by_language'] = {col: by_lang_stats[col].to_dict() for col in by_lang_stats.columns}
        
        # By success status
        by_status_stats = api_calls_df.groupby('status')['duration_seconds'].agg([
            'mean', 'median', 'std'
        ])
        analysis['by_status'] = {col: by_status_stats[col].to_dict() for col in by_status_stats.columns}
        
        return analysis
    
    def _compare_languages(self, api_calls_df: pd.DataFrame, dreams_df: pd.DataFrame) -> Dict:
        """Compare languages across multiple dimensions."""
        comparison = {}
        
        # Success rate comparison
        lang_success = api_calls_df.groupby('language')['status'].apply(
            lambda x: (x == 'success').mean()
        )
        comparison['success_rates'] = lang_success.to_dict()
        comparison['success_rate_ranking'] = lang_success.sort_values(ascending=False).index.tolist()
        
        # Duration comparison
        if 'duration_seconds' in api_calls_df.columns:
            lang_duration = api_calls_df.groupby('language')['duration_seconds'].mean()
            comparison['avg_durations'] = lang_duration.to_dict()
            comparison['duration_ranking'] = lang_duration.sort_values().index.tolist()
        
        # Dream length comparison
        if not dreams_df.empty and 'dream' in dreams_df.columns:
            dreams_df['word_count'] = dreams_df['dream'].str.split().str.len()
            lang_length = dreams_df.groupby('language')['word_count'].mean()
            comparison['avg_dream_lengths'] = lang_length.to_dict()
            comparison['length_ranking'] = lang_length.sort_values(ascending=False).index.tolist()
        
        return comparison
    
    def _analyze_dream_content(self, dreams_df: pd.DataFrame) -> Dict:
        """Analyze dream content patterns."""
        if dreams_df.empty or 'dream' not in dreams_df.columns:
            return {}
        
        analysis = {}
        
        # Length analysis
        dreams_df['char_count'] = dreams_df['dream'].str.len()
        dreams_df['word_count'] = dreams_df['dream'].str.split().str.len()
        
        analysis['length_stats'] = {
            'char_count': dreams_df['char_count'].describe().to_dict(),
            'word_count': dreams_df['word_count'].describe().to_dict()
        }
        
        # By language
        analysis['by_language'] = {}
        for lang in dreams_df['language'].unique():
            lang_dreams = dreams_df[dreams_df['language'] == lang]
            analysis['by_language'][lang] = {
                'count': len(lang_dreams),
                'avg_chars': lang_dreams['char_count'].mean(),
                'avg_words': lang_dreams['word_count'].mean(),
                'std_chars': lang_dreams['char_count'].std(),
                'std_words': lang_dreams['word_count'].std()
            }
        
        return analysis
    
    def _analyze_temporal_patterns(self, api_calls_df: pd.DataFrame) -> Dict:
        """Analyze temporal patterns in the data."""
        if 'timestamp' not in api_calls_df.columns:
            return {}
        
        analysis = {}
        
        # Convert timestamps
        api_calls_df['timestamp'] = pd.to_datetime(api_calls_df['timestamp'])
        api_calls_df['hour'] = api_calls_df['timestamp'].dt.hour
        api_calls_df['day_of_week'] = api_calls_df['timestamp'].dt.day_name()
        
        # Hourly patterns
        hourly_success = api_calls_df.groupby('hour')['status'].apply(
            lambda x: (x == 'success').mean()
        )
        analysis['hourly_success_rates'] = hourly_success.to_dict()
        
        # Daily patterns
        daily_success = api_calls_df.groupby('day_of_week')['status'].apply(
            lambda x: (x == 'success').mean()
        )
        analysis['daily_success_rates'] = daily_success.to_dict()
        
        # Time series analysis
        api_calls_df['date'] = api_calls_df['timestamp'].dt.date
        daily_counts = api_calls_df.groupby('date').size()
        # Convert date keys to strings for JSON serialization
        analysis['daily_volume'] = {str(k): v for k, v in daily_counts.to_dict().items()}
        
        return analysis
    
    def _perform_statistical_tests(self, api_calls_df: pd.DataFrame, dreams_df: pd.DataFrame) -> Dict:
        """Perform statistical hypothesis tests."""
        tests = {}
        
        # Chi-square test for language vs success rate
        if len(api_calls_df['language'].unique()) > 1:
            contingency = pd.crosstab(api_calls_df['language'], api_calls_df['status'])
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            tests['language_success_chi2'] = {
                'chi2_statistic': chi2,
                'p_value': p_value,
                'degrees_of_freedom': dof,
                'significant': p_value < 0.05
            }
        
        # ANOVA for duration differences across languages
        if 'duration_seconds' in api_calls_df.columns and len(api_calls_df['language'].unique()) > 1:
            groups = [group['duration_seconds'].values for name, group in api_calls_df.groupby('language')]
            f_stat, p_value = stats.f_oneway(*groups)
            tests['duration_anova'] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        # Dream length comparison
        if not dreams_df.empty and 'dream' in dreams_df.columns:
            dreams_df['word_count'] = dreams_df['dream'].str.split().str.len()
            if len(dreams_df['language'].unique()) > 1:
                groups = [group['word_count'].values for name, group in dreams_df.groupby('language')]
                f_stat, p_value = stats.f_oneway(*groups)
                tests['dream_length_anova'] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        return tests
    
    def _run_typological_analysis(self, data: Dict[str, Any], api_keys: Optional[Dict[str, str]] = None, 
                                 max_dreams_per_language: int = 50) -> Optional[Dict[str, Any]]:
        """Run typological linguistic analysis on dream data."""
        if not TYPOLOGICAL_AVAILABLE:
            self.logger.warning("Typological analyzer not available")
            return None
        
        try:
            # Prepare dream data for typological analysis
            dreams_by_language = {}
            
            if data['dreams']:
                dreams_df = pd.concat(data['dreams'], ignore_index=True)
                
                # Filter successful dreams only
                successful_dreams = dreams_df[dreams_df.get('status', 'success') == 'success']
                
                for language in successful_dreams['language'].unique():
                    lang_dreams = successful_dreams[successful_dreams['language'] == language]
                    
                    dreams_list = []
                    for idx, row in lang_dreams.head(max_dreams_per_language).iterrows():
                        dream_dict = {
                            'dream_id': f"{language}_{idx}",
                            'dream_text': row.get('dream', ''),
                            'language': language,
                            'language_code': row.get('language_code', language[:2]),
                            'script': row.get('script', 'Latin'),
                            'timestamp': row.get('timestamp', ''),
                            'session_id': row.get('session_id', '')
                        }
                        if dream_dict['dream_text']:  # Only include non-empty dreams
                            dreams_list.append(dream_dict)
                    
                    if dreams_list:
                        dreams_by_language[language] = dreams_list
            
            if not dreams_by_language:
                self.logger.warning("No suitable dreams found for typological analysis")
                return None
            
            # Initialize typological analyzer
            llm_interface = None
            if api_keys and LLM_INTERFACE_AVAILABLE:
                llm_interface = LLMInterface(api_keys)
            
            analyzer = TypologicalAnalyzer(llm_interface=llm_interface)
            
            # Run analysis
            self.logger.info(f"Analyzing {sum(len(dreams) for dreams in dreams_by_language.values())} dreams across {len(dreams_by_language)} languages")
            
            # Use asyncio to run the analysis
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(
                    analyzer.analyze_dreams(dreams_by_language, max_dreams_per_language)
                )
            finally:
                loop.close()
            
            # Add visualizations
            results['visualizations'] = analyzer.create_visualizations(results)
            
            self.logger.info(f"Typological analysis completed: {results['total_analyzed']} dreams analyzed")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in typological analysis: {e}")
            return None
    
    def _generate_report_sections(self, data: Dict, analysis: Dict, metadata: ResearchMetadata) -> Dict:
        """Generate structured report sections."""
        sections = {}
        
        # Abstract
        sections['abstract'] = self._generate_abstract(analysis, metadata)
        
        # Introduction
        sections['introduction'] = self._generate_introduction(metadata)
        
        # Methodology
        sections['methodology'] = self._generate_methodology(data, metadata)
        
        # Results
        sections['results'] = self._generate_results(analysis)
        
        # Typological Analysis (if available)
        if 'typological_analysis' in analysis:
            sections['typological_results'] = self._generate_typological_results(analysis['typological_analysis'])
        
        # Discussion
        sections['discussion'] = self._generate_discussion(analysis)
        
        # Conclusion
        sections['conclusion'] = self._generate_conclusion(analysis)
        
        return sections
    
    def _generate_abstract(self, analysis: Dict, metadata: ResearchMetadata) -> str:
        """Generate abstract section."""
        if metadata.abstract:
            return metadata.abstract
        
        # Auto-generate abstract
        stats = analysis.get('descriptive_stats', {})
        abstract = f"""
This study examines cross-linguistic patterns in AI-generated dream narratives using {metadata.models_used[0]} 
across {len(metadata.languages_analyzed)} languages: {', '.join(metadata.languages_analyzed)}. 
We analyzed {stats.get('total_dreams', 'N/A')} generated dreams with an overall success rate of 
{stats.get('overall_success_rate', 0):.1%}. Results reveal significant differences in dream generation 
patterns across languages, with implications for understanding cultural bias in large language models.
        """.strip()
        
        return abstract
    
    def _generate_introduction(self, metadata: ResearchMetadata) -> str:
        """Generate introduction section."""
        return f"""
## Introduction

The intersection of artificial intelligence and cross-linguistic studies presents unique opportunities 
to understand how language models reflect cultural and linguistic patterns. This study investigates 
{metadata.title.lower()} using systematic prompt-based dream generation across multiple languages.

### Research Questions
1. Do large language models exhibit language-specific patterns in creative generation tasks?
2. How do success rates and content characteristics vary across different linguistic contexts?
3. What temporal and technical factors influence dream generation quality?

### Significance
Understanding these patterns is crucial for developing culturally-aware AI systems and identifying 
potential biases in multilingual applications.
        """.strip()
    
    def _generate_methodology(self, data: Dict, metadata: ResearchMetadata) -> str:
        """Generate methodology section."""
        total_sessions = len(set(session['session_id'] for session in data.get('session_metadata', [])))
        
        return f"""
## Methodology

### Data Collection
- **Languages Analyzed**: {', '.join(metadata.languages_analyzed)}
- **Models Used**: {', '.join(metadata.models_used)}
- **Sessions Conducted**: {total_sessions}
- **Collection Period**: {metadata.date_generated}

### Experimental Design
Dreams were generated using consistent prompts translated into each target language. 
Each session employed controlled parameters including temperature settings and batch processing 
to ensure systematic data collection.

### Analysis Methods
- Descriptive statistics for success rates and response characteristics
- Cross-linguistic comparison of content patterns
- Temporal analysis of generation patterns
- Statistical hypothesis testing for language differences

### Quality Control
All generated content was filtered for completeness and relevance. Rejected dreams were 
logged and analyzed separately to understand failure patterns.
        """.strip()
    
    def _generate_results(self, analysis: Dict) -> str:
        """Generate results section."""
        stats = analysis.get('descriptive_stats', {})
        success_rates = analysis.get('success_rates', {})
        
        results = f"""
## Results

### Overall Performance
- Total API calls: {stats.get('total_api_calls', 'N/A')}
- Overall success rate: {success_rates.get('overall', 0):.1%}
- Languages analyzed: {stats.get('unique_languages', 'N/A')}

### Language-Specific Patterns
"""
        
        # Add language-specific results
        if 'by_language' in success_rates:
            for lang, rate in success_rates['by_language'].items():
                results += f"- {lang.title()}: {rate:.1%} success rate\n"
        
        # Add statistical test results
        if 'statistical_tests' in analysis:
            tests = analysis['statistical_tests']
            results += "\n### Statistical Analysis\n"
            
            if 'language_success_chi2' in tests:
                test = tests['language_success_chi2']
                results += f"- Chi-square test for language-success association: χ² = {test['chi2_statistic']:.3f}, "
                results += f"p = {test['p_value']:.3f} ({'significant' if test['significant'] else 'not significant'})\n"
        
        return results.strip()
    
    def _generate_discussion(self, analysis: Dict) -> str:
        """Generate discussion section."""
        return """
## Discussion

The observed patterns in dream generation reveal important insights into the cultural and linguistic 
biases present in large language models. Language-specific differences in success rates and content 
characteristics suggest that model performance is not uniform across linguistic contexts.

### Implications for AI Development
These findings highlight the importance of considering linguistic diversity in AI system development 
and evaluation. The observed variations suggest that models may reflect training data biases that 
favor certain languages or cultural contexts.

### Limitations
This study is limited by the specific prompt structure used and the languages selected for analysis. 
Future research should expand to include additional languages and prompt variations.
        """.strip()
    
    def _generate_typological_results(self, typological_analysis: Dict) -> str:
        """Generate typological analysis results section."""
        results = typological_analysis
        summary = results.get('summary_stats', {})
        
        content = f"""
## Typological Linguistic Analysis

### Overview
This section presents findings from the typological linguistic analysis, which explores relationships between linguistic structural features (WALS) and narrative patterns in dream content.

### Data Summary
- **Dreams Analyzed**: {results.get('total_analyzed', 0)}
- **Languages**: {', '.join(summary.get('languages', []))}
- **Analysis Methods**: {summary.get('analysis_methods', {})}
  - LLM-scored dreams: {summary.get('analysis_methods', {}).get('llm', 0)}
  - Heuristic-scored dreams: {summary.get('analysis_methods', {}).get('heuristic', 0)}

### WALS Features Analysis
The analysis examines 12 typological features across languages:
- Tense/Aspect systems
- Grammatical alignment patterns  
- Subject expression (pro-drop)
- Modality marking
- Evidentiality systems
- Word order patterns
- Case marking complexity
- Definiteness systems
- Gender marking
- Number systems
- Negation strategies
- Voice systems

### Narrative Dimensions
Each dream was scored on 7 narrative dimensions (0.0-1.0 scale):
- **Dreamer Agency**: Control and agency levels
- **Other Agents**: Presence of other characters
- **Interaction**: Social interaction patterns
- **Emotion**: Emotional intensity
- **Temporal Coherence**: Timeline consistency
- **Cultural Motifs**: Culture-specific elements
- **Narrative Complexity**: Story structure complexity

### Key Findings
"""
        
        # Add language distance findings
        if 'language_distances' in results:
            content += "\n#### Typological Distances\n"
            for pair, distance_obj in results['language_distances'].items():
                content += f"- {pair[0].title()} ↔ {pair[1].title()}: {distance_obj.distance:.3f} "
                content += f"({distance_obj.shared_features}/{distance_obj.total_features} shared features)\n"
        
        # Add narrative pattern findings
        if 'correlations' in results:
            correlations = results['correlations']
            if 'language_narrative_means' in correlations:
                content += "\n#### Narrative Pattern Variations\n"
                lang_means = correlations['language_narrative_means']
                
                # Find most variable dimensions
                dimension_variance = {}
                for dim in ['dreamer_agency', 'other_agents', 'interaction', 'emotion', 
                           'temporal_coherence', 'cultural_motifs', 'narrative_complexity']:
                    values = [lang_means[lang].get(dim, 0) for lang in lang_means.keys()]
                    if values:
                        dimension_variance[dim] = max(values) - min(values)
                
                # Report top varying dimensions
                sorted_dims = sorted(dimension_variance.items(), key=lambda x: x[1], reverse=True)
                content += "Languages show the greatest variation in:\n"
                for dim, variance in sorted_dims[:3]:
                    content += f"- **{dim.replace('_', ' ').title()}**: Range = {variance:.3f}\n"
        
        content += """
### Methodology Note
This analysis employs a purely exploratory, data-driven approach without theoretical preconceptions. 
All patterns reported represent empirical observations in the sample data and should be interpreted 
as associations rather than causal relationships.
        """
        
        return content.strip()
    
    def _generate_conclusion(self, analysis: Dict) -> str:
        """Generate conclusion section."""
        conclusion = """
## Conclusion

This cross-linguistic analysis of AI-generated dreams provides evidence for language-specific patterns 
in large language model outputs. The systematic differences observed across languages have important 
implications for understanding and mitigating cultural bias in AI systems.
        """
        
        # Add typological conclusion if available
        if 'typological_analysis' in analysis:
            conclusion += """

### Typological Insights
The typological linguistic analysis reveals empirical relationships between linguistic structural 
features and narrative patterns in AI-generated dreams. These findings demonstrate the value of 
combining linguistic typology with computational analysis for understanding cross-linguistic 
variation in language model outputs.
            """
        
        conclusion += """

### Future Directions
Future research should expand this methodology to include additional languages, models, and generation 
tasks to develop a more comprehensive understanding of cross-linguistic AI behavior."""
        
        if 'typological_analysis' in analysis:
            conclusion += """
- Extend typological feature coverage
- Validate narrative scoring methods
- Explore causal mechanisms"""
        
        return conclusion.strip()
    
    def _generate_research_tables(self, data: Dict, analysis: Dict) -> Dict:
        """Generate research-quality tables."""
        tables = {}
        
        # Table 1: Language comparison summary
        if 'language_comparison' in analysis:
            comp = analysis['language_comparison']
            table_data = []
            
            for lang in comp.get('success_rates', {}):
                row = {'Language': lang.title()}
                if 'success_rates' in comp:
                    row['Success Rate'] = f"{comp['success_rates'][lang]:.1%}"
                if 'avg_durations' in comp:
                    row['Avg Duration (s)'] = f"{comp['avg_durations'].get(lang, 0):.2f}"
                if 'avg_dream_lengths' in comp:
                    row['Avg Dream Length (words)'] = f"{comp['avg_dream_lengths'].get(lang, 0):.1f}"
                table_data.append(row)
            
            tables['language_comparison'] = pd.DataFrame(table_data)
        
        # Table 2: Statistical test results
        if 'statistical_tests' in analysis:
            tests = analysis['statistical_tests']
            test_data = []
            
            for test_name, test_result in tests.items():
                if isinstance(test_result, dict):
                    test_data.append({
                        'Test': test_name.replace('_', ' ').title(),
                        'Statistic': f"{test_result.get('chi2_statistic', test_result.get('f_statistic', 'N/A')):.3f}",
                        'P-value': f"{test_result.get('p_value', 'N/A'):.3f}",
                        'Significant': 'Yes' if test_result.get('significant', False) else 'No'
                    })
            
            if test_data:
                tables['statistical_tests'] = pd.DataFrame(test_data)
        
        # Table 3: Typological analysis tables (if available)
        if 'typological_analysis' in analysis:
            typo_analysis = analysis['typological_analysis']
            
            # Language-Narrative means table
            if 'correlations' in typo_analysis and 'language_narrative_means' in typo_analysis['correlations']:
                lang_means = typo_analysis['correlations']['language_narrative_means']
                typo_table_data = []
                
                for language, means in lang_means.items():
                    row = {'Language': language.title()}
                    for dim, value in means.items():
                        row[dim.replace('_', ' ').title()] = f"{value:.3f}"
                    typo_table_data.append(row)
                
                if typo_table_data:
                    tables['typological_narrative_means'] = pd.DataFrame(typo_table_data)
            
            # Typological distances table
            if 'language_distances' in typo_analysis:
                distance_data = []
                for pair, distance_obj in typo_analysis['language_distances'].items():
                    distance_data.append({
                        'Language Pair': f"{pair[0].title()} - {pair[1].title()}",
                        'Typological Distance': f"{distance_obj.distance:.3f}",
                        'Shared Features': f"{distance_obj.shared_features}/{distance_obj.total_features}",
                        'Similarity %': f"{(distance_obj.shared_features/distance_obj.total_features)*100:.1f}%"
                    })
                
                if distance_data:
                    tables['typological_distances'] = pd.DataFrame(distance_data)
        
        return tables
    
    def _generate_research_figures(self, data: Dict, analysis: Dict, report_dir: Path) -> Dict:
        """Generate research-quality figures."""
        figures = {}
        figures_dir = report_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        # Figure 1: Success rates by language
        if 'language_comparison' in analysis and 'success_rates' in analysis['language_comparison']:
            plt.figure(figsize=(10, 6))
            success_rates = analysis['language_comparison']['success_rates']
            
            languages = list(success_rates.keys())
            rates = [success_rates[lang] for lang in languages]
            
            bars = plt.bar(languages, rates, color='steelblue', alpha=0.7)
            plt.title('Dream Generation Success Rates by Language', fontsize=14, fontweight='bold')
            plt.xlabel('Language', fontsize=12)
            plt.ylabel('Success Rate', fontsize=12)
            plt.ylim(0, 1)
            
            # Add value labels on bars
            for bar, rate in zip(bars, rates):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            figure_path = figures_dir / "success_rates_by_language.png"
            plt.savefig(figure_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            figures['success_rates'] = str(figure_path)
        
        # Figure 2: Duration analysis
        if 'duration_analysis' in analysis and 'by_language' in analysis['duration_analysis']:
            plt.figure(figsize=(10, 6))
            duration_data = analysis['duration_analysis']['by_language']
            
            languages = list(duration_data['mean'].keys())
            durations = [duration_data['mean'][lang] for lang in languages]
            errors = [duration_data['std'][lang] for lang in languages]
            
            bars = plt.bar(languages, durations, yerr=errors, capsize=5, 
                          color='lightcoral', alpha=0.7, error_kw={'ecolor': 'black'})
            plt.title('Average API Call Duration by Language', fontsize=14, fontweight='bold')
            plt.xlabel('Language', fontsize=12)
            plt.ylabel('Duration (seconds)', fontsize=12)
            
            plt.tight_layout()
            figure_path = figures_dir / "duration_by_language.png"
            plt.savefig(figure_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            figures['durations'] = str(figure_path)
        
        # Typological analysis figures (if available)
        if 'typological_analysis' in analysis:
            typo_analysis = analysis['typological_analysis']
            
            # Figure 3: Narrative dimensions heatmap
            if 'correlations' in typo_analysis and 'language_narrative_means' in typo_analysis['correlations']:
                lang_means = typo_analysis['correlations']['language_narrative_means']
                
                # Create DataFrame for heatmap
                heatmap_data = []
                dimensions = ['dreamer_agency', 'other_agents', 'interaction', 'emotion', 
                             'temporal_coherence', 'cultural_motifs', 'narrative_complexity']
                
                for language in lang_means.keys():
                    row = [lang_means[language].get(dim, 0) for dim in dimensions]
                    heatmap_data.append(row)
                
                if heatmap_data:
                    plt.figure(figsize=(12, 8))
                    
                    # Create heatmap
                    import seaborn as sns
                    df_heatmap = pd.DataFrame(
                        heatmap_data,
                        index=[lang.title() for lang in lang_means.keys()],
                        columns=[dim.replace('_', ' ').title() for dim in dimensions]
                    )
                    
                    sns.heatmap(df_heatmap, annot=True, cmap='RdYlBu_r', center=0.5,
                               fmt='.3f', cbar_kws={'label': 'Score (0.0-1.0)'})
                    plt.title('Narrative Dimension Scores by Language\n(Typological Analysis)', 
                             fontsize=14, fontweight='bold')
                    plt.xlabel('Narrative Dimension', fontsize=12)
                    plt.ylabel('Language', fontsize=12)
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    
                    figure_path = figures_dir / "typological_narrative_heatmap.png"
                    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    figures['typological_heatmap'] = str(figure_path)
            
            # Figure 4: Typological distance matrix
            if 'language_distances' in typo_analysis:
                distances = typo_analysis['language_distances']
                languages = list(set([pair[0] for pair in distances.keys()] + 
                                   [pair[1] for pair in distances.keys()]))
                
                # Create distance matrix
                n_langs = len(languages)
                dist_matrix = np.zeros((n_langs, n_langs))
                
                for i, lang1 in enumerate(languages):
                    for j, lang2 in enumerate(languages):
                        if i != j:
                            pair = (lang1, lang2) if (lang1, lang2) in distances else (lang2, lang1)
                            if pair in distances:
                                dist_matrix[i, j] = distances[pair].distance
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(dist_matrix, 
                           xticklabels=[lang.title() for lang in languages],
                           yticklabels=[lang.title() for lang in languages],
                           annot=True, fmt='.3f', cmap='viridis',
                           cbar_kws={'label': 'Typological Distance'})
                plt.title('Typological Distance Matrix (WALS Features)', 
                         fontsize=14, fontweight='bold')
                plt.xlabel('Language', fontsize=12)
                plt.ylabel('Language', fontsize=12)
                plt.tight_layout()
                
                figure_path = figures_dir / "typological_distance_matrix.png"
                plt.savefig(figure_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                figures['typological_distances'] = str(figure_path)
        
        return figures
    
    def _generate_statistical_summary(self, data: Dict, analysis: Dict) -> Dict:
        """Generate statistical summary for the report."""
        summary = {}
        
        # Extract key statistics
        if 'descriptive_stats' in analysis:
            stats = analysis['descriptive_stats']
            summary.update({
                'total_observations': stats.get('total_api_calls', 0),
                'unique_languages': stats.get('unique_languages', 0),
                'overall_success_rate': stats.get('overall_success_rate', 0),
                'date_range': stats.get('date_range', {})
            })
        
        # Add significance tests
        if 'statistical_tests' in analysis:
            tests = analysis['statistical_tests']
            summary['significant_findings'] = []
            
            for test_name, test_result in tests.items():
                if isinstance(test_result, dict) and test_result.get('significant', False):
                    summary['significant_findings'].append({
                        'test': test_name,
                        'p_value': test_result.get('p_value'),
                        'statistic': test_result.get('chi2_statistic', test_result.get('f_statistic'))
                    })
        
        return summary
    
    def _generate_data_summary(self, data: Dict) -> Dict:
        """Generate summary of the underlying data."""
        summary = {}
        
        # Count data by type
        summary['data_files'] = {
            'api_calls_files': len(data.get('api_calls', [])),
            'dreams_files': len(data.get('dreams', [])),
            'session_metadata_files': len(data.get('session_metadata', [])),
            'temporal_stats_files': len(data.get('temporal_stats', [])),
            'rejected_dreams_files': len(data.get('rejected_dreams', []))
        }
        
        # Total records
        total_api_calls = sum(len(df) for df in data.get('api_calls', []))
        total_dreams = sum(len(df) for df in data.get('dreams', []))
        
        summary['record_counts'] = {
            'total_api_calls': total_api_calls,
            'total_dreams': total_dreams,
            'total_sessions': len(data.get('session_metadata', []))
        }
        
        return summary
    
    def _save_structured_report(self, report: Dict, report_dir: Path, report_id: str):
        """Save the structured report in multiple formats."""
        
        # Save complete report as JSON
        report_file = report_dir / f"{report_id}_complete_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        # Save metadata separately
        metadata_file = report_dir / f"{report_id}_metadata.yaml"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            yaml.dump(report['metadata'], f, default_flow_style=False)
        
        # Save tables as CSV
        tables_dir = report_dir / "tables"
        tables_dir.mkdir(exist_ok=True)
        
        for table_name, table_df in report['tables'].items():
            table_file = tables_dir / f"{table_name}.csv"
            table_df.to_csv(table_file, index=False)
    
    def _generate_latex_report(self, report: Dict, report_dir: Path, report_id: str):
        """Generate LaTeX format report for academic submission."""
        latex_content = f"""
\\documentclass[12pt]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{amsmath}}
\\usepackage{{amsfonts}}
\\usepackage{{amssymb}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{hyperref}}

\\title{{{report['metadata']['title']}}}
\\author{{{', '.join(report['metadata']['authors'])}}}
\\date{{{report['metadata']['date_generated']}}}

\\begin{{document}}

\\maketitle

{report['sections']['abstract']}

{report['sections']['introduction']}

{report['sections']['methodology']}

{report['sections']['results']}

{report['sections']['discussion']}

{report['sections']['conclusion']}

\\end{{document}}
        """.strip()
        
        latex_file = report_dir / f"{report_id}_report.tex"
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write(latex_content)
    
    def _generate_markdown_report(self, report: Dict, report_dir: Path, report_id: str):
        """Generate Markdown format report."""
        markdown_content = f"""
# {report['metadata']['title']}

**Authors**: {', '.join(report['metadata']['authors'])}  
**Institution**: {report['metadata']['institution']}  
**Date**: {report['metadata']['date_generated']}  
**Study ID**: {report['metadata']['study_id']}

---

{report['sections']['abstract']}

{report['sections']['introduction']}

{report['sections']['methodology']}

{report['sections']['results']}
"""
        
        # Add typological results if available
        if 'typological_results' in report['sections']:
            markdown_content += f"""
{report['sections']['typological_results']}
"""
        
        markdown_content += """
## Tables

"""
        
        # Add table references
        for table_name, table_df in report['tables'].items():
            markdown_content += f"### {table_name.replace('_', ' ').title()}\n\n"
            markdown_content += table_df.to_markdown(index=False) + "\n\n"
        
        markdown_content += f"""
## Figures

"""
        
        # Add figure references
        for figure_name, figure_path in report['figures'].items():
            markdown_content += f"### {figure_name.replace('_', ' ').title()}\n\n"
            markdown_content += f"![{figure_name}]({figure_path})\n\n"
        
        markdown_content += f"""
{report['sections']['discussion']}

{report['sections']['conclusion']}

---

**Keywords**: {', '.join(report['metadata']['keywords'])}
        """.strip()
        
        markdown_file = report_dir / f"{report_id}_report.md"
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
    
    def _create_data_package(self, data: Dict, analysis: Dict, report_dir: Path, report_id: str):
        """Create a comprehensive data package for sharing."""
        data_dir = report_dir / "data_package"
        data_dir.mkdir(exist_ok=True)
        
        # Save combined datasets
        if data['api_calls']:
            combined_api_calls = pd.concat(data['api_calls'], ignore_index=True)
            combined_api_calls.to_csv(data_dir / "combined_api_calls.csv", index=False)
        
        if data['dreams']:
            combined_dreams = pd.concat(data['dreams'], ignore_index=True)
            combined_dreams.to_csv(data_dir / "combined_dreams.csv", index=False)
        
        # Save analysis results
        analysis_file = data_dir / "analysis_results.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
        
        # Create data dictionary
        data_dictionary = {
            "api_calls.csv": {
                "description": "Complete record of all API calls made during dream generation",
                "columns": {
                    "timestamp": "ISO timestamp of the API call",
                    "language": "Target language for dream generation",
                    "script": "Writing script used",
                    "status": "Call result (success/error/filtered)",
                    "duration_seconds": "Time taken for the API call",
                    "batch_id": "Unique identifier for the batch",
                    "prompt_id": "Unique identifier for the prompt used",
                    "user_id": "User identifier",
                    "session_id": "Session identifier"
                }
            },
            "dreams.csv": {
                "description": "Generated dream content with metadata",
                "columns": {
                    "timestamp": "ISO timestamp of dream generation",
                    "language": "Language of the generated dream",
                    "script": "Writing script used",
                    "dream": "The generated dream text",
                    "prompt_id": "Unique identifier for the prompt used",
                    "batch_id": "Unique identifier for the batch",
                    "session_id": "Session identifier"
                }
            },
            "analysis_results.json": {
                "description": "Complete statistical analysis results",
                "sections": {
                    "descriptive_stats": "Basic descriptive statistics",
                    "success_rates": "Success rate analysis by various factors",
                    "duration_analysis": "API call duration analysis",
                    "language_comparison": "Cross-linguistic comparison results",
                    "content_analysis": "Dream content pattern analysis",
                    "statistical_tests": "Hypothesis test results"
                }
            }
        }
        
        dictionary_file = data_dir / "data_dictionary.json"
        with open(dictionary_file, 'w', encoding='utf-8') as f:
            json.dump(data_dictionary, f, indent=2, ensure_ascii=False)
    
    def _create_sharing_package(self, report_dir: Path, report_id: str, metadata: ResearchMetadata) -> str:
        """Create a compressed package for easy sharing."""
        
        # Create sharing info
        sharing_info = {
            "study_title": metadata.title,
            "study_id": metadata.study_id,
            "authors": metadata.authors,
            "institution": metadata.institution,
            "date_generated": metadata.date_generated,
            "languages_analyzed": metadata.languages_analyzed,
            "total_dreams": metadata.total_dreams,
            "package_contents": {
                "reports": ["LaTeX format (.tex)", "Markdown format (.md)", "JSON format (.json)"],
                "data": ["Combined datasets (CSV)", "Analysis results (JSON)", "Data dictionary"],
                "figures": ["High-resolution figures (PNG)", "Publication-ready charts"],
                "tables": ["Research tables (CSV)", "Formatted for publication"]
            },
            "usage_instructions": {
                "citation": f"{', '.join(metadata.authors)}. ({datetime.now().year}). {metadata.title}. {metadata.institution}.",
                "data_format": "CSV files can be opened in Excel, R, Python, or any statistical software",
                "figures": "PNG files are publication-ready at 300 DPI",
                "latex": "Use the .tex file for academic submission to journals"
            }
        }
        
        readme_file = report_dir / "README.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(f"""# {metadata.title}

## Study Information
- **Study ID**: {metadata.study_id}
- **Authors**: {', '.join(metadata.authors)}
- **Institution**: {metadata.institution}
- **Date**: {metadata.date_generated}

## Package Contents
- `{report_id}_report.md` - Main report in Markdown format
- `{report_id}_report.tex` - LaTeX format for academic submission
- `{report_id}_complete_report.json` - Complete structured report
- `data_package/` - All datasets and analysis results
- `figures/` - Publication-ready figures
- `tables/` - Research tables in CSV format

## Citation
{sharing_info['usage_instructions']['citation']}

## Keywords
{', '.join(metadata.keywords)}
""")
        
        # Create ZIP package
        zip_path = report_dir.parent / f"{report_id}_sharing_package.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in report_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(report_dir)
                    zipf.write(file_path, arcname)
        
        return str(zip_path)

# Example usage functions
def generate_sample_report():
    """Generate a sample research report."""
    
    # Sample metadata
    metadata = ResearchMetadata(
        study_id="DREAM_2025_001",
        title="Cross-Linguistic Patterns in AI-Generated Dream Narratives",
        authors=["Dr. Jane Smith", "Dr. John Doe"],
        institution="University of AI Research",
        date_generated=datetime.now().strftime("%Y-%m-%d"),
        languages_analyzed=["english", "serbian", "hebrew", "slovenian", "basque"],
        models_used=["gpt-4o"],
        total_dreams=500,
        analysis_methods=["Descriptive Statistics", "Chi-square Tests", "ANOVA", "Content Analysis"],
        keywords=["cross-linguistic", "AI", "dreams", "cultural bias", "LLM"],
        abstract="This study examines cross-linguistic patterns in AI-generated dream narratives..."
    )
    
    # Initialize reporter
    reporter = ResearchReporter()
    
    # Generate report (assuming session IDs exist)
    session_ids = ["20250625_155722"]  # Replace with actual session IDs
    
    try:
        result = reporter.generate_research_report(session_ids, metadata)
        print(f"Research report generated successfully!")
        print(f"Report ID: {result['report_id']}")
        print(f"Report directory: {result['report_dir']}")
        print(f"Sharing package: {result['sharing_package']}")
        return result
    except Exception as e:
        print(f"Error generating report: {e}")
        return None

if __name__ == "__main__":
    # Generate sample report
    generate_sample_report()