#!/usr/bin/env python3
"""
Streamlit Dream Analysis Interface - Optimized V2 Data
Comprehensive dashboard for dream data analysis with session management
Works with logs_optimized_v2 directory
"""

import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from collections import defaultdict, Counter
import subprocess
import sys
import io
import zipfile
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import asyncio

# Import analysis modules
sys.path.append('.')
try:
    from cultural_dream_analysis import CulturalDreamAnalyzer
    BASIC_CULTURAL_AVAILABLE = True
except ImportError:
    BASIC_CULTURAL_AVAILABLE = False
    
# Import LLM interface for semiotic analysis
try:
    from src.models.llm_interface import LLMInterface, GenerationConfig
    LLM_INTERFACE_AVAILABLE = True
except ImportError:
    LLM_INTERFACE_AVAILABLE = False

# Import cultural dream analyst
try:
    from cultural_dream_analyst_persona import CulturalDreamAnalyst
    CULTURAL_ANALYST_AVAILABLE = True
except ImportError:
    try:
        # Try alternative path
        sys.path.append('July5reports/session_20250705_213110/cultural_analysis')
        from cultural_dream_analyst_persona import CulturalDreamAnalyst
        CULTURAL_ANALYST_AVAILABLE = True
    except ImportError:
        CULTURAL_ANALYST_AVAILABLE = False
        CulturalDreamAnalyst = None

# Import multilingual analyzer
try:
    from Tests.analyze_multilingual_data import MultilingualDreamAnalyzer
    MULTILINGUAL_ANALYZER_AVAILABLE = True
except ImportError:
    MULTILINGUAL_ANALYZER_AVAILABLE = False
    MultilingualDreamAnalyzer = None

# Import typological analyzer
try:
    from src.analysis.typological_analyzer import TypologicalAnalyzer
    TYPOLOGICAL_ANALYZER_AVAILABLE = True
except ImportError:
    TYPOLOGICAL_ANALYZER_AVAILABLE = False
    TypologicalAnalyzer = None

class StreamlitDreamAnalyzerOptimized:
    def __init__(self):
        # MODIFIED: Point to logs_optimized_v2 directory
        self.logs_dir = Path("logs_optimized_v2")
        self.output_dir = Path("analysis_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Session management
        if 'session_id' not in st.session_state:
            st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize analysis state
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}
        if 'available_data' not in st.session_state:
            st.session_state.available_data = None
            
        # Cultural semiotic analysis system prompt
        self.cultural_semiotic_prompt = """You are a cultural semiotician analyzing dream narratives. For each dream, extract the following narrative markers:
1. Is the dream anchored in a specific spatial, historical, or cultural world? (Yes/No)
2. Describe the markers of this anchoring (e.g. named places, ritual objects, ethnic language, war imagery).
3. What is the dominant emotional affect? (e.g. utopian, melancholic, anxious, euphoric, uncanny)
4. Does the dream show signs of collective memory or trauma? (Yes/No; explain)
5. What is the narrative structure? (e.g. journey, return, transformation, arrest)
6. Does the language resemble oral storytelling? (Yes/No; if Yes, describe how)
7. Assign a cultural specificity score from 0.0 (completely generic) to 1.0 (highly situated).

Return your output in structured JSON format."""
            
        # Load thematic analysis data - Enhanced for optimized data
        self.theme_data = {
            'Nature Spiritual': {'English': 94.0, 'Basque': 99.0, 'Hebrew': 81.0, 'Serbian': 90.0, 'Slovenian': 86.9},
            'Light Illumination': {'English': 76.0, 'Basque': 67.0, 'Hebrew': 61.0, 'Serbian': 58.0, 'Slovenian': 40.4},
            'Time Change': {'English': 63.0, 'Basque': 35.0, 'Hebrew': 61.0, 'Serbian': 61.0, 'Slovenian': 61.6},
            'Transportation': {'English': 89.0, 'Basque': 34.0, 'Hebrew': 66.0, 'Serbian': 44.0, 'Slovenian': 40.4},
            'Food Nourishment': {'English': 71.0, 'Basque': 58.0, 'Hebrew': 44.0, 'Serbian': 51.0, 'Slovenian': 32.3},
            'Water Emotion': {'English': 70.0, 'Basque': 45.0, 'Hebrew': 43.0, 'Serbian': 46.0, 'Slovenian': 44.4},
            'Animals Instinct': {'English': 84.0, 'Basque': 39.0, 'Hebrew': 30.0, 'Serbian': 22.0, 'Slovenian': 27.3},
            'Violence Conflict': {'English': 68.0, 'Basque': 22.0, 'Hebrew': 34.0, 'Serbian': 61.0, 'Slovenian': 47.5},
            'Money Security': {'English': 69.0, 'Basque': 19.0, 'Hebrew': 41.0, 'Serbian': 26.0, 'Slovenian': 24.2},
            'Magic Supernatural': {'English': 66.0, 'Basque': 18.0, 'Hebrew': 34.0, 'Serbian': 32.0, 'Slovenian': 20.2}
        }
        
        # Statistical analysis data
        self.dream_stats = {
            'English': {'count': 93, 'success_rate': 93.0, 'avg_length': 414.4},
            'Basque': {'count': 55, 'success_rate': 55.0, 'avg_length': 115.7},
            'Hebrew': {'count': 75, 'success_rate': 75.0, 'avg_length': 147.2},
            'Serbian': {'count': 77, 'success_rate': 77.0, 'avg_length': 141.7},
            'Slovenian': {'count': 96, 'success_rate': 96.0, 'avg_length': 334.6}
        }
        
        self.languages = ['English', 'Basque', 'Hebrew', 'Serbian', 'Slovenian']
            
    def load_available_data(self):
        """Load and cache available dream data from logs_optimized_v2"""
        if st.session_state.available_data is None:
            data = self.scan_dream_data()
            st.session_state.available_data = data
        return st.session_state.available_data
    
    def scan_dream_data(self):
        """Scan logs_optimized_v2 directory for available dream data"""
        languages = {}
        total_dreams = 0
        
        if not self.logs_dir.exists():
            st.error(f"❌ Directory {self.logs_dir} not found!")
            return {'languages': {}, 'total_dreams': 0, 'total_languages': 0}
        
        for lang_dir in self.logs_dir.iterdir():
            if lang_dir.is_dir() and not lang_dir.name.startswith('.'):
                language = lang_dir.name
                
                # Find GPT-4o sessions
                gpt4o_dir = lang_dir / "gpt-4o"
                if gpt4o_dir.exists():
                    sessions = []
                    for session_dir in gpt4o_dir.iterdir():
                        if session_dir.is_dir():
                            dreams_file = session_dir / "dreams.csv"
                            if dreams_file.exists():
                                try:
                                    df = pd.read_csv(dreams_file)
                                    dream_count = len(df[df['status'] == 'success'])
                                    sessions.append({
                                        'session_id': session_dir.name,
                                        'dreams_count': dream_count,
                                        'total_calls': len(df),
                                        'success_rate': dream_count / len(df) * 100 if len(df) > 0 else 0,
                                        'last_modified': session_dir.stat().st_mtime
                                    })
                                    total_dreams += dream_count
                                except Exception as e:
                                    st.warning(f"Error reading {dreams_file}: {e}")
                    
                    if sessions:
                        languages[language] = {
                            'sessions': sessions,
                            'total_dreams': sum(s['dreams_count'] for s in sessions),
                            'latest_session': max(sessions, key=lambda x: x['last_modified'])
                        }
        
        return {
            'languages': languages,
            'total_dreams': total_dreams,
            'total_languages': len(languages)
        }
    
    def create_session_output_dir(self, analysis_type):
        """Create session-specific output directory"""
        session_dir = self.output_dir / f"session_{st.session_state.session_id}" / analysis_type
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir
    
    def load_dream_data_for_analysis(self, max_dreams_per_language=50):
        """Load dream data from logs_optimized_v2 for analysis"""
        dreams_by_language = {}
        
        for lang_dir in self.logs_dir.iterdir():
            if lang_dir.is_dir() and not lang_dir.name.startswith('.'):
                language = lang_dir.name
                
                # Find the latest session
                gpt4o_dir = lang_dir / "gpt-4o"
                if gpt4o_dir.exists():
                    sessions = []
                    for session_dir in gpt4o_dir.iterdir():
                        if session_dir.is_dir() and session_dir.name.startswith('session_'):
                            dreams_file = session_dir / "dreams.csv"
                            if dreams_file.exists():
                                sessions.append((session_dir.stat().st_mtime, session_dir))
                    
                    if sessions:
                        # Use the most recent session
                        latest_session = max(sessions, key=lambda x: x[0])[1]
                        dreams_file = latest_session / "dreams.csv"
                        
                        try:
                            df = pd.read_csv(dreams_file)
                            successful_dreams = df[df['status'] == 'success']
                            
                            # Limit number of dreams
                            if len(successful_dreams) > max_dreams_per_language:
                                successful_dreams = successful_dreams.head(max_dreams_per_language)
                            
                            dreams_by_language[language] = successful_dreams['dream'].tolist()
                            
                        except Exception as e:
                            st.warning(f"Error loading dreams for {language}: {e}")
        
        return dreams_by_language
    
    def create_thematic_visualizations(self):
        """Create interactive thematic visualizations"""
        # Theme dominance heatmap
        df = pd.DataFrame(self.theme_data).T
        
        fig_heatmap = px.imshow(
            df.values,
            x=df.columns,
            y=[theme.replace('_', ' ').title() for theme in df.index],
            color_continuous_scale='YlOrRd',
            title='Dream Theme Prevalence Across Languages (%) - Optimized Data',
            labels={'x': 'Language', 'y': 'Theme', 'color': 'Prevalence (%)'}
        )
        fig_heatmap.update_layout(height=800)
        
        # English dominance chart
        english_percentages = [self.theme_data[theme]['English'] for theme in self.theme_data]
        themes = [theme.replace('_', ' ').title() for theme in self.theme_data.keys()]
        
        fig_english = px.bar(
            x=english_percentages,
            y=themes,
            orientation='h',
            title='English Dominance Across Dream Themes - Optimized Data',
            labels={'x': 'English Prevalence (%)', 'y': 'Theme'},
            color=english_percentages,
            color_continuous_scale='RdYlBu_r'
        )
        fig_english.update_layout(height=800)
        
        # Theme variation chart
        theme_ranges = {}
        for theme, data in self.theme_data.items():
            values = list(data.values())
            theme_ranges[theme] = max(values) - min(values)
        
        sorted_themes = sorted(theme_ranges.items(), key=lambda x: x[1], reverse=True)
        themes_sorted, ranges_sorted = zip(*sorted_themes)
        
        fig_variation = px.bar(
            x=[theme.replace('_', ' ').title() for theme in themes_sorted],
            y=ranges_sorted,
            title='Cultural Variation in Dream Themes - Optimized Data',
            labels={'x': 'Theme', 'y': 'Variation Range (%)'},
            color=ranges_sorted,
            color_continuous_scale='Viridis'
        )
        fig_variation.update_xaxes(tickangle=45)
        fig_variation.update_layout(height=600)
        
        return {
            'heatmap': fig_heatmap,
            'english_dominance': fig_english,
            'variation': fig_variation
        }
    
    def create_statistical_visualizations(self):
        """Create statistical analysis visualizations"""
        available_data = self.load_available_data()
        
        if not available_data['languages']:
            st.warning("No data available for visualization")
            return {}
        
        # Extract real data from available_data
        languages = list(available_data['languages'].keys())
        dream_counts = [available_data['languages'][lang]['total_dreams'] for lang in languages]
        
        # Success rates from latest sessions
        success_rates = []
        for lang in languages:
            latest_session = available_data['languages'][lang]['latest_session']
            success_rates.append(latest_session['success_rate'])
        
        # Create success rate comparison
        fig_success = px.bar(
            x=languages,
            y=success_rates,
            title='Success Rates by Language - Optimized V2 Data',
            labels={'x': 'Language', 'y': 'Success Rate (%)'},
            color=success_rates,
            color_continuous_scale='RdYlGn'
        )
        
        # Create dream count comparison
        fig_counts = px.bar(
            x=languages,
            y=dream_counts,
            title='Dream Count by Language - Optimized V2 Data',
            labels={'x': 'Language', 'y': 'Number of Dreams'},
            color=dream_counts,
            color_continuous_scale='Blues'
        )
        
        # Success vs Count scatter
        fig_scatter = px.scatter(
            x=success_rates,
            y=dream_counts,
            text=languages,
            title='Success Rate vs Dream Count - Optimized V2 Data',
            labels={'x': 'Success Rate (%)', 'y': 'Number of Dreams'},
            size=dream_counts,
            color=languages,
            size_max=30
        )
        fig_scatter.update_traces(textposition="middle right")
        
        return {
            'success_rates': fig_success,
            'dream_counts': fig_counts,
            'scatter': fig_scatter
        }

def main():
    st.set_page_config(
        page_title="Dream Analysis - Optimized V2",
        page_icon="🌙",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🌙 Dream Analysis Dashboard - Optimized V2 Data")
    st.markdown("*Comprehensive analysis of optimized dream generation data*")
    
    # Initialize analyzer
    analyzer = StreamlitDreamAnalyzerOptimized()
    
    # Load available data
    available_data = analyzer.load_available_data()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Data Overview")
        
        if available_data['total_dreams'] > 0:
            st.metric("Total Dreams", available_data['total_dreams'])
            st.metric("Languages", available_data['total_languages'])
            
            st.subheader("Available Languages")
            for lang, data in available_data['languages'].items():
                with st.expander(f"{lang.title()} ({data['total_dreams']} dreams)"):
                    latest = data['latest_session']
                    st.write(f"**Latest Session:** {latest['session_id']}")
                    st.write(f"**Success Rate:** {latest['success_rate']:.1f}%")
                    st.write(f"**Dreams:** {latest['dreams_count']}")
                    st.write(f"**Total Calls:** {latest['total_calls']}")
        else:
            st.warning("No dream data found in logs_optimized_v2")
            st.info("Make sure the logs_optimized_v2 directory contains processed dream data")
    
    # Main content
    if available_data['total_dreams'] > 0:
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎨 Thematic Analysis", "📈 Statistical Analysis", "🔍 Data Explorer"])
        
        with tab1:
            st.header("Data Overview - Optimized V2")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Dreams", available_data['total_dreams'])
            with col2:
                st.metric("Languages", available_data['total_languages'])
            with col3:
                avg_success = sum(data['latest_session']['success_rate'] 
                                for data in available_data['languages'].values()) / len(available_data['languages'])
                st.metric("Avg Success Rate", f"{avg_success:.1f}%")
            
            # Show visualizations
            st.subheader("Statistical Overview")
            viz = analyzer.create_statistical_visualizations()
            
            if viz:
                col1, col2 = st.columns(2)
                with col1:
                    if 'success_rates' in viz:
                        st.plotly_chart(viz['success_rates'], use_container_width=True)
                with col2:
                    if 'dream_counts' in viz:
                        st.plotly_chart(viz['dream_counts'], use_container_width=True)
                
                if 'scatter' in viz:
                    st.plotly_chart(viz['scatter'], use_container_width=True)
        
        with tab2:
            st.header("Thematic Analysis")
            st.info("Using enhanced theme detection with optimized data")
            
            # Create visualizations
            thematic_viz = analyzer.create_thematic_visualizations()
            
            st.plotly_chart(thematic_viz['heatmap'], use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(thematic_viz['english_dominance'], use_container_width=True)
            with col2:
                st.plotly_chart(thematic_viz['variation'], use_container_width=True)
        
        with tab3:
            st.header("Statistical Analysis")
            st.info("Detailed statistical analysis of optimized dream generation")
            
            # Show statistical data in tables
            st.subheader("Language Statistics")
            
            stats_data = []
            for lang, data in available_data['languages'].items():
                latest = data['latest_session']
                stats_data.append({
                    'Language': lang.title(),
                    'Dreams': data['total_dreams'],
                    'Success Rate (%)': latest['success_rate'],
                    'Total Calls': latest['total_calls'],
                    'Latest Session': latest['session_id']
                })
            
            df_stats = pd.DataFrame(stats_data)
            st.dataframe(df_stats, use_container_width=True)
        
        with tab4:
            st.header("Data Explorer")
            st.info("Browse and explore individual dreams from optimized data")
            
            # Language selector
            selected_lang = st.selectbox(
                "Select Language",
                options=list(available_data['languages'].keys()),
                format_func=str.title
            )
            
            if selected_lang:
                lang_data = available_data['languages'][selected_lang]
                st.subheader(f"{selected_lang.title()} Dreams")
                
                # Load dreams for selected language
                dreams_data = analyzer.load_dream_data_for_analysis()
                
                if selected_lang in dreams_data:
                    dreams = dreams_data[selected_lang]
                    st.write(f"Showing {len(dreams)} dreams")
                    
                    # Dream selector
                    dream_idx = st.selectbox(
                        "Select Dream",
                        range(len(dreams)),
                        format_func=lambda x: f"Dream {x+1}"
                    )
                    
                    if dream_idx is not None:
                        st.subheader(f"Dream {dream_idx + 1}")
                        st.text_area(
                            "Dream Content",
                            dreams[dream_idx],
                            height=200,
                            disabled=True
                        )
                        
                        # Basic stats
                        dream_text = dreams[dream_idx]
                        st.write(f"**Word Count:** {len(dream_text.split())}")
                        st.write(f"**Character Count:** {len(dream_text)}")
    else:
        st.error("❌ No dream data found in logs_optimized_v2 directory")
        st.info("Please ensure the logs_optimized_v2 directory contains processed dream data with the following structure:")
        st.code("""
logs_optimized_v2/
├── english/gpt-4o/session_*/dreams.csv
├── basque/gpt-4o/session_*/dreams.csv
├── hebrew/gpt-4o/session_*/dreams.csv
├── serbian/gpt-4o/session_*/dreams.csv
└── slovenian/gpt-4o/session_*/dreams.csv
        """)

if __name__ == "__main__":
    main() 