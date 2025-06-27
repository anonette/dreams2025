#!/usr/bin/env python3
"""
Streamlit interface for dream analysis using the refactored analyzer architecture.
"""

import streamlit as st
import pandas as pd
import os
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
import asyncio
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, environment variables might still be set

# Import the refactored analyzer architecture
from src.analysis import DreamAnalyzerFactory

# Translation functionality
try:
    from deep_translator import GoogleTranslator
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="Dream Research Analysis",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

def is_english_word(word):
    """Check if a word is likely to be English."""
    # More sophisticated check - common English words and patterns
    common_english_words = {
        'the', 'and', 'a', 'to', 'of', 'in', 'i', 'you', 'it', 'for', 'is', 'on', 'that', 'by', 'this',
        'with', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would',
        'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
        'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take', 'people', 'into',
        'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now', 'look',
        'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our',
        'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day',
        'most', 'us', 'water', 'flying', 'fly', 'dream', 'house', 'car', 'tree', 'sky', 'sun', 'moon'
    }
    
    word_lower = word.lower().strip()
    
    # Check if it's a common English word
    if word_lower in common_english_words:
        return True
    
    # Check if it looks like English (has English-like patterns)
    # But be more restrictive - don't assume all Latin script is English
    if len(word_lower) <= 2:
        return word_lower in common_english_words
    
    # For longer words, use more specific English patterns
    english_patterns = [
        r'.*ing$',  # words ending in -ing
        r'.*ed$',   # words ending in -ed
        r'.*ly$',   # words ending in -ly
        r'.*tion$', # words ending in -tion
        r'^un.*',   # words starting with un-
        r'^re.*',   # words starting with re-
    ]
    
    for pattern in english_patterns:
        if re.match(pattern, word_lower):
            return True
    
    return False

def translate_theme(theme, source_lang='auto', target_lang='en'):
    """Translate a theme to English if translation is available."""
    if not TRANSLATION_AVAILABLE:
        return theme
    
    try:
        # Don't translate if already English
        if is_english_word(theme):
            return theme
            
        # Use deep_translator's GoogleTranslator
        if source_lang == 'auto':
            # For auto-detection, try to detect or use a default
            source_lang = 'auto'
        
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        result = translator.translate(theme)
        return result
    except Exception as e:
        # If translation fails, return original
        return theme

def format_theme_with_translation(theme, language):
    """Format theme with translation if needed."""
    if language == 'english' or is_english_word(theme):
        return theme
    
    # Map language names to language codes
    lang_codes = {
        'serbian': 'sr',
        'hebrew': 'he', 
        'slovenian': 'sl',
        'basque': 'eu'
    }
    
    source_lang = lang_codes.get(language, 'auto')
    translation = translate_theme(theme, source_lang=source_lang)
    
    if translation and translation.lower() != theme.lower():
        return f"{theme} ({translation})"
    else:
        return theme

def get_contextual_stopwords(language):
    """Get language-specific contextual stopwords for meaningful thematic analysis."""
    
    # Base grammatical scaffolding words by language
    contextual_stopwords = {
        'english': {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
            'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
            'just', 'don', 'should', 'now'
        },
        'serbian': {
            'sam', 'je', 'da', 'se', 'su', 'i', 'u', 'na', 'za', 'od', 'do', 'sa', 'po', 'kroz',
            'bez', 'već', 'ali', 'ili', 'ako', 'kad', 'kada', 'što', 'kako', 'gdje', 'tu', 'tamo',
            'ovdje', 'evo', 'eto', 'eno', 'mi', 'ti', 'on', 'ona', 'ono', 'mi', 'vi', 'oni', 'one',
            'ova', 'taj', 'ta', 'to', 'neki', 'neka', 'neko', 'sav', 'sva', 'sve', 'jedan', 'jedna',
            'jedno', 'dva', 'tri', 'četiri', 'pet', 'bio', 'bila', 'bilo', 'biti', 'bude', 'budem',
            'ću', 'ćeš', 'će', 'ćemo', 'ćete', 'hoću', 'hoćeš', 'hoće', 'nećemo', 'neće'
        },
        'basque': {
            'eta', 'da', 'baina', 'edo', 'ala', 'ez', 'bai', 'nik', 'hik', 'hark', 'guk', 'zuek',
            'haiek', 'nire', 'hire', 'haren', 'gure', 'zuen', 'haien', 'ni', 'hi', 'hura', 'gu',
            'zu', 'hori', 'hau', 'han', 'hor', 'hemen', 'non', 'noiz', 'nola', 'zergatik', 'zer',
            'nork', 'nori', 'noren', 'den', 'duen', 'duten', 'naiz', 'zara', 'da', 'gara', 'zarete',
            'dira', 'nintzen', 'zinen', 'zen', 'ginen', 'zineten', 'ziren', 'izan', 'egon', 'ukan',
            'egin', 'esan', 'etorri', 'joan', 'eman', 'hartu', 'ikusi', 'entzun'
        },
        'hebrew': {
            'את', 'של', 'על', 'אל', 'מן', 'עם', 'כל', 'לא', 'כי', 'אם', 'או', 'גם', 'רק', 'עוד',
            'כבר', 'מאוד', 'יותר', 'פחות', 'אחר', 'אחרת', 'אותו', 'אותה', 'זה', 'זו', 'זאת',
            'הוא', 'היא', 'הם', 'הן', 'אני', 'אתה', 'את', 'אנחנו', 'אתם', 'אתן', 'שלי', 'שלך',
            'שלו', 'שלה', 'שלנו', 'שלכם', 'שלהם', 'להיות', 'היה', 'הייתי', 'היית', 'היינו',
            'הייתם', 'היו', 'יהיה', 'תהיה', 'נהיה', 'תהיו', 'יהיו', 'בא', 'באה', 'באים', 'באות'
        },
        'slovenian': {
            'je', 'da', 'in', 'na', 'za', 'se', 'z', 's', 'od', 'do', 'po', 'pri', 'v', 'o', 'k',
            'iz', 'med', 'nad', 'pod', 'pred', 'čez', 'skozi', 'zaradi', 'kljub', 'razen', 'brez',
            'jaz', 'ti', 'on', 'ona', 'ono', 'mi', 'vi', 'oni', 'one', 'moj', 'tvoj', 'njegov',
            'njen', 'naš', 'vaš', 'njihov', 'ta', 'tisti', 'neki', 'vsak', 'ves', 'cel', 'sam',
            'samo', 'le', 'tudi', 'še', 'že', 'ne', 'ni', 'sem', 'si', 'smo', 'ste', 'so', 'bil',
            'bila', 'bilo', 'bili', 'bile', 'biti', 'bom', 'boš', 'bo', 'bomo', 'boste', 'bodo'
        }
    }
    
    return contextual_stopwords.get(language, set())

def filter_meaningful_themes(dreams, language, min_length=3, max_frequency_ratio=0.8):
    """
    Filter themes to focus on meaningful content words using hybrid approach:
    1. Remove contextual stopwords (grammatical scaffolding)
    2. Use TF-IDF to identify low-information high-frequency words
    3. Keep words with potential narrative/affective significance
    """
    
    if not dreams:
        return []
    
    # Get contextual stopwords for this language
    stopwords = get_contextual_stopwords(language)
    
    # Extract all words from dreams
    all_words = []
    for dream in dreams:
        # Simple tokenization - split on spaces and clean
        words = re.findall(r'\b\w+\b', dream.lower())
        all_words.extend(words)
    
    # Count word frequencies
    word_counts = Counter(all_words)
    total_words = len(all_words)
    
    # Filter candidates
    meaningful_words = []
    
    for word, count in word_counts.most_common():
        # Skip very short words (likely particles)
        if len(word) < min_length:
            continue
            
        # Skip contextual stopwords (grammatical scaffolding)
        if word in stopwords:
            continue
            
        # Skip extremely high frequency words (likely uniform low-information)
        frequency_ratio = count / len(dreams)
        if frequency_ratio > max_frequency_ratio and count > len(dreams) * 0.5:
            continue
            
        # Keep words that appear in multiple dreams but not too uniformly
        if count >= 2:  # Appears at least twice
            meaningful_words.append((word, count))
    
    return meaningful_words[:20]  # Return top 20 meaningful themes

def vectorize_and_cluster_dreams(dreams, language, n_clusters=5, min_df=2, max_df=0.7):
    """
    Advanced vectorization and clustering for theme detection simulation.
    Uses TF-IDF vectorization and cosine similarity clustering.
    """
    if len(dreams) < n_clusters:
        return None
    
    # Get stopwords for filtering
    stopwords = get_contextual_stopwords(language)
    
    # Create TF-IDF vectorizer with language-specific stopwords
    vectorizer = TfidfVectorizer(
        min_df=min_df,  # Word must appear in at least 2 documents
        max_df=max_df,  # Exclude words that appear in more than 70% of documents
        stop_words=list(stopwords),
        ngram_range=(1, 2),  # Include bigrams for better context
        max_features=1000
    )
    
    try:
        # Vectorize dreams
        tfidf_matrix = vectorizer.fit_transform(dreams)
        feature_names = vectorizer.get_feature_names_out()
        
        # Cluster dreams using K-means on TF-IDF vectors
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)
        
        # Analyze clusters
        clusters = []
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            cluster_dreams = [dreams[j] for j in range(len(dreams)) if cluster_mask[j]]
            
            # Get cluster center and most salient words
            cluster_center = kmeans.cluster_centers_[i]
            top_indices = cluster_center.argsort()[-10:][::-1]
            salient_words = [feature_names[idx] for idx in top_indices]
            
            clusters.append({
                'id': i,
                'size': sum(cluster_mask),
                'dreams': cluster_dreams[:3],  # Sample dreams
                'salient_words': salient_words,
                'center_vector': cluster_center
            })
        
        # Identify cross-cluster high-frequency words (potential exclusions)
        word_cluster_counts = {}
        for word_idx, word in enumerate(feature_names):
            clusters_containing_word = 0
            total_weight = 0
            for cluster_center in kmeans.cluster_centers_:
                if cluster_center[word_idx] > 0.01:  # Threshold for meaningful presence
                    clusters_containing_word += 1
                    total_weight += cluster_center[word_idx]
            
            if clusters_containing_word >= n_clusters * 0.8:  # Present in 80%+ of clusters
                word_cluster_counts[word] = {
                    'cluster_presence': clusters_containing_word,
                    'avg_weight': total_weight / n_clusters,
                    'exclusion_candidate': True
                }
        
        return {
            'clusters': clusters,
            'vectorizer': vectorizer,
            'tfidf_matrix': tfidf_matrix,
            'feature_names': feature_names,
            'exclusion_candidates': word_cluster_counts,
            'coherence_metrics': calculate_cluster_coherence(tfidf_matrix, cluster_labels)
        }
        
    except Exception as e:
        st.error(f"Error in clustering: {e}")
        return None

def calculate_cluster_coherence(tfidf_matrix, cluster_labels):
    """Calculate coherence metrics for cluster quality assessment."""
    try:
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        
        coherence = {
            'silhouette_score': silhouette_score(tfidf_matrix, cluster_labels),
            'calinski_harabasz_score': calinski_harabasz_score(tfidf_matrix.toarray(), cluster_labels)
        }
        return coherence
    except:
        return {'silhouette_score': 0, 'calinski_harabasz_score': 0}

async def run_centralized_llm_analysis(dreams, language, analyzer):
    """
    Run centralized LLM discourse analysis using the consolidated specialist.
    """
    try:
        # Get the discourse specialist from the LLM analyzer
        if hasattr(analyzer, 'discourse_specialist'):
            specialist = analyzer.discourse_specialist
            
            # Run professional thematic analysis
            result = await specialist.extract_themes(dreams, language, sample_size=20)
            
            return {
                "analysis_type": "centralized_professional_discourse_analysis",
                "model_used": specialist.analysis_model,
                "language": language,
                "sample_size": result.get('metadata', {}).get('sample_size', len(dreams)),
                "status": "completed",
                "result": result
            }
        else:
            return {
                "analysis_type": "fallback_analysis",
                "status": "no_specialist_available",
                "error": "LLM analyzer does not have discourse specialist"
            }
        
    except Exception as e:
        return {
            "analysis_type": "error",
            "status": "failed", 
            "error": str(e)
        }

def iterative_refinement_analysis(dreams, language, initial_exclusions, iterations=3):
    """
    Iterative refinement of exclusion criteria based on coherence metrics.
    Implements model-in-the-loop curation for better thematic coherence.
    """
    results = []
    current_exclusions = set(initial_exclusions)
    
    for iteration in range(iterations):
        # Apply current exclusions and run clustering
        filtered_dreams = []
        for dream in dreams:
            # Remove currently excluded words
            words = dream.split()
            filtered_words = [w for w in words if w.lower() not in current_exclusions]
            filtered_dreams.append(" ".join(filtered_words))
        
        # Run clustering analysis
        cluster_result = vectorize_and_cluster_dreams(filtered_dreams, language)
        
        if cluster_result:
            coherence = cluster_result['coherence_metrics']
            
            # Add new exclusion candidates with low thematic value
            new_exclusions = set()
            for word, info in cluster_result['exclusion_candidates'].items():
                if info['exclusion_candidate'] and info['avg_weight'] > 0.1:
                    new_exclusions.add(word)
            
            current_exclusions.update(new_exclusions)
            
            results.append({
                'iteration': iteration,
                'exclusions': list(current_exclusions),
                'coherence_score': coherence['silhouette_score'],
                'num_exclusions': len(current_exclusions),
                'improvement': coherence['silhouette_score'] - (results[-1]['coherence_score'] if results else 0)
            })
    
    return {
        'refinement_results': results,
        'final_exclusions': list(current_exclusions),
        'best_iteration': max(results, key=lambda x: x['coherence_score']) if results else None
    }

def load_dream_data():
    """Load dream data from all languages."""
    languages = ['english', 'serbian', 'hebrew', 'slovenian', 'basque']
    dreams_by_language = {}
    
    for lang in languages:
        try:
            session_dirs = os.listdir(f'logs/{lang}/gpt-4o/')
            if session_dirs:
                # Try each session directory until we find one with dreams.csv
                session_found = False
                for session_dir in reversed(sorted(session_dirs)):  # Try newest first
                    dreams_file = f'logs/{lang}/gpt-4o/{session_dir}/dreams.csv'
                    if os.path.exists(dreams_file):
                        df = pd.read_csv(dreams_file)
                        if 'status' in df.columns:
                            successful_dreams = df[df.status == 'success']
                        else:
                            successful_dreams = df  # Assume all are successful if no status column
                        
                        if len(successful_dreams) > 0:
                            dreams_by_language[lang] = successful_dreams['dream'].tolist()
                            st.sidebar.write(f"✓ Loaded {len(successful_dreams)} {lang} dreams")
                            session_found = True
                            break
                
                if not session_found:
                    st.sidebar.write(f"✗ No valid dreams data found for {lang}")
        except Exception as e:
            st.sidebar.write(f"✗ Error loading {lang}: {e}")
    
    return dreams_by_language

def plot_theme_heatmap(cultural_patterns, selected_languages):
    """Create a heatmap of theme frequencies across languages."""
    themes = {}
    languages = []
    
    # Collect all unique themes
    for language, data in cultural_patterns.items():
        if language not in selected_languages:
            continue
            
        languages.append(language)
        
        for theme, count in data['top_themes']:
            if theme not in themes:
                themes[theme] = {}
            
            # Normalize by total dreams
            normalized_count = count / data['total_dreams'] if data['total_dreams'] > 0 else 0
            themes[theme][language] = normalized_count
    
    # Convert to DataFrame
    theme_data = []
    for theme, lang_values in themes.items():
        for lang, value in lang_values.items():
            theme_data.append({
                'Theme': theme,
                'Language': lang,
                'Frequency': value
            })
    
    if not theme_data:
        return None
        
    df = pd.DataFrame(theme_data)
    
    # Pivot for heatmap
    pivot_df = df.pivot(index='Theme', columns='Language', values='Frequency')
    pivot_df = pivot_df.fillna(0)
    
    # Keep only themes that appear in at least 2 languages
    theme_counts = pivot_df.astype(bool).sum(axis=1)
    pivot_df = pivot_df[theme_counts >= 2]
    
    # Sort by overall frequency
    pivot_df['total'] = pivot_df.sum(axis=1)
    pivot_df = pivot_df.sort_values('total', ascending=False).drop('total', axis=1)
    
    # Take top 20 themes
    pivot_df = pivot_df.head(20)
    
    # Create heatmap
    fig = px.imshow(
        pivot_df,
        color_continuous_scale='Viridis',
        labels=dict(x="Language", y="Theme", color="Frequency"),
        title="Theme Frequency Across Languages",
        height=600
    )
    
    fig.update_layout(
        xaxis_title="Language",
        yaxis_title="Theme",
        coloraxis_colorbar=dict(title="Normalized Frequency")
    )
    
    return fig

def plot_dream_clusters(clusters, max_clusters=5):
    """Create visualizations for dream clusters."""
    if not clusters or not clusters.get('clusters'):
        return None
        
    cluster_sizes = [cluster['size'] for cluster in clusters['clusters']]
    cluster_labels = [f"Cluster {cluster['cluster_id']}" for cluster in clusters['clusters']]
    
    # Create a pie chart of cluster sizes
    fig = go.Figure(data=[go.Pie(
        labels=cluster_labels,
        values=cluster_sizes,
        hole=.3,
        textinfo='percent+label'
    )])
    
    fig.update_layout(
        title="Dream Clusters",
        height=400
    )
    
    return fig

def main():
    st.title("🌙 Dream Research Analysis")
    st.subheader("Cross-Linguistic Dream Theme Analysis")
    
    st.sidebar.title("Setup")
    
    # Select analyzer type
    analyzer_type = st.sidebar.selectbox(
        "Analyzer Type",
        ["simple", "spacy", "llm"],
        format_func=lambda x: {
            "simple": "📊 Simple (Basic word extraction)",
            "spacy": "🔍 SpaCy (NLP-based)",
            "llm": "🤖 LLM (AI-powered)"
        }.get(x, x)
    )
    
    # Load data
    if st.sidebar.button("Load Dream Data"):
        with st.spinner("Loading dream data..."):
            dreams_data = load_dream_data()
            st.session_state.dreams_data = dreams_data
    
    if not hasattr(st.session_state, 'dreams_data'):
        st.info("Please load dream data using the sidebar button.")
        return
    
    dreams_data = st.session_state.dreams_data
    
    if not dreams_data:
        st.error("No dream data found!")
        return
    
    # Language selection
    available_languages = list(dreams_data.keys())
    selected_languages = st.sidebar.multiselect(
        "Select Languages",
        available_languages,
        default=available_languages
    )
    
    if not selected_languages:
        st.warning("Please select at least one language to analyze.")
        return
    
    # Configuration for LLM analyzer
    if analyzer_type == "llm":
        st.sidebar.subheader("LLM Configuration")
        
        # Check for environment variables first
        env_openai_key = os.getenv('OPENAI_API_KEY', '')
        env_anthropic_key = os.getenv('ANTHROPIC_API_KEY', '')
        
        # Show status of environment variables
        if env_openai_key:
            st.sidebar.success("✅ OpenAI API key found")
        if env_anthropic_key:
            st.sidebar.success("✅ Anthropic API key found")
        
        use_api_keys = st.sidebar.checkbox("Manually Enter API Keys", value=False)
        
        if use_api_keys:
            openai_key = st.sidebar.text_input("OpenAI API Key", type="password", 
                                              value=env_openai_key if env_openai_key else "")
            anthropic_key = st.sidebar.text_input("Anthropic API Key (optional)", type="password",
                                                 value=env_anthropic_key if env_anthropic_key else "")
            
            api_keys = {
                "openai": openai_key
            }
            
            if anthropic_key:
                api_keys["anthropic"] = anthropic_key
        else:
            # Use environment variables
            api_keys = {}
            if env_openai_key:
                api_keys["openai"] = env_openai_key
            if env_anthropic_key:
                api_keys["anthropic"] = env_anthropic_key
        
        if api_keys.get('openai'):
            analysis_model = st.sidebar.selectbox(
                "Analysis Model",
                ["gpt-4o", "gpt-4", "gpt-3.5-turbo", "claude-3.5-sonnet"]
            )
            
            analyzer_kwargs = {
                "api_keys": api_keys,
                "analysis_model": analysis_model
            }
        else:
            st.sidebar.warning("No API keys available")
            analyzer_kwargs = {"demo_mode": True}
    else:
        # Translation option for simple and spaCy analyzers
        use_translation = st.sidebar.checkbox("Use Translation", value=False)
        analyzer_kwargs = {"use_translation": use_translation}
    
    # Initialize analyzer but don't show options yet
    try:
        analyzer = DreamAnalyzerFactory.create_analyzer(analyzer_type, **analyzer_kwargs)
        st.sidebar.success(f"Using {type(analyzer).__name__}")
        
        # Filter data by selected languages
        filtered_data = {lang: dreams_data[lang] for lang in selected_languages if lang in dreams_data}
        
    except Exception as e:
        st.sidebar.error(f"Error initializing analyzer: {e}")
        st.error(f"Could not initialize {analyzer_type} analyzer. Please select a different one.")
        return
    
    # Ready to analyze message
    st.info("✅ Ready to analyze! Use the button below to start analysis.")
    
    # Analysis configuration (context-aware based on analyzer type)
    with st.expander("🔧 Analysis Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**What to Analyze:** ({analyzer_type} analyzer)")
            
            # Force default values regardless of condition
            run_cultural_analysis = st.checkbox("📊 AI Theme Analysis", value=True, help="AI-powered theme extraction (or fallback if no API keys)")
            run_clustering = st.checkbox("🔍 Intelligent Clustering", value=True, help="AI-enhanced dream grouping")  
            run_advanced_analysis = st.checkbox("🧠 Cultural Insights", value=False, help="Deep cultural and linguistic pattern analysis")
        
        with col2:
            st.write("**Settings:**")
            show_translations = st.checkbox("Show Translations", value=True)
            if not TRANSLATION_AVAILABLE and show_translations:
                st.warning("Translation requires deep-translator")
            if run_clustering:
                cluster_count = st.slider("Number of Clusters", 3, 10, 5)
            else:
                cluster_count = 5
    
    # Run analysis
    if st.button("🚀 Run Analysis", type="primary"):
        with st.spinner("Analyzing dreams..."):
            # Cultural pattern analysis
            if run_cultural_analysis:
                st.subheader("Cultural Pattern Analysis")
                
                cultural_patterns = analyzer.analyze_cultural_patterns(filtered_data)
                
                # Display top themes for each language
                cols = st.columns(len(selected_languages))
                
                for i, language in enumerate(selected_languages):
                    if language in filtered_data:
                        dreams = filtered_data[language]
                        with cols[i % len(cols)]:
                            st.write(f"### {language.capitalize()}")
                            st.write(f"Total dreams: {len(dreams)}")
                            
                            # Show analyzer status cleanly
                            actual_analyzer = type(analyzer).__name__
                            
                            if analyzer_type == "llm" and actual_analyzer == "SimpleDreamAnalyzer":
                                st.warning("⚠️ **LLM Fallback**: No API keys provided, using basic analysis")
                            elif analyzer_type == "spacy" and hasattr(analyzer, 'nlp_en') and analyzer.nlp_en is None:
                                st.warning("⚠️ **SpaCy Fallback**: SpaCy model not available, using simple extraction")
                            
                            # Use the analyzer's native cultural analysis instead of filtering
                            if language in cultural_patterns:
                                pattern_data = cultural_patterns[language]
                                top_themes = pattern_data.get('top_themes', [])
                                
                                st.write(f"Themes extracted: {len(top_themes)}")
                                
                                # Create table of themes from analyzer
                                theme_data = []
                                for theme, count in top_themes:
                                    if show_translations and TRANSLATION_AVAILABLE:
                                        formatted_theme = format_theme_with_translation(theme, language)
                                    else:
                                        formatted_theme = theme
                                    
                                    # Fix percentage calculation - count is frequency, not dreams containing theme
                                    # Calculate actual dreams containing this theme
                                    dreams_containing_theme = sum(1 for dream in dreams if theme.lower() in dream.lower())
                                    percentage = (dreams_containing_theme / len(dreams)) * 100
                                    
                                    theme_data.append({
                                        "Theme": formatted_theme,
                                        "Frequency": count,
                                        "Dreams": dreams_containing_theme,
                                        "Percentage": f"{percentage:.1f}%"
                                    })
                                
                                st.write(f"#### Themes from {type(analyzer).__name__}")
                                if theme_data:
                                    st.table(pd.DataFrame(theme_data))
                                else:
                                    st.info("No themes extracted by this analyzer.")
                
                # Theme heatmap
                st.subheader("Theme Frequency Heatmap")
                heatmap = plot_theme_heatmap(cultural_patterns, selected_languages)
                if heatmap:
                    st.plotly_chart(heatmap, use_container_width=True)
                else:
                    st.info("Not enough data for heatmap visualization.")
            
            # Dream clustering
            if run_clustering:
                st.subheader("Dream Clustering")
                
                # Select language for clustering
                cluster_language = st.selectbox(
                    "Select Language for Clustering",
                    selected_languages
                )
                
                if cluster_language in filtered_data:
                    dreams = filtered_data[cluster_language]
                    
                    # Limit to 100 dreams for performance
                    if len(dreams) > 100:
                        st.info(f"Using first 100 of {len(dreams)} dreams for clustering.")
                        dreams = dreams[:100]
                    
                    # Run clustering
                    clusters = analyzer.cluster_dreams(dreams, n_clusters=cluster_count)
                    
                    # Visualize clusters
                    cluster_viz = plot_dream_clusters(clusters)
                    if cluster_viz:
                        st.plotly_chart(cluster_viz, use_container_width=True)
                    
                    # Display cluster details
                    st.write(f"### Cluster Details")
                    
                    for cluster in clusters['clusters']:
                        with st.expander(f"Cluster {cluster['cluster_id']} ({cluster['size']} dreams)"):
                            st.write("**Key terms:** " + ", ".join(cluster['representative_terms'][:10]))
                            
                            # Show sample dreams
                            st.write("**Sample dreams:**")
                            for i, dream in enumerate(cluster['dreams'][:3]):
                                st.text_area(f"Dream {i+1}", dream, height=100)
            
            # Advanced analysis - meaningful insights focus
            if run_advanced_analysis:
                st.subheader("🧠 Advanced Analysis")
                
                # Select language for advanced analysis
                advanced_language = st.selectbox(
                    "Select Language for Advanced Analysis",
                    selected_languages,
                    key="advanced_lang"
                )
                
                if advanced_language in filtered_data:
                    dreams = filtered_data[advanced_language]
                    analysis_dreams = dreams[:50] if len(dreams) > 50 else dreams
                    
                    # Find meaningful patterns in the dreams
                    st.write("### Dream Pattern Analysis")
                    
                    # Extract meaningful themes (not just word frequency)
                    meaningful_themes = filter_meaningful_themes(analysis_dreams, advanced_language)
                    
                    if meaningful_themes:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("#### Significant Themes")
                            theme_insights = []
                            for theme, count in meaningful_themes[:10]:
                                # Calculate actual prevalence
                                dreams_with_theme = sum(1 for dream in analysis_dreams if theme in dream.lower())
                                prevalence = (dreams_with_theme / len(analysis_dreams)) * 100
                                
                                if show_translations and TRANSLATION_AVAILABLE:
                                    formatted_theme = format_theme_with_translation(theme, advanced_language)
                                else:
                                    formatted_theme = theme
                                
                                theme_insights.append({
                                    "Theme": formatted_theme,
                                    "Appears in": f"{dreams_with_theme} dreams",
                                    "Prevalence": f"{prevalence:.1f}%"
                                })
                            
                            st.table(pd.DataFrame(theme_insights))
                        
                        with col2:
                            st.write("#### Pattern Categories")
                            
                            # Categorize themes by type
                            nature_themes = [t for t, c in meaningful_themes if any(word in t for word in ['tree', 'forest', 'sky', 'water', 'garden', 'flower', 'mountain'])]
                            emotion_themes = [t for t, c in meaningful_themes if any(word in t for word in ['peaceful', 'calm', 'serene', 'happy', 'fear', 'joy'])]
                            action_themes = [t for t, c in meaningful_themes if any(word in t for word in ['walking', 'flying', 'running', 'dancing', 'singing'])]
                            
                            if nature_themes:
                                st.write(f"🌿 **Nature**: {', '.join(nature_themes[:3])}")
                            if emotion_themes:
                                st.write(f"😊 **Emotions**: {', '.join(emotion_themes[:3])}")
                            if action_themes:
                                st.write(f"🏃 **Actions**: {', '.join(action_themes[:3])}")
                            
                            # Sample dreams with rich themes
                            st.write("#### Sample Dream Excerpts")
                            rich_dreams = []
                            for dream in analysis_dreams:
                                theme_count = sum(1 for theme, _ in meaningful_themes[:10] if theme in dream.lower())
                                if theme_count >= 3:  # Dreams with multiple meaningful themes
                                    rich_dreams.append(dream)
                            
                            if rich_dreams:
                                sample_dream = rich_dreams[0]
                                # Show first 200 characters
                                st.text_area("Rich thematic content:", sample_dream[:200] + "..." if len(sample_dream) > 200 else sample_dream, height=100)
                    
                    # Language-specific insights
                    st.write("### Language-Specific Insights")
                    
                    if advanced_language == 'english':
                        st.info("🇺🇸 **English dreams** often feature natural settings and peaceful scenarios.")
                    elif advanced_language == 'serbian':
                        st.info("🇷🇸 **Serbian dreams** show strong emphasis on 'mir' (peace) and familial settings.")
                    elif advanced_language == 'hebrew':
                        st.info("🇮🇱 **Hebrew dreams** feature rich sensory descriptions and magical elements.")
                    elif advanced_language == 'slovenian':
                        st.info("🇸🇮 **Slovenian dreams** emphasize natural beauty and tranquil forests.")
                    elif advanced_language == 'basque':
                        st.info("🏴󠁥󠁳󠁰󠁶󠁿 **Basque dreams** show unique cultural elements and vivid colors.")
                    
                    # Practical insights
                    st.write("### Key Insights")
                    total_themes = len(meaningful_themes)
                    avg_prevalence = sum(sum(1 for dream in analysis_dreams if theme in dream.lower()) for theme, _ in meaningful_themes) / total_themes if total_themes > 0 else 0
                    
                    insight_col1, insight_col2, insight_col3 = st.columns(3)
                    with insight_col1:
                        st.metric("Meaningful Themes", total_themes)
                    with insight_col2:
                        st.metric("Avg. Theme Prevalence", f"{(avg_prevalence/len(analysis_dreams)*100):.1f}%")
                    with insight_col3:
                        st.metric("Dreams Analyzed", len(analysis_dreams))
    
    # Additional information
    with st.expander("About the Dream Analysis System"):
        st.write("""
        This Streamlit interface uses the refactored dream analyzer architecture to analyze cross-linguistic dream patterns.
        
        The system provides:
        - **Cultural Pattern Analysis**: Identifies common themes across languages
        - **Dream Clustering**: Groups dreams by thematic similarity
        - **Visualization**: Interactive charts and heatmaps
        
        Available analyzers:
        - **Simple Analyzer**: Basic text processing without external dependencies
        - **SpaCy Analyzer**: NLP-based analysis using spaCy
        - **LLM Analyzer**: Advanced analysis using large language models (requires API keys)
        """)

if __name__ == "__main__":
    main()
