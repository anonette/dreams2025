"""
Visualization and reporting tools for dream research results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class DreamReportGenerator:
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def create_theme_comparison_chart(self, cultural_patterns: Dict) -> go.Figure:
        """Create comparison chart of themes across languages."""
        # Prepare data
        data = []
        
        # Handle empty or missing cultural_patterns
        if not cultural_patterns:
            # Create empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No cultural patterns data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(title="Theme Frequency Across Languages")
            return fig
        
        for language, patterns in cultural_patterns.items():
            # Check if patterns has the expected structure
            if not isinstance(patterns, dict) or 'top_themes' not in patterns:
                continue
                
            top_themes = patterns['top_themes']
            if not isinstance(top_themes, list):
                continue
                
            for theme_item in top_themes:
                # Handle both tuple (theme, count) and dict formats
                if isinstance(theme_item, tuple) and len(theme_item) == 2:
                    theme, count = theme_item
                elif isinstance(theme_item, dict) and 'theme' in theme_item and 'count' in theme_item:
                    theme = theme_item['theme']
                    count = theme_item['count']
                else:
                    continue
                    
                data.append({
                    'Language': language,
                    'Theme': theme,
                    'Count': count
                })
        
        if not data:
            # Create empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No theme data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(title="Theme Frequency Across Languages")
            return fig
        
        df = pd.DataFrame(data)
        
        # Create heatmap
        pivot_df = df.pivot(index='Theme', columns='Language', values='Count').fillna(0)
        
        fig = px.imshow(
            pivot_df,
            title="Theme Frequency Across Languages",
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            xaxis_title="Language",
            yaxis_title="Theme",
            height=600
        )
        
        return fig
    
    def create_cluster_visualization(self, clustering_results: Dict) -> go.Figure:
        """Visualize dream clusters."""
        clusters = clustering_results.get('clusters', [])
        
        if not clusters:
            # Create empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No clustering data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(title="Dream Clusters by Representative Terms")
            return fig
        
        # Create subplots for each cluster
        fig = make_subplots(
            rows=len(clusters), cols=1,
            subplot_titles=[f"Cluster {c.get('cluster_id', i)}" for i, c in enumerate(clusters)],
            vertical_spacing=0.1
        )
        
        for i, cluster in enumerate(clusters):
            # Get representative terms, fallback to characteristics if available
            terms = cluster.get('representative_terms', [])
            if not terms:
                terms = cluster.get('characteristics', [])
            if not terms:
                terms = [f"Term {j+1}" for j in range(5)]  # Fallback
            
            terms = terms[:10]  # Limit to 10 terms
            counts = list(range(len(terms), 0, -1))  # Simple count for visualization
            
            fig.add_trace(
                go.Bar(
                    x=terms,
                    y=counts,
                    name=f"Cluster {cluster.get('cluster_id', i)}",
                    showlegend=False
                ),
                row=i+1, col=1
            )
        
        fig.update_layout(
            title="Dream Clusters by Representative Terms",
            height=300 * len(clusters)
        )
        
        return fig
    
    def create_language_comparison_dashboard(self, analysis_results: Dict) -> go.Figure:
        """Create comprehensive dashboard comparing languages."""
        cultural_patterns = analysis_results.get('cultural_patterns', {})
        
        if not cultural_patterns:
            # Create empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No cultural patterns data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(title="Cross-Linguistic Dream Analysis Dashboard")
            return fig
        
        # Prepare data
        languages = list(cultural_patterns.keys())
        dream_counts = [patterns.get('total_dreams', 0) for patterns in cultural_patterns.values()]
        theme_counts = [patterns.get('unique_themes', 0) for patterns in cultural_patterns.values()]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Dreams Generated per Language",
                "Unique Themes per Language",
                "Theme Diversity vs Dream Count",
                "Language Summary"
            ]
        )
        
        # Plot 1: Dreams per language
        fig.add_trace(
            go.Bar(x=languages, y=dream_counts, name="Dreams"),
            row=1, col=1
        )
        
        # Plot 2: Themes per language
        fig.add_trace(
            go.Bar(x=languages, y=theme_counts, name="Themes"),
            row=1, col=2
        )
        
        # Plot 3: Scatter plot
        fig.add_trace(
            go.Scatter(
                x=dream_counts,
                y=theme_counts,
                mode='markers+text',
                text=languages,
                name="Diversity"
            ),
            row=2, col=1
        )
        
        # Plot 4: Summary as text annotation instead of table
        avg_themes_per_dream = [round(t/d, 2) if d > 0 else 0 for t, d in zip(theme_counts, dream_counts)]
        summary_text = "<br>".join([
            f"{lang}: {dreams} dreams, {themes} themes, {avg:.2f} avg"
            for lang, dreams, themes, avg in zip(languages, dream_counts, theme_counts, avg_themes_per_dream)
        ])
        
        fig.add_annotation(
            text=summary_text,
            xref="x4", yref="y4",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=10),
            align="left"
        )
        
        fig.update_layout(height=800, title="Cross-Linguistic Dream Analysis Dashboard")
        
        return fig 