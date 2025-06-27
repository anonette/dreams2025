"""
Simple NLP-based dream analysis for thematic extraction and clustering.
Focus: Basic thematic analysis and clustering of dream motives.
"""

import re
from typing import List, Dict, Any
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


class NLPDreamAnalyzer:
    """Simple NLP analyzer for dream thematic analysis and clustering."""
    
    def __init__(self):
        # Common stop words across languages
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
            'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
        }
    
    def extract_themes(self, dream_text: str) -> List[str]:
        """Extract thematic words from dream text."""
        # Extract meaningful words (3+ characters, not numbers)
        words = re.findall(r'\b\w{3,}\b', dream_text.lower())
        
        # Filter themes: remove stop words and numbers
        themes = [
            word for word in words 
            if word not in self.stop_words and not word.isdigit()
        ]
        
        return themes
    
    def analyze_dream_motives(self, dreams: List[str]) -> Dict[str, Any]:
        """Analyze dream motives across multiple dreams."""
        all_themes = []
        
        # Extract themes from all dreams
        for dream in dreams:
            if dream and dream.strip():
                themes = self.extract_themes(dream)
                all_themes.extend(themes)
        
        # Count theme frequencies
        theme_counts = Counter(all_themes)
        top_motives = theme_counts.most_common(20)
        
        return {
            'top_motives': top_motives,
            'total_themes': len(all_themes),
            'unique_themes': len(theme_counts),
            'diversity_score': len(theme_counts) / len(all_themes) if all_themes else 0
        }
    
    def cluster_dreams(self, dreams: List[str], n_clusters: int = 5) -> Dict[str, Any]:
        """Cluster dreams by thematic similarity."""
        # Filter valid dreams
        valid_dreams = [dream for dream in dreams if dream and dream.strip()]
        
        if len(valid_dreams) < 2:
            return {
                'clusters': [],
                'cluster_labels': [],
                'n_clusters': 0,
                'message': 'Need at least 2 dreams for clustering'
            }
        
        # Adjust clusters if needed
        n_clusters = min(n_clusters, len(valid_dreams))
        
        # Vectorize dreams using TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        
        try:
            X = vectorizer.fit_transform(valid_dreams)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            # Extract cluster characteristics
            feature_names = vectorizer.get_feature_names_out()
            cluster_centers = kmeans.cluster_centers_
            
            clusters = []
            for i in range(n_clusters):
                # Get top terms for this cluster
                center = cluster_centers[i]
                top_indices = np.argsort(center)[-10:][::-1]
                top_terms = [feature_names[idx] for idx in top_indices if center[idx] > 0]
                
                # Get dreams in this cluster
                cluster_dreams = [
                    valid_dreams[j] for j, label in enumerate(cluster_labels) if label == i
                ]
                
                clusters.append({
                    'cluster_id': i,
                    'size': len(cluster_dreams),
                    'key_motives': top_terms[:5],
                    'sample_dreams': cluster_dreams[:2],  # Show first 2 as examples
                    'total_dreams': len(cluster_dreams)
                })
            
            return {
                'clusters': clusters,
                'cluster_labels': cluster_labels.tolist(),
                'n_clusters': n_clusters,
                'total_dreams': len(valid_dreams)
            }
            
        except Exception as e:
            return {
                'clusters': [],
                'cluster_labels': [],
                'n_clusters': 0,
                'error': str(e)
            }
    
    def compare_languages(self, dreams_by_language: Dict[str, List[str]]) -> Dict[str, Any]:
        """Compare dream motives across languages."""
        language_analysis = {}
        
        for language, dreams in dreams_by_language.items():
            if dreams:
                motives = self.analyze_dream_motives(dreams)
                language_analysis[language] = {
                    'top_motives': motives['top_motives'][:10],
                    'total_dreams': len(dreams),
                    'theme_diversity': motives['diversity_score']
                }
        
        return language_analysis
