"""
Statistical analysis module for cross-linguistic dream research.
Prepares data for multilevel modeling and mixed-effects logistic regression.
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import mixedlm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class DreamStatisticalAnalyzer:
    def __init__(self, logs_dir: str = 'logs'):
        self.logs_dir = logs_dir
        self.results_dir = 'statistical_results'
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def load_session_data(self, session_id: str) -> pd.DataFrame:
        """Load all API calls data for a specific session."""
        all_calls_file = f"{self.logs_dir}/all_api_calls_{session_id}.csv"
        all_dreams_file = f"{self.logs_dir}/all_dreams_{session_id}.csv"
        
        if not os.path.exists(all_calls_file):
            raise FileNotFoundError(f"Session data not found: {all_calls_file}")
        
        # Load API calls data
        api_calls_df = pd.read_csv(all_calls_file)
        dreams_df = pd.read_csv(all_dreams_file) if os.path.exists(all_dreams_file) else None
        
        logging.info(f"Loaded {len(api_calls_df)} API calls for session {session_id}")
        
        return api_calls_df, dreams_df
    
    def prepare_data_for_analysis(self, api_calls_df: pd.DataFrame, 
                                dreams_df: pd.DataFrame = None) -> pd.DataFrame:
        """Prepare data for statistical analysis with proper encoding."""
        
        # Start with API calls data
        df = api_calls_df.copy()
        
        # Convert timestamps to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])
        
        # Create time-based features
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['batch_sequence'] = df.groupby('batch_id').cumcount() + 1
        
        # Create binary features for dream content analysis
        df['success_binary'] = (df['status'] == 'success').astype(int)
        df['error_binary'] = (df['status'] == 'error').astype(int)
        df['filtered_binary'] = (df['status'] == 'filtered').astype(int)
        
        # Language encoding (categorical)
        df['language_factor'] = pd.Categorical(df['language'])
        
        # Script encoding
        df['script_factor'] = pd.Categorical(df['script'])
        
        # Create interaction terms
        df['language_script'] = df['language'] + '_' + df['script']
        
        # Duration features
        df['duration_log'] = np.log1p(df['duration_seconds'])
        
        # Batch-level features
        batch_stats = df.groupby('batch_id').agg({
            'success_binary': ['mean', 'sum', 'count'],
            'duration_seconds': ['mean', 'std'],
            'timestamp': ['min', 'max']
        }).reset_index()
        
        batch_stats.columns = ['batch_id', 'batch_success_rate', 'batch_success_count', 
                             'batch_size', 'batch_duration_mean', 'batch_duration_std',
                             'batch_start', 'batch_end']
        
        # Merge batch statistics
        df = df.merge(batch_stats, on='batch_id', how='left')
        
        # Create temporal features
        df['batch_duration_hours'] = (df['batch_end'] - df['batch_start']).dt.total_seconds() / 3600
        
        logging.info(f"Prepared data with {len(df)} observations and {len(df.columns)} features")
        
        return df
    
    def create_binary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary features for dream content analysis."""
        
        # Common dream themes (you can expand this list)
        dream_themes = {
            'flying': ['fly', 'flying', 'flight', 'soar', 'float'],
            'falling': ['fall', 'falling', 'drop', 'plunge'],
            'chase': ['chase', 'running', 'escape', 'pursue'],
            'water': ['water', 'swim', 'drown', 'ocean', 'river'],
            'family': ['family', 'mother', 'father', 'parent', 'child'],
            'work': ['work', 'office', 'job', 'meeting', 'colleague'],
            'anxiety': ['anxiety', 'worry', 'stress', 'fear', 'nervous'],
            'travel': ['travel', 'journey', 'trip', 'road', 'car']
        }
        
        # Create binary features for each theme
        for theme, keywords in dream_themes.items():
            pattern = '|'.join(keywords)
            df[f'contains_{theme}'] = df['dream'].str.contains(pattern, case=False, na=False).astype(int)
        
        # Create length-based features
        df['dream_length'] = df['dream'].str.len()
        df['dream_word_count'] = df['dream'].str.split().str.len()
        df['long_dream'] = (df['dream_word_count'] > 50).astype(int)
        
        # Create language-specific features
        for language in df['language'].unique():
            df[f'is_{language}'] = (df['language'] == language).astype(int)
        
        logging.info(f"Created {len(dream_themes)} theme-based binary features")
        
        return df
    
    def run_descriptive_statistics(self, df: pd.DataFrame) -> Dict:
        """Run descriptive statistics on the dataset."""
        
        stats = {}
        
        # Overall statistics
        stats['total_observations'] = len(df)
        stats['total_languages'] = df['language'].nunique()
        stats['total_batches'] = df['batch_id'].nunique()
        stats['total_users'] = df['user_id'].nunique()
        
        # Success rates
        stats['overall_success_rate'] = df['success_binary'].mean()
        stats['success_rate_by_language'] = df.groupby('language')['success_binary'].mean().to_dict()
        stats['success_rate_by_script'] = df.groupby('script')['success_binary'].mean().to_dict()
        
        # Duration statistics
        stats['mean_duration'] = df['duration_seconds'].mean()
        stats['duration_by_language'] = df.groupby('language')['duration_seconds'].mean().to_dict()
        
        # Temporal statistics
        stats['temporal_coverage_hours'] = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
        stats['calls_per_hour'] = len(df) / stats['temporal_coverage_hours']
        
        # Batch statistics
        stats['mean_batch_size'] = df.groupby('batch_id').size().mean()
        stats['batch_success_rates'] = df.groupby('batch_id')['success_binary'].mean().describe().to_dict()
        
        return stats
    
    def run_multilevel_analysis(self, df: pd.DataFrame) -> Dict:
        """Run multilevel modeling analysis."""
        
        results = {}
        
        # Filter for successful calls only
        success_df = df[df['success_binary'] == 1].copy()
        
        if len(success_df) == 0:
            logging.warning("No successful calls found for multilevel analysis")
            return results
        
        # 1. Basic logistic regression (language as fixed effect)
        try:
            # Model 1: Language effect on success rate
            model1 = smf.glm(
                formula="success_binary ~ language_factor",
                data=df,
                family=sm.families.Binomial()
            ).fit()
            
            results['language_fixed_effect'] = {
                'summary': model1.summary().as_text(),
                'aic': model1.aic,
                'bic': model1.bic,
                'pseudo_r2': model1.pseudo_rsquared()
            }
            
        except Exception as e:
            logging.error(f"Error in basic logistic regression: {e}")
            results['language_fixed_effect'] = {'error': str(e)}
        
        # 2. Mixed-effects model (batch as random effect)
        try:
            # Model 2: Mixed effects with batch as random intercept
            model2 = mixedlm(
                "success_binary ~ language_factor", 
                df, 
                groups=df["batch_id"]
            ).fit()
            
            results['mixed_effects_batch'] = {
                'summary': model2.summary().as_text(),
                'aic': model2.aic,
                'bic': model2.bic,
                'random_effects_variance': model2.cov_re.iloc[0, 0]
            }
            
        except Exception as e:
            logging.error(f"Error in mixed effects model: {e}")
            results['mixed_effects_batch'] = {'error': str(e)}
        
        # 3. Theme analysis (for successful dreams)
        theme_features = [col for col in success_df.columns if col.startswith('contains_')]
        
        if theme_features:
            theme_results = {}
            for theme in theme_features:
                try:
                    theme_model = smf.glm(
                        formula=f"{theme} ~ language_factor",
                        data=success_df,
                        family=sm.families.Binomial()
                    ).fit()
                    
                    theme_results[theme] = {
                        'summary': theme_model.summary().as_text(),
                        'aic': theme_model.aic,
                        'pseudo_r2': theme_model.pseudo_rsquared()
                    }
                    
                except Exception as e:
                    theme_results[theme] = {'error': str(e)}
            
            results['theme_analysis'] = theme_results
        
        return results
    
    def create_visualizations(self, df: pd.DataFrame, save_dir: str = None):
        """Create visualizations for the analysis."""
        
        if save_dir is None:
            save_dir = self.results_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Success rates by language
        plt.figure(figsize=(12, 6))
        success_by_lang = df.groupby('language')['success_binary'].mean().sort_values(ascending=False)
        success_by_lang.plot(kind='bar')
        plt.title('Success Rate by Language')
        plt.ylabel('Success Rate')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/success_rate_by_language.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Duration distribution by language
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='language', y='duration_seconds')
        plt.title('API Call Duration by Language')
        plt.ylabel('Duration (seconds)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/duration_by_language.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Temporal patterns
        plt.figure(figsize=(15, 8))
        
        # Success rate over time
        df['date'] = df['timestamp'].dt.date
        daily_success = df.groupby('date')['success_binary'].mean()
        
        plt.subplot(2, 1, 1)
        daily_success.plot(kind='line', marker='o')
        plt.title('Daily Success Rate Over Time')
        plt.ylabel('Success Rate')
        plt.grid(True, alpha=0.3)
        
        # Call volume over time
        plt.subplot(2, 1, 2)
        daily_calls = df.groupby('date').size()
        daily_calls.plot(kind='bar')
        plt.title('Daily Call Volume')
        plt.ylabel('Number of Calls')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/temporal_patterns.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Batch analysis
        plt.figure(figsize=(12, 6))
        batch_success = df.groupby('batch_id')['success_binary'].mean()
        plt.hist(batch_success, bins=20, alpha=0.7, edgecolor='black')
        plt.title('Distribution of Batch Success Rates')
        plt.xlabel('Batch Success Rate')
        plt.ylabel('Number of Batches')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/batch_success_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Created visualizations in {save_dir}")
    
    def save_analysis_results(self, results: Dict, session_id: str):
        """Save analysis results to files."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save descriptive statistics
        desc_stats_file = f"{self.results_dir}/descriptive_stats_{session_id}_{timestamp}.json"
        with open(desc_stats_file, 'w', encoding='utf-8') as f:
            json.dump(results.get('descriptive_stats', {}), f, ensure_ascii=False, indent=2, default=str)
        
        # Save multilevel analysis results
        ml_results_file = f"{self.results_dir}/multilevel_analysis_{session_id}_{timestamp}.json"
        with open(ml_results_file, 'w', encoding='utf-8') as f:
            json.dump(results.get('multilevel_analysis', {}), f, ensure_ascii=False, indent=2, default=str)
        
        # Save summary report
        summary_file = f"{self.results_dir}/analysis_summary_{session_id}_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("CROSS-LINGUISTIC DREAM RESEARCH - STATISTICAL ANALYSIS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            desc_stats = results.get('descriptive_stats', {})
            f.write(f"Total Observations: {desc_stats.get('total_observations', 'N/A')}\n")
            f.write(f"Languages: {desc_stats.get('total_languages', 'N/A')}\n")
            f.write(f"Batches: {desc_stats.get('total_batches', 'N/A')}\n")
            f.write(f"Overall Success Rate: {desc_stats.get('overall_success_rate', 'N/A'):.3f}\n")
            f.write(f"Temporal Coverage: {desc_stats.get('temporal_coverage_hours', 'N/A'):.1f} hours\n\n")
            
            f.write("SUCCESS RATES BY LANGUAGE:\n")
            for lang, rate in desc_stats.get('success_rate_by_language', {}).items():
                f.write(f"  {lang}: {rate:.3f}\n")
            
            f.write("\nMULTILEVEL ANALYSIS RESULTS:\n")
            ml_results = results.get('multilevel_analysis', {})
            if 'language_fixed_effect' in ml_results:
                f.write("  Language Fixed Effect Model: ✓\n")
            if 'mixed_effects_batch' in ml_results:
                f.write("  Mixed Effects Model: ✓\n")
            if 'theme_analysis' in ml_results:
                f.write(f"  Theme Analysis: {len(ml_results['theme_analysis'])} themes analyzed\n")
        
        logging.info(f"Analysis results saved: {desc_stats_file}, {ml_results_file}, {summary_file}")
    
    def run_complete_analysis(self, session_id: str) -> Dict:
        """Run complete statistical analysis for a session."""
        
        logging.info(f"Starting complete analysis for session {session_id}")
        
        # Load data
        api_calls_df, dreams_df = self.load_session_data(session_id)
        
        # Prepare data
        df = self.prepare_data_for_analysis(api_calls_df, dreams_df)
        df = self.create_binary_features(df)
        
        # Run analyses
        results = {
            'descriptive_stats': self.run_descriptive_statistics(df),
            'multilevel_analysis': self.run_multilevel_analysis(df)
        }
        
        # Create visualizations
        self.create_visualizations(df, f"{self.results_dir}/{session_id}")
        
        # Save results
        self.save_analysis_results(results, session_id)
        
        logging.info(f"Completed analysis for session {session_id}")
        
        return results

def main():
    """Main function for running statistical analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Statistical Analysis for Dream Research')
    parser.add_argument('--session-id', type=str, required=True, help='Session ID to analyze')
    parser.add_argument('--logs-dir', type=str, default='logs', help='Directory containing log files')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = DreamStatisticalAnalyzer(args.logs_dir)
    
    # Run analysis
    try:
        results = analyzer.run_complete_analysis(args.session_id)
        print(f"Analysis completed successfully for session {args.session_id}")
        print(f"Results saved in {analyzer.results_dir}/")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        logging.error(f"Analysis failed: {e}")

if __name__ == "__main__":
    main() 