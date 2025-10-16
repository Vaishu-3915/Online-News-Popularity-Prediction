#!/usr/bin/env python3
"""
Data Preprocessing Module for Online News Popularity Analysis

This module handles all data loading, cleaning, preprocessing, and feature engineering
for the Online News Popularity dataset.
"""

import numpy as np
import pandas as pd
from scipy.stats import levene, kruskal
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    """Handles data loading, cleaning, and preprocessing."""
    
    def __init__(self, file_path):
        """
        Initialize DataProcessor with file path.
        
        Args:
            file_path (str): Path to the CSV file
        """
        self.file_path = file_path
        self.df = None
        
    def load_data(self):
        """Load data from CSV file."""
        print("Loading data...")
        self.df = pd.read_csv(self.file_path)
        print(f"Data loaded successfully. Shape: {self.df.shape}")
        return self.df
    
    def clean_column_names(self):
        """Remove spaces from column names."""
        self.df.columns = self.df.columns.str.replace(" ", "")
        print("Column names cleaned.")
    
    def drop_irrelevant_columns(self):
        """Drop irrelevant columns like URL and timedelta."""
        columns_to_drop = ['url', 'timedelta']
        self.df.drop(columns_to_drop, axis=1, inplace=True)
        print(f"Dropped columns: {columns_to_drop}")
    
    def create_day_of_week_feature(self):
        """Create a single day_of_week feature from multiple weekday columns."""
        weekday_cols = ['weekday_is_monday', 'weekday_is_tuesday',
                       'weekday_is_wednesday', 'weekday_is_thursday', 'weekday_is_friday',
                       'weekday_is_saturday', 'weekday_is_sunday']
        
        self.df['day_of_week'] = self.df[weekday_cols].idxmax(axis=1)
        weekday_map = dict(zip(weekday_cols, np.arange(1, 8)))
        self.df['day_of_week'] = self.df['day_of_week'].map(weekday_map)
        print("Created day_of_week feature.")
    
    def create_popularity_target(self, threshold=1400):
        """
        Create binary popularity target based on shares threshold.
        
        Args:
            threshold (int): Threshold for determining popularity
        """
        self.df['popularity'] = self.df['shares'].apply(lambda x: 0 if x < threshold else 1)
        print(f"Created binary popularity target with threshold {threshold}")
    
    def remove_highly_correlated_features(self):
        """Remove highly correlated features to reduce redundancy."""
        columns_to_drop = [
            'self_reference_avg_sharess', 'kw_max_min', 'avg_positive_polarity',
            'abs_title_sentiment_polarity', 'global_rate_negative_words',
            'n_non_stop_words', 'self_reference_min_shares', 'is_weekend',
            'kw_max_avg', 'avg_negative_polarity', 'rate_positive_words',
            'kw_max_max', 'n_non_stop_unique_tokens', 'LDA_00', 'LDA_02', 'LDA_04'
        ]
        
        self.df.drop(columns_to_drop, inplace=True, axis=1)
        print(f"Dropped {len(columns_to_drop)} highly correlated features.")
    
    def apply_transformations(self):
        """Apply log transformation to skewed numeric features."""
        numeric_cols = []
        cat_cols = ['data_channel_is_lifestyle', 'data_channel_is_entertainment', 
                   'data_channel_is_bus', 'data_channel_is_socmed', 'data_channel_is_tech', 
                   'data_channel_is_world', 'weekday_is_monday', 'weekday_is_tuesday',
                   'weekday_is_wednesday', 'weekday_is_thursday', 'weekday_is_friday',
                   'weekday_is_saturday', 'weekday_is_sunday']
        
        for col in self.df.columns:
            if col not in cat_cols and col != 'popularity':
                numeric_cols.append(col)
        
        # Apply log transformation to skewed features
        skewed_cols = []
        for col in numeric_cols:
            skewness = self.df[col].skew()
            if abs(skewness) > 0.5 and (self.df[col] > 0).all():
                self.df[col] = np.log(self.df[col])
                skewed_cols.append(col)
        
        print(f"Applied log transformation to {len(skewed_cols)} features.")
    
    def create_engineered_features(self):
        """Create new features through feature engineering."""
        # Create spread features
        self.df['kw_min_max_spread'] = self.df['kw_min_max'] - self.df['kw_min_min']
        self.df['kw_avg_spread'] = self.df['kw_avg_max'] - self.df['kw_avg_min']
        
        # Create interaction features
        self.df['average_token_length_times_num_keywords'] = (
            self.df['average_token_length'] * self.df['num_keywords']
        )
        
        # Create average features
        self.df['average_lda_topic_score'] = (self.df['LDA_01'] + self.df['LDA_03']) / 2
        self.df['average_positive_polarity'] = (
            self.df['min_positive_polarity'] + self.df['max_positive_polarity']
        ) / 2
        self.df['average_negative_polarity'] = (
            self.df['min_negative_polarity'] + self.df['max_negative_polarity']
        ) / 2
        
        # Drop original features that were used for engineering
        features_to_drop = [
            'kw_min_max', 'kw_min_min', 'kw_avg_max', 'kw_avg_min',
            'n_tokens_title', 'n_tokens_content', 'num_hrefs', 'LDA_01', 'LDA_03',
            'num_self_hrefs', 'num_imgs', 'num_videos', 'average_token_length',
            'num_keywords', 'min_positive_polarity', 'max_positive_polarity',
            'min_negative_polarity', 'max_negative_polarity'
        ]
        
        self.df.drop(features_to_drop, axis=1, inplace=True)
        print("Created engineered features and dropped original features.")
    
    def feature_selection(self):
        """Perform statistical feature selection using Kruskal-Wallis test."""
        print("Performing feature selection...")
        
        # Perform Kruskal-Wallis test for each feature
        alpha = 0.05
        feature_results = {}
        target_groups = self.df.groupby('popularity')
        
        for feature in self.df.columns:
            if feature != 'popularity':
                group_values = [target_groups.get_group(group)[feature] 
                              for group in target_groups.groups]
                _, p_value = kruskal(*group_values)
                
                if p_value < alpha:
                    significance = 'Significant'
                else:
                    significance = 'Not Significant'
                
                feature_results[feature] = {'p-value': p_value, 'Significance': significance}
        
        # Remove non-significant features
        exclude_cols = ['average_negative_polarity', 'average_lda_topic_score',
                       'abs_title_subjectivity', 'data_channel_is_bus']
        self.df.drop(columns=exclude_cols, axis=1, inplace=True)
        print(f"Feature selection completed. Removed {len(exclude_cols)} non-significant features.")
    
    def final_cleanup(self):
        """Final data cleanup steps."""
        # Drop shares and day_of_week columns
        self.df.drop(['shares', 'day_of_week'], axis=1, inplace=True)
        
        # Remove any remaining NaN values
        self.df = self.df.dropna()
        
        print(f"Final dataset shape: {self.df.shape}")
        print("Data preprocessing completed.")
    
    def process_data(self):
        """Run the complete data processing pipeline."""
        self.load_data()
        self.clean_column_names()
        self.drop_irrelevant_columns()
        self.create_day_of_week_feature()
        self.create_popularity_target()
        self.remove_highly_correlated_features()
        self.apply_transformations()
        self.create_engineered_features()
        self.feature_selection()
        self.final_cleanup()
        return self.df


def get_data_summary(df):
    """
    Get a summary of the processed dataset.
    
    Args:
        df (pd.DataFrame): Processed dataset
        
    Returns:
        dict: Summary statistics
    """
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'target_distribution': df['popularity'].value_counts().to_dict(),
        'missing_values': df.isnull().sum().sum(),
        'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_features': len(df.select_dtypes(include=['object']).columns)
    }
    return summary


def print_data_summary(summary):
    """Print formatted data summary."""
    print("\n" + "="*50)
    print("DATA SUMMARY")
    print("="*50)
    print(f"Dataset shape: {summary['shape']}")
    print(f"Number of features: {len(summary['columns'])}")
    print(f"Numeric features: {summary['numeric_features']}")
    print(f"Categorical features: {summary['categorical_features']}")
    print(f"Missing values: {summary['missing_values']}")
    print(f"Target distribution: {summary['target_distribution']}")
    print("="*50)
