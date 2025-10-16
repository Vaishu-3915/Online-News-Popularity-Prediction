#!/usr/bin/env python3
"""
Cross-Validation Module for Online News Popularity Analysis

This module provides comprehensive cross-validation functionality
for better model evaluation without retraining.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from models import LogisticRegression, NaiveBayes, TORCH_AVAILABLE
from evaluation import calculate_metrics


class CrossValidator:
    """Handles cross-validation for different models."""
    
    def __init__(self, cv_folds=5, random_state=42):
        """
        Initialize CrossValidator.
        
        Args:
            cv_folds (int): Number of cross-validation folds
            random_state (int): Random state for reproducibility
        """
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.results = {}
    
    def evaluate_logistic_regression(self, df, learning_rate=0.00001, max_iteration=1000):
        """
        Perform cross-validation for Logistic Regression.
        
        Args:
            df (pd.DataFrame): Dataset
            learning_rate (float): Learning rate
            max_iteration (int): Maximum iterations
            
        Returns:
            dict: Cross-validation results
        """
        print("Performing cross-validation for Logistic Regression...")
        
        # Prepare data
        X = df.drop('popularity', axis=1)
        y = df['popularity']
        
        # Initialize cross-validation
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        cv_results = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': []
        }
        
        fold = 1
        for train_idx, val_idx in skf.split(X, y):
            print(f"  Fold {fold}/{self.cv_folds}...")
            
            # Split data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Create train/test dataframes
            train_df = pd.concat([X_train, y_train], axis=1)
            val_df = pd.concat([X_val, y_val], axis=1)
            
            # Train model
            model = LogisticRegression(train_df, val_df, learning_rate=learning_rate, max_iteration=max_iteration)
            model.fit()
            
            # Make predictions
            y_pred = model.predict(model.X_test_scaled)
            
            # Calculate metrics
            metrics = calculate_metrics(y_val, y_pred)
            
            cv_results['accuracy'].append(metrics['accuracy'])
            cv_results['precision'].append(metrics['precision'])
            cv_results['recall'].append(metrics['recall'])
            cv_results['f1_score'].append(metrics['f1_score'])
            
            fold += 1
        
        # Calculate summary statistics
        summary = self._calculate_summary(cv_results)
        self.results['Logistic Regression'] = {
            'cv_results': cv_results,
            'summary': summary
        }
        
        return summary
    
    def evaluate_naive_bayes(self, df):
        """
        Perform cross-validation for Naive Bayes.
        
        Args:
            df (pd.DataFrame): Dataset
            
        Returns:
            dict: Cross-validation results
        """
        print("Performing cross-validation for Naive Bayes...")
        
        # Prepare data
        X = df.drop('popularity', axis=1)
        y = df['popularity']
        
        # Initialize cross-validation
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        cv_results = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': []
        }
        
        fold = 1
        for train_idx, val_idx in skf.split(X, y):
            print(f"  Fold {fold}/{self.cv_folds}...")
            
            # Split data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Create train/test dataframes
            train_df = pd.concat([X_train, y_train], axis=1)
            val_df = pd.concat([X_val, y_val], axis=1)
            
            # Train model
            model = NaiveBayes(train_df, val_df)
            model.fit()
            
            # Make predictions
            y_pred = model.predict(model.X_test_cont, model.X_test_cat)
            
            # Calculate metrics
            metrics = calculate_metrics(y_val, y_pred)
            
            cv_results['accuracy'].append(metrics['accuracy'])
            cv_results['precision'].append(metrics['precision'])
            cv_results['recall'].append(metrics['recall'])
            cv_results['f1_score'].append(metrics['f1_score'])
            
            fold += 1
        
        # Calculate summary statistics
        summary = self._calculate_summary(cv_results)
        self.results['Naive Bayes'] = {
            'cv_results': cv_results,
            'summary': summary
        }
        
        return summary
    
    def _calculate_summary(self, cv_results):
        """
        Calculate summary statistics for cross-validation results.
        
        Args:
            cv_results (dict): Cross-validation results
            
        Returns:
            dict: Summary statistics
        """
        summary = {}
        
        for metric, values in cv_results.items():
            summary[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }
        
        return summary
    
    def print_results(self):
        """Print cross-validation results in a formatted way."""
        print("\n" + "="*80)
        print("CROSS-VALIDATION RESULTS")
        print("="*80)
        
        for model_name, result in self.results.items():
            print(f"\n{model_name.upper()}:")
            print("-" * 50)
            
            summary = result['summary']
            
            for metric, stats in summary.items():
                print(f"{metric.capitalize():>12}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                      f"(min: {stats['min']:.4f}, max: {stats['max']:.4f})")
    
    def plot_results(self, figsize=(15, 10)):
        """
        Plot cross-validation results.
        
        Args:
            figsize (tuple): Figure size
        """
        if not self.results:
            print("No results to plot. Run cross-validation first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Prepare data for plotting
            plot_data = []
            model_names = []
            
            for model_name, result in self.results.items():
                values = result['cv_results'][metric]
                plot_data.extend(values)
                model_names.extend([model_name] * len(values))
            
            # Create box plot
            df_plot = pd.DataFrame({
                'Model': model_names,
                'Score': plot_data
            })
            
            sns.boxplot(data=df_plot, x='Model', y='Score', ax=ax)
            ax.set_title(f'{metric.capitalize()} Distribution')
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_best_model(self, metric='f1_score'):
        """
        Get the best performing model based on specified metric.
        
        Args:
            metric (str): Metric to use for comparison
            
        Returns:
            tuple: (model_name, mean_score, std_score)
        """
        if not self.results:
            return None, None, None
        
        best_model = None
        best_score = -1
        best_std = 0
        
        for model_name, result in self.results.items():
            mean_score = result['summary'][metric]['mean']
            if mean_score > best_score:
                best_score = mean_score
                best_std = result['summary'][metric]['std']
                best_model = model_name
        
        return best_model, best_score, best_std
    
    def save_results(self, filename="cross_validation_results.json"):
        """
        Save cross-validation results to JSON file.
        
        Args:
            filename (str): Output filename
        """
        import json
        
        # Convert numpy types to Python types for JSON serialization
        serializable_results = {}
        
        for model_name, result in self.results.items():
            serializable_results[model_name] = {}
            
            # Convert cv_results
            cv_results = {}
            for metric, values in result['cv_results'].items():
                cv_results[metric] = [float(v) for v in values]
            serializable_results[model_name]['cv_results'] = cv_results
            
            # Convert summary
            summary = {}
            for metric, stats in result['summary'].items():
                summary[metric] = {
                    'mean': float(stats['mean']),
                    'std': float(stats['std']),
                    'min': float(stats['min']),
                    'max': float(stats['max'])
                }
            serializable_results[model_name]['summary'] = summary
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Cross-validation results saved to {filename}")


def quick_cross_validation(df, models=['logistic', 'naive_bayes'], cv_folds=5):
    """
    Quick cross-validation for multiple models.
    
    Args:
        df (pd.DataFrame): Dataset
        models (list): List of models to evaluate
        cv_folds (int): Number of CV folds
        
    Returns:
        CrossValidator: CrossValidator object with results
    """
    cv = CrossValidator(cv_folds=cv_folds)
    
    if 'logistic' in models:
        cv.evaluate_logistic_regression(df)
    
    if 'naive_bayes' in models:
        cv.evaluate_naive_bayes(df)
    
    # Print and plot results
    cv.print_results()
    cv.plot_results()
    
    return cv


def compare_cv_with_single_split(df, test_size=0.3, cv_folds=5):
    """
    Compare cross-validation results with single train-test split.
    
    Args:
        df (pd.DataFrame): Dataset
        test_size (float): Test set size for single split
        cv_folds (int): Number of CV folds
        
    Returns:
        dict: Comparison results
    """
    from sklearn.model_selection import train_test_split
    from models import LogisticRegression, NaiveBayes
    from evaluation import evaluate_logistic_regression, evaluate_naive_bayes
    
    print("Comparing Cross-Validation vs Single Split...")
    
    # Single split
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df['popularity'])
    
    # Train models on single split
    lr_model = LogisticRegression(train_df, test_df)
    lr_model.fit()
    lr_train_metrics, lr_test_metrics = evaluate_logistic_regression(lr_model)
    
    nb_model = NaiveBayes(train_df, test_df)
    nb_model.fit()
    nb_train_metrics, nb_test_metrics = evaluate_naive_bayes(nb_model)
    
    # Cross-validation
    cv = CrossValidator(cv_folds=cv_folds)
    cv.evaluate_logistic_regression(df)
    cv.evaluate_naive_bayes(df)
    
    # Compare results
    comparison = {
        'Logistic Regression': {
            'single_split': lr_test_metrics,
            'cross_validation': cv.results['Logistic Regression']['summary']
        },
        'Naive Bayes': {
            'single_split': nb_test_metrics,
            'cross_validation': cv.results['Naive Bayes']['summary']
        }
    }
    
    # Print comparison
    print("\n" + "="*80)
    print("CROSS-VALIDATION vs SINGLE SPLIT COMPARISON")
    print("="*80)
    
    for model_name, results in comparison.items():
        print(f"\n{model_name.upper()}:")
        print("-" * 50)
        
        single_split = results['single_split']
        cv_results = results['cross_validation']
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for metric in metrics:
            single_score = single_split[metric]
            cv_mean = cv_results[metric]['mean']
            cv_std = cv_results[metric]['std']
            
            print(f"{metric.capitalize():>12}: Single={single_score:.4f}, "
                  f"CV={cv_mean:.4f}±{cv_std:.4f}")
    
    return comparison
