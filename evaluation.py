#!/usr/bin/env python3
"""
Evaluation Utilities Module for Online News Popularity Analysis

This module contains functions for model evaluation, metrics calculation,
and visualization of results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')


def calculate_metrics(y_true, y_pred):
    """
    Calculate classification metrics.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        
    Returns:
        dict: Dictionary containing accuracy, precision, recall, and F1-score
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Convert to binary if needed
    y_true_binary = (y_true == 1)
    y_pred_binary = (y_pred == 1)
    
    # Calculate metrics
    accuracy = (y_true_binary == y_pred_binary).sum() / len(y_true_binary)
    
    # Handle division by zero
    true_positives = (y_true_binary & y_pred_binary).sum()
    false_positives = (~y_true_binary & y_pred_binary).sum()
    false_negatives = (y_true_binary & ~y_pred_binary).sum()
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", figsize=(8, 6)):
    """
    Plot confusion matrix.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        title (str): Title for the plot
        figsize (tuple): Figure size
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()


def print_classification_report(model_name, train_metrics, test_metrics):
    """
    Print formatted classification report.
    
    Args:
        model_name (str): Name of the model
        train_metrics (dict): Training metrics
        test_metrics (dict): Test metrics
    """
    print("\n" + "="*60)
    print(f"{model_name.upper()} RESULTS")
    print("="*60)
    
    print("TRAINING RESULTS:")
    print(f"Accuracy:  {train_metrics['accuracy']:.4f}")
    print(f"Precision: {train_metrics['precision']:.4f}")
    print(f"Recall:    {train_metrics['recall']:.4f}")
    print(f"F1-Score:  {train_metrics['f1_score']:.4f}")
    
    print("\nTESTING RESULTS:")
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1-Score:  {test_metrics['f1_score']:.4f}")


def evaluate_logistic_regression(model):
    """
    Evaluate Logistic Regression model and return metrics.
    
    Args:
        model: Trained LogisticRegression model
        
    Returns:
        tuple: (train_metrics, test_metrics)
    """
    # Make predictions
    y_train_pred = model.predict(model.X_train_scaled)
    y_test_pred = model.predict(model.X_test_scaled)
    
    # Calculate metrics
    train_metrics = calculate_metrics(model.y_train, y_train_pred)
    test_metrics = calculate_metrics(model.y_test, y_test_pred)
    
    return train_metrics, test_metrics


def evaluate_naive_bayes(model):
    """
    Evaluate Naive Bayes model and return metrics.
    
    Args:
        model: Trained NaiveBayes model
        
    Returns:
        tuple: (train_metrics, test_metrics)
    """
    # Make predictions
    y_train_pred = model.predict(model.X_train_cont, model.X_train_cat)
    y_test_pred = model.predict(model.X_test_cont, model.X_test_cat)
    
    # Calculate metrics
    train_metrics = calculate_metrics(model.y_train, y_train_pred)
    test_metrics = calculate_metrics(model.y_test, y_test_pred)
    
    return train_metrics, test_metrics


def evaluate_svm(model):
    """
    Evaluate SVM model and return metrics.
    
    Args:
        model: Trained KernelSVM model
        
    Returns:
        tuple: (train_metrics, test_metrics)
    """
    # Make predictions
    y_train_pred = model.predict(model.X_train)
    y_test_pred = model.predict(model.X_test)
    
    # Convert SVM predictions back to 0/1 format
    y_train_pred_binary = (y_train_pred == 1).astype(int)
    y_test_pred_binary = (y_test_pred == 1).astype(int)
    
    # Convert SVM targets back to 0/1 format
    y_train_binary = (model.y_train == 1).astype(int)
    y_test_binary = (model.y_test == 1).astype(int)
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train_binary, y_train_pred_binary)
    test_metrics = calculate_metrics(y_test_binary, y_test_pred_binary)
    
    return train_metrics, test_metrics


def compare_models(results_dict):
    """
    Compare performance of multiple models.
    
    Args:
        results_dict (dict): Dictionary with model names as keys and 
                           (train_metrics, test_metrics) tuples as values
        
    Returns:
        pd.DataFrame: Comparison table
    """
    comparison_data = []
    
    for model_name, (train_metrics, test_metrics) in results_dict.items():
        comparison_data.append({
            'Model': model_name,
            'Train_Accuracy': train_metrics['accuracy'],
            'Test_Accuracy': test_metrics['accuracy'],
            'Train_Precision': train_metrics['precision'],
            'Test_Precision': test_metrics['precision'],
            'Train_Recall': train_metrics['recall'],
            'Test_Recall': test_metrics['recall'],
            'Train_F1': train_metrics['f1_score'],
            'Test_F1': test_metrics['f1_score']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df


def plot_model_comparison(comparison_df, metric='Test_Accuracy', figsize=(12, 6)):
    """
    Plot model comparison for a specific metric.
    
    Args:
        comparison_df (pd.DataFrame): Model comparison dataframe
        metric (str): Metric to plot
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    
    # Create bar plot
    bars = plt.bar(comparison_df['Model'], comparison_df[metric])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.title(f'Model Comparison - {metric}')
    plt.xlabel('Model')
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_all_metrics_comparison(comparison_df, figsize=(15, 10)):
    """
    Plot comparison of all metrics across models.
    
    Args:
        comparison_df (pd.DataFrame): Model comparison dataframe
        figsize (tuple): Figure size
    """
    metrics = ['Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1']
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        bars = axes[i].bar(comparison_df['Model'], comparison_df[metric])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
        
        axes[i].set_title(metric)
        axes[i].set_xlabel('Model')
        axes[i].set_ylabel(metric)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def generate_summary_report(results_dict):
    """
    Generate a comprehensive summary report.
    
    Args:
        results_dict (dict): Dictionary with model results
        
    Returns:
        str: Formatted summary report
    """
    report = []
    report.append("="*80)
    report.append("ONLINE NEWS POPULARITY ANALYSIS - SUMMARY REPORT")
    report.append("="*80)
    
    # Model comparison table
    comparison_df = compare_models(results_dict)
    
    report.append("\nMODEL PERFORMANCE COMPARISON:")
    report.append("-" * 50)
    report.append(comparison_df.to_string(index=False, float_format='%.4f'))
    
    # Best performing model for each metric
    report.append("\nBEST PERFORMING MODELS:")
    report.append("-" * 30)
    
    metrics = ['Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1']
    for metric in metrics:
        best_model = comparison_df.loc[comparison_df[metric].idxmax(), 'Model']
        best_score = comparison_df[metric].max()
        report.append(f"{metric}: {best_model} ({best_score:.4f})")
    
    # Overall best model (based on F1-score)
    best_overall = comparison_df.loc[comparison_df['Test_F1'].idxmax(), 'Model']
    best_f1 = comparison_df['Test_F1'].max()
    report.append(f"\nOVERALL BEST MODEL: {best_overall} (F1-Score: {best_f1:.4f})")
    
    report.append("\n" + "="*80)
    
    return "\n".join(report)


def save_results(results_dict, filename="model_results.csv"):
    """
    Save model results to CSV file.
    
    Args:
        results_dict (dict): Dictionary with model results
        filename (str): Output filename
    """
    comparison_df = compare_models(results_dict)
    comparison_df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


def plot_training_history(errors, title="Training History", figsize=(10, 6)):
    """
    Plot training history (for models that track errors).
    
    Args:
        errors (list): List of error values
        title (str): Plot title
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    plt.plot(errors)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Cost/Loss')
    plt.grid(True)
    plt.show()
