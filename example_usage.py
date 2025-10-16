#!/usr/bin/env python3
"""
Example Script: Using Individual Modules

This script demonstrates how to use the individual modules separately
for custom analysis workflows.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# Import our custom modules
from preprocessing import DataProcessor, get_data_summary, print_data_summary
from models import LogisticRegression, NaiveBayes, TORCH_AVAILABLE
from evaluation import (
    evaluate_logistic_regression, evaluate_naive_bayes,
    print_classification_report, plot_model_comparison, compare_models
)


def example_custom_analysis():
    """Example of custom analysis using individual modules."""
    
    print("="*60)
    print("CUSTOM ANALYSIS EXAMPLE")
    print("="*60)
    
    # Step 1: Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    processor = DataProcessor("data/OnlineNewsPopularity/OnlineNewsPopularity.csv")
    df = processor.process_data()
    
    # Get data summary
    summary = get_data_summary(df)
    print_data_summary(summary)
    
    # Step 2: Custom train-test split
    print("\n2. Creating custom train-test split...")
    train_df, test_df = train_test_split(df, test_size=0.25, random_state=123, stratify=df['popularity'])
    print(f"Training set: {train_df.shape}")
    print(f"Test set: {test_df.shape}")
    
    # Step 3: Train specific models
    print("\n3. Training models...")
    
    # Train Logistic Regression with custom parameters
    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(train_df, test_df, 
                                 learning_rate=0.001, epsilon=1e-5, max_iteration=500)
    lr_model.fit()
    
    # Train Naive Bayes
    print("\nTraining Naive Bayes...")
    nb_model = NaiveBayes(train_df, test_df)
    nb_model.fit()
    
    # Step 4: Evaluate models
    print("\n4. Evaluating models...")
    
    # Evaluate Logistic Regression
    lr_train_metrics, lr_test_metrics = evaluate_logistic_regression(lr_model)
    print_classification_report("Logistic Regression", lr_train_metrics, lr_test_metrics)
    
    # Evaluate Naive Bayes
    nb_train_metrics, nb_test_metrics = evaluate_naive_bayes(nb_model)
    print_classification_report("Naive Bayes", nb_train_metrics, nb_test_metrics)
    
    # Step 5: Compare models
    print("\n5. Comparing models...")
    results = {
        'Logistic Regression': (lr_train_metrics, lr_test_metrics),
        'Naive Bayes': (nb_train_metrics, nb_test_metrics)
    }
    
    comparison_df = compare_models(results)
    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False, float_format='%.4f'))
    
    # Plot comparison
    plot_model_comparison(comparison_df, 'Test_Accuracy')
    
    print("\nCustom analysis completed!")


def example_preprocessing_only():
    """Example of using only the preprocessing module."""
    
    print("="*60)
    print("PREPROCESSING ONLY EXAMPLE")
    print("="*60)
    
    # Initialize processor
    processor = DataProcessor("data/OnlineNewsPopularity/OnlineNewsPopularity.csv")
    
    # Run preprocessing pipeline
    df = processor.process_data()
    
    # Get detailed information about the processed data
    print(f"\nProcessed dataset shape: {df.shape}")
    print(f"Features: {list(df.columns)}")
    print(f"Target distribution: {df['popularity'].value_counts().to_dict()}")
    
    # Save processed data
    df.to_csv("processed_data.csv", index=False)
    print("Processed data saved to 'processed_data.csv'")


def example_single_model():
    """Example of training and evaluating a single model."""
    
    print("="*60)
    print("SINGLE MODEL EXAMPLE")
    print("="*60)
    
    # Load and preprocess data
    processor = DataProcessor("data/OnlineNewsPopularity/OnlineNewsPopularity.csv")
    df = processor.process_data()
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['popularity'])
    
    # Train only Logistic Regression
    print("\nTraining Logistic Regression model...")
    lr_model = LogisticRegression(train_df, test_df)
    lr_model.fit()
    
    # Evaluate and print results
    train_metrics, test_metrics = evaluate_logistic_regression(lr_model)
    print_classification_report("Logistic Regression", train_metrics, test_metrics)
    
    print("\nSingle model analysis completed!")


if __name__ == "__main__":
    print("Choose an example to run:")
    print("1. Custom analysis with multiple models")
    print("2. Preprocessing only")
    print("3. Single model training")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        example_custom_analysis()
    elif choice == "2":
        example_preprocessing_only()
    elif choice == "3":
        example_single_model()
    else:
        print("Invalid choice. Running custom analysis example...")
        example_custom_analysis()
