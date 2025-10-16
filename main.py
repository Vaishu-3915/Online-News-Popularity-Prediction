#!/usr/bin/env python3
"""
Main Script for Online News Popularity Analysis

This script orchestrates the complete analysis pipeline including:
- Data preprocessing
- Model training and evaluation
- Results comparison and visualization
"""

import argparse
import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Import our custom modules
from preprocessing import DataProcessor, get_data_summary, print_data_summary
from models import LogisticRegression, NaiveBayes, NeuralNetworkTrainer, KernelSVM, TORCH_AVAILABLE
from evaluation import (
    evaluate_logistic_regression, evaluate_naive_bayes, evaluate_svm,
    print_classification_report, compare_models, plot_model_comparison,
    plot_all_metrics_comparison, generate_summary_report, save_results
)
from model_persistence import ModelPersistence, save_training_results
from cross_validation import CrossValidator, quick_cross_validation
from report_generator import ReportGenerator, generate_all_reports


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Online News Popularity Analysis')
    
    parser.add_argument('--data-path', type=str, 
                       default='data/OnlineNewsPopularity/OnlineNewsPopularity.csv',
                       help='Path to the dataset CSV file')
    
    parser.add_argument('--test-size', type=float, default=0.3,
                       help='Test set size (default: 0.3)')
    
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility (default: 42)')
    
    parser.add_argument('--models', nargs='+', 
                       choices=['logistic', 'naive_bayes', 'neural_network', 'svm', 'all'],
                       default=['all'],
                       help='Models to train (default: all)')
    
    parser.add_argument('--svm-sample-size', type=int, default=6000,
                       help='Sample size for SVM training (default: 6000)')
    
    parser.add_argument('--neural-network-epochs', type=int, default=10,
                       help='Number of epochs for neural network (default: 10)')
    
    parser.add_argument('--neural-network-hidden-size', type=int, default=128,
                       help='Hidden layer size for neural network (default: 128)')
    
    parser.add_argument('--save-results', action='store_true',
                       help='Save results to CSV file')
    
    parser.add_argument('--plot-results', action='store_true',
                       help='Generate comparison plots')
    
    parser.add_argument('--save-models', action='store_true',
                       help='Save trained models for future use')
    
    parser.add_argument('--cross-validation', action='store_true',
                       help='Perform cross-validation analysis')
    
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds (default: 5)')
    
    parser.add_argument('--generate-reports', action='store_true',
                       help='Generate professional HTML and text reports')
    
    parser.add_argument('--launch-dashboard', action='store_true',
                       help='Launch interactive Streamlit dashboard')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    return parser.parse_args()


def check_data_file(data_path):
    """Check if data file exists."""
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please check the file path and try again.")
        sys.exit(1)


def train_logistic_regression(train_df, test_df, save_model=False, verbose=False):
    """Train and evaluate Logistic Regression model."""
    if verbose:
        print("\n" + "="*50)
        print("TRAINING LOGISTIC REGRESSION")
        print("="*50)
    
    # Initialize and train model
    lr_model = LogisticRegression(train_df, test_df, 
                                 learning_rate=0.00001, epsilon=0.0, max_iteration=1000)
    lr_model.fit()
    
    # Evaluate model
    train_metrics, test_metrics = evaluate_logistic_regression(lr_model)
    
    # Save model if requested
    if save_model:
        persistence = ModelPersistence()
        metadata = {
            'model_type': 'Logistic Regression',
            'learning_rate': 0.00001,
            'max_iterations': 1000,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        }
        persistence.save_model(lr_model, "logistic_regression", metadata)
    
    # Print results
    print_classification_report("Logistic Regression", train_metrics, test_metrics)
    
    return train_metrics, test_metrics, lr_model


def train_naive_bayes(train_df, test_df, verbose=False):
    """Train and evaluate Naive Bayes model."""
    if verbose:
        print("\n" + "="*50)
        print("TRAINING NAIVE BAYES")
        print("="*50)
    
    # Initialize and train model
    nb_model = NaiveBayes(train_df, test_df)
    nb_model.fit()
    
    # Evaluate model
    train_metrics, test_metrics = evaluate_naive_bayes(nb_model)
    
    # Print results
    print_classification_report("Naive Bayes", train_metrics, test_metrics)
    
    return train_metrics, test_metrics


def train_neural_network(train_df, test_df, hidden_size=128, epochs=10, verbose=False):
    """Train and evaluate Neural Network model."""
    if not TORCH_AVAILABLE:
        print("\nSkipping Neural Network (PyTorch not available)")
        return None, None
    
    if verbose:
        print("\n" + "="*50)
        print("TRAINING NEURAL NETWORK")
        print("="*50)
    
    # Initialize and train model
    nn_trainer = NeuralNetworkTrainer(train_df, test_df, 
                                     hidden_size=hidden_size, epochs=epochs)
    nn_trainer.train()
    
    # Evaluate model
    test_metrics = nn_trainer.evaluate()
    
    # For consistency, create dummy train metrics (neural network doesn't track train metrics separately)
    train_metrics = test_metrics.copy()
    
    # Print results
    print_classification_report("Neural Network", train_metrics, test_metrics)
    
    return train_metrics, test_metrics


def train_svm(train_df, test_df, sample_size=6000, verbose=False):
    """Train and evaluate SVM model."""
    if verbose:
        print("\n" + "="*50)
        print("TRAINING SUPPORT VECTOR MACHINE")
        print("="*50)
    
    # Sample data for SVM (it's computationally expensive)
    df_svm = pd.concat([train_df, test_df]).sample(sample_size, random_state=42)
    train_df_svm, test_df_svm = train_test_split(df_svm, test_size=0.2, 
                                                random_state=42, stratify=df_svm['popularity'])
    
    # Initialize and train model
    svm_model = KernelSVM(train_df_svm, test_df_svm, kernel='linear', C=1, max_iter=10)
    
    # Evaluate model
    train_metrics, test_metrics = evaluate_svm(svm_model)
    
    # Print results
    print_classification_report("Support Vector Machine", train_metrics, test_metrics)
    
    return train_metrics, test_metrics


def main():
    """Main function to run the complete analysis."""
    # Parse command line arguments
    args = parse_arguments()
    
    print("="*80)
    print("ONLINE NEWS POPULARITY ANALYSIS")
    print("="*80)
    
    # Check if data file exists
    check_data_file(args.data_path)
    
    try:
        # Step 1: Data Processing
        print("\n1. DATA PROCESSING")
        print("-" * 50)
        
        processor = DataProcessor(args.data_path)
        df = processor.process_data()
        
        # Print data summary
        summary = get_data_summary(df)
        print_data_summary(summary)
        
        # Step 2: Train-Test Split
        print(f"\n2. TRAIN-TEST SPLIT (Test size: {args.test_size})")
        print("-" * 50)
        
        train_df, test_df = train_test_split(df, test_size=args.test_size, 
                                           random_state=args.random_state, 
                                           stratify=df['popularity'])
        
        print(f"Training set shape: {train_df.shape}")
        print(f"Test set shape: {test_df.shape}")
        
        # Step 3: Model Training and Evaluation
        print(f"\n3. MODEL TRAINING AND EVALUATION")
        print("-" * 50)
        
        results = {}
        
        # Determine which models to train
        models_to_train = args.models
        if 'all' in models_to_train:
            models_to_train = ['logistic', 'naive_bayes', 'neural_network', 'svm']
        
        # Train Logistic Regression
        if 'logistic' in models_to_train:
            train_metrics, test_metrics, lr_model = train_logistic_regression(
                train_df, test_df, args.save_models, args.verbose)
            results['Logistic Regression'] = (train_metrics, test_metrics)
        
        # Train Naive Bayes
        if 'naive_bayes' in models_to_train:
            train_metrics, test_metrics = train_naive_bayes(train_df, test_df, args.verbose)
            results['Naive Bayes'] = (train_metrics, test_metrics)
        
        # Train Neural Network
        if 'neural_network' in models_to_train:
            train_metrics, test_metrics = train_neural_network(
                train_df, test_df, args.neural_network_hidden_size, 
                args.neural_network_epochs, args.verbose
            )
            if train_metrics is not None:  # Only add if training was successful
                results['Neural Network'] = (train_metrics, test_metrics)
        
        # Train SVM
        if 'svm' in models_to_train:
            train_metrics, test_metrics = train_svm(train_df, test_df, args.svm_sample_size, args.verbose)
            results['Support Vector Machine'] = (train_metrics, test_metrics)
        
        # Step 4: Results Analysis
        if results:
            print(f"\n4. RESULTS ANALYSIS")
            print("-" * 50)
            
            # Generate summary report
            summary_report = generate_summary_report(results)
            print(summary_report)
            
            # Save results if requested
            if args.save_results:
                save_results(results, "model_results.csv")
                save_training_results(results, "training_results.json")
            
            # Generate plots if requested
            if args.plot_results:
                print("\nGenerating comparison plots...")
                comparison_df = compare_models(results)
                
                # Plot individual metric comparisons
                plot_model_comparison(comparison_df, 'Test_Accuracy')
                plot_model_comparison(comparison_df, 'Test_F1')
                
                # Plot all metrics comparison
                plot_all_metrics_comparison(comparison_df)
            
            # Cross-validation analysis
            cv_results = None
            if args.cross_validation:
                print(f"\n5. CROSS-VALIDATION ANALYSIS")
                print("-" * 50)
                cv = CrossValidator(cv_folds=args.cv_folds)
                
                if 'logistic' in models_to_train:
                    cv.evaluate_logistic_regression(df)
                if 'naive_bayes' in models_to_train:
                    cv.evaluate_naive_bayes(df)
                
                cv.print_results()
                cv.plot_results()
                cv_results = cv.results
            
            # Generate professional reports
            if args.generate_reports:
                print(f"\n6. GENERATING PROFESSIONAL REPORTS")
                print("-" * 50)
                report_paths = generate_all_reports(df, results, cv_results)
                print(f"Reports generated:")
                for report_type, path in report_paths.items():
                    print(f"  {report_type.upper()}: {path}")
            
            # Launch dashboard
            if args.launch_dashboard:
                print(f"\n7. LAUNCHING INTERACTIVE DASHBOARD")
                print("-" * 50)
                print("Starting Streamlit dashboard...")
                print("Dashboard will open in your default web browser.")
                print("Press Ctrl+C to stop the dashboard.")
                
                import subprocess
                try:
                    subprocess.run(["streamlit", "run", "dashboard.py"], check=True)
                except subprocess.CalledProcessError:
                    print("Error launching dashboard. Make sure Streamlit is installed:")
                    print("pip install streamlit")
                except FileNotFoundError:
                    print("Streamlit not found. Please install it:")
                    print("pip install streamlit")
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*80)
        
    except Exception as e:
        print(f"\nAn error occurred during analysis: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
