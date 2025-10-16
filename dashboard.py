#!/usr/bin/env python3
"""
Interactive Dashboard for Online News Popularity Analysis

This Streamlit app provides an interactive interface to explore
model results, data, and perform predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime

# Import our modules
from preprocessing import DataProcessor, get_data_summary
from model_persistence import ModelPersistence, load_training_results
from evaluation import calculate_metrics, plot_model_comparison, compare_models

# Page configuration
st.set_page_config(
    page_title="Online News Popularity Analysis",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    """Load and cache the dataset."""
    if 'df' not in st.session_state:
        try:
            processor = DataProcessor("data/OnlineNewsPopularity/OnlineNewsPopularity.csv")
            st.session_state.df = processor.process_data()
            st.session_state.data_summary = get_data_summary(st.session_state.df)
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
    return st.session_state.df

def load_model_results():
    """Load model results if available."""
    try:
        if os.path.exists("training_results.json"):
            return load_training_results("training_results.json")
        else:
            return None
    except Exception as e:
        st.warning(f"Could not load model results: {str(e)}")
        return None

def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<h1 class="main-header">📰 Online News Popularity Analysis</h1>', unsafe_allow_html=True)
    
    # Sidebar Navigation
    st.sidebar.title("📰 Navigation")
    
    # Create navigation buttons
    pages = {
        "🏠 Overview": "overview",
        "📊 Data Explorer": "data_explorer", 
        "🤖 Model Results": "model_results",
        "🔮 Predictions": "predictions",
        "📈 Analytics": "analytics"
    }
    
    # Display navigation buttons
    page = None
    for page_name, page_key in pages.items():
        if st.sidebar.button(page_name, key=f"nav_{page_key}", use_container_width=True):
            page = page_name
            st.session_state.current_page = page_key
    
    # Use session state to maintain page selection
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'overview'
    
    # Map session state to page name
    page_mapping = {v: k for k, v in pages.items()}
    current_page_name = page_mapping.get(st.session_state.current_page, "🏠 Overview")
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Load model results
    model_results = load_model_results()
    
    # Main content based on selected page
    if st.session_state.current_page == "overview":
        show_overview(df, model_results)
    elif st.session_state.current_page == "data_explorer":
        show_data_explorer(df)
    elif st.session_state.current_page == "model_results":
        show_model_results(model_results)
    elif st.session_state.current_page == "predictions":
        show_predictions(df)
    elif st.session_state.current_page == "analytics":
        show_analytics(df, model_results)

def show_overview(df, model_results):
    """Show overview dashboard."""
    
    st.header("📊 Dataset Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    
    with col2:
        st.metric("Features", len(df.columns) - 1)
    
    with col3:
        popular_articles = df['popularity'].sum()
        st.metric("Popular Articles", f"{popular_articles:,}")
    
    with col4:
        popularity_rate = (df['popularity'].mean() * 100)
        st.metric("Popularity Rate", f"{popularity_rate:.1f}%")
    
    # Data summary
    st.subheader("📋 Data Summary")
    summary = st.session_state.data_summary
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dataset Shape:**", summary['shape'])
        st.write("**Missing Values:**", summary['missing_values'])
        st.write("**Numeric Features:**", summary['numeric_features'])
    
    with col2:
        st.write("**Target Distribution:**")
        target_dist = summary['target_distribution']
        st.write(f"- Popular (1): {target_dist[1]:,} ({target_dist[1]/sum(target_dist.values())*100:.1f}%)")
        st.write(f"- Not Popular (0): {target_dist[0]:,} ({target_dist[0]/sum(target_dist.values())*100:.1f}%)")
    
    # Quick insights
    st.subheader("💡 Quick Insights")
    
    if model_results:
        st.success("✅ Model results are available! Check the 'Model Results' page.")
        
        # Show best model
        best_model = None
        best_f1 = 0
        for model_name, results in model_results.items():
            f1_score = results['test_metrics']['f1_score']
            if f1_score > best_f1:
                best_f1 = f1_score
                best_model = model_name
        
        if best_model:
            st.info(f"🏆 **Best Performing Model:** {best_model} (F1-Score: {best_f1:.4f})")
    else:
        st.warning("⚠️ No model results found. Run the analysis to see model performance.")
    
    # Data preview
    st.subheader("👀 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

def show_data_explorer(df):
    """Show data exploration tools."""
    
    st.header("🔍 Data Explorer")
    
    # Feature selection
    st.subheader("📊 Feature Analysis")
    
    feature_cols = [col for col in df.columns if col != 'popularity']
    selected_features = st.multiselect(
        "Select features to analyze:",
        feature_cols,
        default=feature_cols[:5] if len(feature_cols) > 5 else feature_cols
    )
    
    if selected_features:
        # Correlation heatmap
        st.subheader("🔗 Correlation Heatmap")
        corr_data = df[selected_features + ['popularity']].corr()
        
        fig = px.imshow(
            corr_data,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r"
        )
        fig.update_layout(title="Feature Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature distributions
        st.subheader("📈 Feature Distributions")
        
        n_cols = min(3, len(selected_features))
        n_rows = (len(selected_features) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=selected_features
        )
        
        for i, feature in enumerate(selected_features):
            row = i // n_cols + 1
            col = i % n_cols + 1
            
            fig.add_trace(
                go.Histogram(x=df[feature], name=feature, showlegend=False),
                row=row, col=col
            )
        
        fig.update_layout(height=300 * n_rows, title="Feature Distributions")
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature vs Popularity
        st.subheader("🎯 Feature vs Popularity")
        
        feature_to_analyze = st.selectbox("Select feature for popularity analysis:", selected_features)
        
        if feature_to_analyze:
            # Box plot
            fig = px.box(
                df,
                x='popularity',
                y=feature_to_analyze,
                title=f"{feature_to_analyze} vs Popularity"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Statistics by Popularity:**")
                stats = df.groupby('popularity')[feature_to_analyze].describe()
                st.dataframe(stats)
            
            with col2:
                st.write("**Popularity Rate by Feature Quartiles:**")
                df['quartile'] = pd.qcut(df[feature_to_analyze], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
                quartile_stats = df.groupby('quartile')['popularity'].agg(['count', 'sum', 'mean'])
                quartile_stats['popularity_rate'] = quartile_stats['mean'] * 100
                st.dataframe(quartile_stats)

def show_model_results(model_results):
    """Show model performance results."""
    
    st.header("🤖 Model Performance Results")
    
    if not model_results:
        st.warning("No model results found. Please run the analysis first.")
        return
    
    # Model comparison table
    st.subheader("📊 Model Comparison")
    
    comparison_data = []
    for model_name, results in model_results.items():
        comparison_data.append({
            'Model': model_name,
            'Test Accuracy': f"{results['test_metrics']['accuracy']:.4f}",
            'Test Precision': f"{results['test_metrics']['precision']:.4f}",
            'Test Recall': f"{results['test_metrics']['recall']:.4f}",
            'Test F1-Score': f"{results['test_metrics']['f1_score']:.4f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Performance visualization
    st.subheader("📈 Performance Visualization")
    
    # Create performance comparison chart
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    fig = go.Figure()
    
    for model_name, results in model_results.items():
        values = [results['test_metrics'][metric] for metric in metrics]
        fig.add_trace(go.Scatter(
            x=metrics,
            y=values,
            mode='lines+markers',
            name=model_name,
            line=dict(width=3),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Metrics",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Best model highlight
    best_model = max(model_results.keys(), 
                    key=lambda x: model_results[x]['test_metrics']['f1_score'])
    best_f1 = model_results[best_model]['test_metrics']['f1_score']
    
    st.success(f"🏆 **Best Model:** {best_model} with F1-Score of {best_f1:.4f}")

def show_predictions(df):
    """Show prediction interface."""
    
    st.header("🔮 Make Predictions")
    
    st.info("This feature requires trained models. Please ensure models are saved in the 'saved_models' directory.")
    
    # Model selection
    persistence = ModelPersistence()
    saved_models = persistence.list_saved_models()
    
    if not saved_models:
        st.warning("No saved models found. Please train and save models first.")
        return
    
    model_options = [f"{m['model_name']} ({m['timestamp']})" for m in saved_models]
    selected_model_info = st.selectbox("Select a model:", model_options)
    
    if selected_model_info:
        # Extract model name and timestamp
        model_name = selected_model_info.split(' (')[0]
        timestamp = selected_model_info.split(' (')[1].rstrip(')')
        
        try:
            model, metadata = persistence.load_model(model_name, timestamp)
            st.success(f"✅ Loaded model: {model_name}")
            
            # Show model metadata
            if metadata:
                st.subheader("📋 Model Information")
                st.json(metadata)
            
            # Prediction interface
            st.subheader("🎯 Input Features")
            
            # Create input form
            feature_inputs = {}
            feature_cols = [col for col in df.columns if col != 'popularity']
            
            col1, col2 = st.columns(2)
            
            for i, feature in enumerate(feature_cols):
                col = col1 if i % 2 == 0 else col2
                
                with col:
                    min_val = float(df[feature].min())
                    max_val = float(df[feature].max())
                    mean_val = float(df[feature].mean())
                    
                    feature_inputs[feature] = st.number_input(
                        f"{feature}:",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        step=(max_val - min_val) / 100
                    )
            
            # Make prediction
            if st.button("🔮 Make Prediction", type="primary"):
                try:
                    # Prepare input data
                    input_data = pd.DataFrame([feature_inputs])
                    
                    # Make prediction based on model type
                    if hasattr(model, 'predict'):
                        if hasattr(model, 'X_train_scaled'):
                            # Logistic Regression
                            input_scaled = model.test_scaling(input_data)
                            prediction = model.predict(input_scaled)[0]
                        elif hasattr(model, 'X_train_cont'):
                            # Naive Bayes
                            input_cont, input_cat = model.extract_features(input_data)
                            prediction = model.predict(input_cont, input_cat)[0]
                        else:
                            # Other models
                            prediction = model.predict(input_data)[0]
                        
                        # Display result
                        prediction_prob = "High" if prediction == 1 else "Low"
                        prediction_color = "success" if prediction == 1 else "info"
                        
                        st.markdown(f"### Prediction Result:")
                        st.markdown(f"**Popularity:** {prediction_prob} ({prediction})")
                        
                        if prediction == 1:
                            st.success("🎉 This article is predicted to be POPULAR!")
                        else:
                            st.info("📰 This article is predicted to be NOT POPULAR.")
                    
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
        
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")

def show_analytics(df, model_results):
    """Show advanced analytics."""
    
    st.header("📈 Advanced Analytics")
    
    # Data insights
    st.subheader("💡 Data Insights")
    
    # Popularity trends
    st.write("**Popularity Distribution:**")
    popularity_dist = df['popularity'].value_counts()
    
    fig = px.pie(
        values=popularity_dist.values,
        names=['Not Popular', 'Popular'],
        title="Article Popularity Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance (if available)
    if model_results:
        st.subheader("🎯 Model Insights")
        
        # Performance trends
        st.write("**Model Performance Summary:**")
        
        performance_data = []
        for model_name, results in model_results.items():
            performance_data.append({
                'Model': model_name,
                'Accuracy': results['test_metrics']['accuracy'],
                'Precision': results['test_metrics']['precision'],
                'Recall': results['test_metrics']['recall'],
                'F1-Score': results['test_metrics']['f1_score']
            })
        
        perf_df = pd.DataFrame(performance_data)
        
        # Create radar chart
        fig = go.Figure()
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for _, row in perf_df.iterrows():
            values = [row[metric] for metric in metrics]
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name=row['Model']
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Model Performance Radar Chart"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Data quality metrics
    st.subheader("🔍 Data Quality Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Completeness", f"{(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%")
    
    with col2:
        st.metric("Uniqueness", f"{df.nunique().sum() / (len(df) * len(df.columns)) * 100:.1f}%")
    
    with col3:
        st.metric("Consistency", f"{len(df) - len(df.drop_duplicates())} duplicates")

if __name__ == "__main__":
    main()
