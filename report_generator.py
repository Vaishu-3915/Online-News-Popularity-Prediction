#!/usr/bin/env python3
"""
Report Generation Module for Online News Popularity Analysis

This module generates professional HTML and PDF reports
with comprehensive analysis results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import json
from datetime import datetime
import os
from jinja2 import Template
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from preprocessing import get_data_summary
from evaluation import compare_models, calculate_metrics


class ReportGenerator:
    """Generates professional reports for the analysis."""
    
    def __init__(self, output_dir="reports"):
        """
        Initialize ReportGenerator.
        
        Args:
            output_dir (str): Directory to save reports
        """
        self.output_dir = output_dir
        self.ensure_output_dir()
    
    def ensure_output_dir(self):
        """Create output directory if it doesn't exist."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created reports directory: {self.output_dir}")
    
    def plot_to_base64(self, fig):
        """
        Convert matplotlib figure to base64 string.
        
        Args:
            fig: Matplotlib figure
            
        Returns:
            str: Base64 encoded image
        """
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        return image_base64
    
    def generate_data_overview_plots(self, df):
        """Generate data overview plots."""
        plots = {}
        
        # Target distribution
        fig, ax = plt.subplots(figsize=(8, 6))
        df['popularity'].value_counts().plot(kind='bar', ax=ax)
        ax.set_title('Target Distribution')
        ax.set_xlabel('Popularity')
        ax.set_ylabel('Count')
        ax.set_xticklabels(['Not Popular', 'Popular'], rotation=0)
        plots['target_distribution'] = self.plot_to_base64(fig)
        
        # Feature correlation heatmap (sample of features)
        fig, ax = plt.subplots(figsize=(12, 10))
        feature_cols = [col for col in df.columns if col != 'popularity'][:10]  # First 10 features
        corr_matrix = df[feature_cols + ['popularity']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Feature Correlation Heatmap (Top 10 Features)')
        plots['correlation_heatmap'] = self.plot_to_base64(fig)
        
        # Feature distributions
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for i, col in enumerate(numeric_cols[:6]):
            if i < len(axes):
                df[col].hist(bins=30, ax=axes[i])
                axes[i].set_title(f'{col} Distribution')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
        
        # Remove empty subplots
        for i in range(len(numeric_cols), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plots['feature_distributions'] = self.plot_to_base64(fig)
        
        return plots
    
    def generate_model_performance_plots(self, results_dict):
        """Generate model performance plots."""
        plots = {}
        
        # Model comparison bar chart
        comparison_df = compare_models(results_dict)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics = ['Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i//2, i%2]
            bars = ax.bar(comparison_df['Model'], comparison_df[metric])
            ax.set_title(f'{name} Comparison')
            ax.set_ylabel(name)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plots['model_comparison'] = self.plot_to_base64(fig)
        
        # Performance radar chart
        fig = plt.figure(figsize=(10, 8))
        
        # Create radar chart data
        models = comparison_df['Model'].tolist()
        metrics_data = [comparison_df[metric].tolist() for metric in metrics]
        
        # Simple bar chart as radar alternative
        x = np.arange(len(models))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax.bar(x + i*width, comparison_df[metric], width, label=name)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plots['performance_radar'] = self.plot_to_base64(fig)
        
        return plots
    
    def generate_html_report(self, df, results_dict, cv_results=None, filename=None):
        """
        Generate comprehensive HTML report.
        
        Args:
            df (pd.DataFrame): Processed dataset
            results_dict (dict): Model results
            cv_results (dict): Cross-validation results
            filename (str): Output filename
            
        Returns:
            str: Path to generated HTML file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_report_{timestamp}.html"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Generate plots
        data_plots = self.generate_data_overview_plots(df)
        model_plots = self.generate_model_performance_plots(results_dict)
        
        # Get data summary
        data_summary = get_data_summary(df)
        
        # Generate comparison table
        comparison_df = compare_models(results_dict)
        
        # Find best model
        best_model = comparison_df.loc[comparison_df['Test_F1'].idxmax()]
        
        # HTML template
        html_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Online News Popularity Analysis Report</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                }
                .header {
                    text-align: center;
                    border-bottom: 3px solid #2c3e50;
                    padding-bottom: 20px;
                    margin-bottom: 30px;
                }
                .header h1 {
                    color: #2c3e50;
                    margin: 0;
                    font-size: 2.5em;
                }
                .header p {
                    color: #7f8c8d;
                    margin: 10px 0 0 0;
                    font-size: 1.1em;
                }
                .section {
                    margin: 30px 0;
                    padding: 20px;
                    border-left: 4px solid #3498db;
                    background-color: #f8f9fa;
                }
                .section h2 {
                    color: #2c3e50;
                    margin-top: 0;
                    border-bottom: 2px solid #ecf0f1;
                    padding-bottom: 10px;
                }
                .metric-card {
                    background-color: white;
                    padding: 20px;
                    margin: 10px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    display: inline-block;
                    min-width: 150px;
                    text-align: center;
                }
                .metric-value {
                    font-size: 2em;
                    font-weight: bold;
                    color: #2c3e50;
                }
                .metric-label {
                    color: #7f8c8d;
                    font-size: 0.9em;
                    margin-top: 5px;
                }
                .plot-container {
                    text-align: center;
                    margin: 20px 0;
                }
                .plot-container img {
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 8px;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }
                th, td {
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                th {
                    background-color: #f2f2f2;
                    font-weight: bold;
                }
                .best-model {
                    background-color: #d4edda;
                    border-left: 4px solid #28a745;
                }
                .footer {
                    text-align: center;
                    margin-top: 40px;
                    padding-top: 20px;
                    border-top: 2px solid #ecf0f1;
                    color: #7f8c8d;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📰 Online News Popularity Analysis</h1>
                    <p>Comprehensive Machine Learning Analysis Report</p>
                    <p>Generated on: {{ timestamp }}</p>
                </div>
                
                <div class="section">
                    <h2>📊 Dataset Overview</h2>
                    <div class="metric-card">
                        <div class="metric-value">{{ data_summary.shape[0] }}</div>
                        <div class="metric-label">Total Records</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{{ data_summary.shape[1] - 1 }}</div>
                        <div class="metric-label">Features</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{{ data_summary.target_distribution[1] }}</div>
                        <div class="metric-label">Popular Articles</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{{ "%.1f"|format(data_summary.target_distribution[1] / sum(data_summary.target_distribution.values()) * 100) }}%</div>
                        <div class="metric-label">Popularity Rate</div>
                    </div>
                    
                    <h3>Target Distribution</h3>
                    <div class="plot-container">
                        <img src="data:image/png;base64,{{ data_plots.target_distribution }}" alt="Target Distribution">
                    </div>
                    
                    <h3>Feature Correlation</h3>
                    <div class="plot-container">
                        <img src="data:image/png;base64,{{ data_plots.correlation_heatmap }}" alt="Correlation Heatmap">
                    </div>
                    
                    <h3>Feature Distributions</h3>
                    <div class="plot-container">
                        <img src="data:image/png;base64,{{ data_plots.feature_distributions }}" alt="Feature Distributions">
                    </div>
                </div>
                
                <div class="section">
                    <h2>🤖 Model Performance Results</h2>
                    
                    <h3>Performance Summary</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Model</th>
                                <th>Test Accuracy</th>
                                <th>Test Precision</th>
                                <th>Test Recall</th>
                                <th>Test F1-Score</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for _, row in comparison_df.iterrows() %}
                            <tr {% if row['Model'] == best_model['Model'] %}class="best-model"{% endif %}>
                                <td>{{ row['Model'] }}</td>
                                <td>{{ "%.4f"|format(row['Test_Accuracy']) }}</td>
                                <td>{{ "%.4f"|format(row['Test_Precision']) }}</td>
                                <td>{{ "%.4f"|format(row['Test_Recall']) }}</td>
                                <td>{{ "%.4f"|format(row['Test_F1']) }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                    
                    <h3>Model Comparison Visualization</h3>
                    <div class="plot-container">
                        <img src="data:image/png;base64,{{ model_plots.model_comparison }}" alt="Model Comparison">
                    </div>
                    
                    <div class="plot-container">
                        <img src="data:image/png;base64,{{ model_plots.performance_radar }}" alt="Performance Radar">
                    </div>
                    
                    <div class="best-model" style="padding: 20px; margin: 20px 0;">
                        <h3>🏆 Best Performing Model</h3>
                        <p><strong>{{ best_model['Model'] }}</strong> achieved the highest F1-Score of <strong>{{ "%.4f"|format(best_model['Test_F1']) }}</strong></p>
                        <ul>
                            <li>Accuracy: {{ "%.4f"|format(best_model['Test_Accuracy']) }}</li>
                            <li>Precision: {{ "%.4f"|format(best_model['Test_Precision']) }}</li>
                            <li>Recall: {{ "%.4f"|format(best_model['Test_Recall']) }}</li>
                        </ul>
                    </div>
                </div>
                
                {% if cv_results %}
                <div class="section">
                    <h2>📈 Cross-Validation Analysis</h2>
                    <p>Cross-validation provides more robust performance estimates by evaluating models on multiple data splits.</p>
                    
                    {% for model_name, cv_result in cv_results.items() %}
                    <h3>{{ model_name }}</h3>
                    <div class="metric-card">
                        <div class="metric-value">{{ "%.4f"|format(cv_result.summary.f1_score.mean) }}</div>
                        <div class="metric-label">Mean F1-Score</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{{ "%.4f"|format(cv_result.summary.f1_score.std) }}</div>
                        <div class="metric-label">F1-Score Std Dev</div>
                    </div>
                    {% endfor %}
                </div>
                {% endif %}
                
                <div class="section">
                    <h2>💡 Key Insights</h2>
                    <ul>
                        <li>The dataset contains {{ data_summary.shape[0] }} articles with {{ data_summary.shape[1] - 1 }} features</li>
                        <li>{{ "%.1f"|format(data_summary.target_distribution[1] / sum(data_summary.target_distribution.values()) * 100) }}% of articles are classified as popular</li>
                        <li><strong>{{ best_model['Model'] }}</strong> performs best with an F1-Score of {{ "%.4f"|format(best_model['Test_F1']) }}</li>
                        {% if cv_results %}
                        <li>Cross-validation confirms model performance stability</li>
                        {% endif %}
                    </ul>
                </div>
                
                <div class="footer">
                    <p>Report generated by Online News Popularity Analysis System</p>
                    <p>For questions or more information, please refer to the project documentation</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Render template
        template = Template(html_template)
        html_content = template.render(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            data_summary=data_summary,
            data_plots=data_plots,
            model_plots=model_plots,
            comparison_df=comparison_df,
            best_model=best_model,
            cv_results=cv_results
        )
        
        # Save HTML file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML report generated: {filepath}")
        return filepath
    
    def generate_summary_report(self, df, results_dict, filename=None):
        """
        Generate a concise summary report.
        
        Args:
            df (pd.DataFrame): Processed dataset
            results_dict (dict): Model results
            filename (str): Output filename
            
        Returns:
            str: Path to generated report
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"summary_report_{timestamp}.txt"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Get data summary
        data_summary = get_data_summary(df)
        
        # Generate comparison
        comparison_df = compare_models(results_dict)
        best_model = comparison_df.loc[comparison_df['Test_F1'].idxmax()]
        
        # Generate report content
        report_content = f"""
ONLINE NEWS POPULARITY ANALYSIS - SUMMARY REPORT
{'='*60}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

DATASET OVERVIEW:
{'-'*30}
Total Records: {data_summary.shape[0]:,}
Features: {data_summary.shape[1] - 1}
Popular Articles: {data_summary.target_distribution[1]:,}
Popularity Rate: {data_summary.target_distribution[1] / sum(data_summary.target_distribution.values()) * 100:.1f}%

MODEL PERFORMANCE:
{'-'*30}
"""
        
        for _, row in comparison_df.iterrows():
            report_content += f"""
{row['Model']}:
  Accuracy:  {row['Test_Accuracy']:.4f}
  Precision: {row['Test_Precision']:.4f}
  Recall:    {row['Test_Recall']:.4f}
  F1-Score:  {row['Test_F1']:.4f}
"""
        
        report_content += f"""

BEST MODEL: {best_model['Model']}
F1-Score: {best_model['Test_F1']:.4f}

KEY INSIGHTS:
{'-'*30}
• Dataset is {'balanced' if abs(data_summary.target_distribution[1] - data_summary.target_distribution[0]) < 1000 else 'imbalanced'}
• Best performing model: {best_model['Model']}
• Performance gap between models: {comparison_df['Test_F1'].max() - comparison_df['Test_F1'].min():.4f}

RECOMMENDATIONS:
{'-'*30}
• Consider ensemble methods for improved performance
• Investigate feature importance for model interpretability
• Validate results with cross-validation
• Monitor model performance on new data

{'='*60}
Report generated by Online News Popularity Analysis System
"""
        
        # Save report
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"Summary report generated: {filepath}")
        return filepath


def generate_all_reports(df, results_dict, cv_results=None):
    """
    Generate all types of reports.
    
    Args:
        df (pd.DataFrame): Processed dataset
        results_dict (dict): Model results
        cv_results (dict): Cross-validation results
        
    Returns:
        dict: Paths to generated reports
    """
    generator = ReportGenerator()
    
    reports = {}
    
    # Generate HTML report
    reports['html'] = generator.generate_html_report(df, results_dict, cv_results)
    
    # Generate summary report
    reports['summary'] = generator.generate_summary_report(df, results_dict)
    
    return reports
