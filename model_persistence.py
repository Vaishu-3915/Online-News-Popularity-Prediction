#!/usr/bin/env python3
"""
Model Persistence Module for Online News Popularity Analysis

This module handles saving and loading trained models to avoid retraining.
"""

import pickle
import joblib
import os
from datetime import datetime
import json


class ModelPersistence:
    """Handles saving and loading of trained models."""
    
    def __init__(self, models_dir="saved_models"):
        """
        Initialize ModelPersistence.
        
        Args:
            models_dir (str): Directory to save/load models
        """
        self.models_dir = models_dir
        self.ensure_models_dir()
    
    def ensure_models_dir(self):
        """Create models directory if it doesn't exist."""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
            print(f"Created models directory: {self.models_dir}")
    
    def save_model(self, model, model_name, metadata=None):
        """
        Save a trained model.
        
        Args:
            model: Trained model object
            model_name (str): Name for the model
            metadata (dict): Additional information about the model
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}.pkl"
        filepath = os.path.join(self.models_dir, filename)
        
        # Prepare save data
        save_data = {
            'model': model,
            'metadata': metadata or {},
            'timestamp': timestamp,
            'model_name': model_name
        }
        
        # Save model
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        # Save metadata separately for easy reading
        metadata_file = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump({
                'model_name': model_name,
                'timestamp': timestamp,
                'metadata': metadata or {}
            }, f, indent=2)
        
        print(f"Model saved: {filepath}")
        return filepath
    
    def load_model(self, model_name=None, timestamp=None):
        """
        Load a saved model.
        
        Args:
            model_name (str): Name of the model to load
            timestamp (str): Specific timestamp to load
            
        Returns:
            tuple: (model, metadata)
        """
        if model_name is None:
            # Load the most recent model
            models = self.list_saved_models()
            if not models:
                raise ValueError("No saved models found")
            model_name = models[0]['model_name']
            timestamp = models[0]['timestamp']
        
        if timestamp is None:
            # Load the most recent version of the model
            models = self.list_saved_models()
            matching_models = [m for m in models if m['model_name'] == model_name]
            if not matching_models:
                raise ValueError(f"No saved models found for {model_name}")
            timestamp = matching_models[0]['timestamp']
        
        filename = f"{model_name}_{timestamp}.pkl"
        filepath = os.path.join(self.models_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        print(f"Model loaded: {filepath}")
        return save_data['model'], save_data['metadata']
    
    def list_saved_models(self):
        """
        List all saved models.
        
        Returns:
            list: List of model information dictionaries
        """
        models = []
        
        for filename in os.listdir(self.models_dir):
            if filename.endswith('.pkl'):
                # Extract model name and timestamp
                parts = filename.replace('.pkl', '').split('_')
                if len(parts) >= 2:
                    timestamp = '_'.join(parts[-2:])  # Last two parts are date_time
                    model_name = '_'.join(parts[:-2])  # Everything else is model name
                    
                    models.append({
                        'model_name': model_name,
                        'timestamp': timestamp,
                        'filename': filename,
                        'filepath': os.path.join(self.models_dir, filename)
                    })
        
        # Sort by timestamp (newest first)
        models.sort(key=lambda x: x['timestamp'], reverse=True)
        return models
    
    def delete_model(self, model_name, timestamp):
        """
        Delete a saved model.
        
        Args:
            model_name (str): Name of the model
            timestamp (str): Timestamp of the model
        """
        filename = f"{model_name}_{timestamp}.pkl"
        filepath = os.path.join(self.models_dir, filename)
        metadata_file = filepath.replace('.pkl', '_metadata.json')
        
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"Deleted model: {filepath}")
        
        if os.path.exists(metadata_file):
            os.remove(metadata_file)
            print(f"Deleted metadata: {metadata_file}")
    
    def get_model_info(self, model_name, timestamp):
        """
        Get information about a saved model.
        
        Args:
            model_name (str): Name of the model
            timestamp (str): Timestamp of the model
            
        Returns:
            dict: Model information
        """
        metadata_file = os.path.join(self.models_dir, f"{model_name}_{timestamp}_metadata.json")
        
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                return json.load(f)
        else:
            return None


def save_training_results(results_dict, filename="training_results.json"):
    """
    Save training results to JSON file.
    
    Args:
        results_dict (dict): Dictionary containing model results
        filename (str): Output filename
    """
    # Convert numpy types to Python types for JSON serialization
    serializable_results = {}
    
    for model_name, (train_metrics, test_metrics) in results_dict.items():
        serializable_results[model_name] = {
            'train_metrics': {
                'accuracy': float(train_metrics['accuracy']),
                'precision': float(train_metrics['precision']),
                'recall': float(train_metrics['recall']),
                'f1_score': float(train_metrics['f1_score'])
            },
            'test_metrics': {
                'accuracy': float(test_metrics['accuracy']),
                'precision': float(test_metrics['precision']),
                'recall': float(test_metrics['recall']),
                'f1_score': float(test_metrics['f1_score'])
            }
        }
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Training results saved to {filename}")


def load_training_results(filename="training_results.json"):
    """
    Load training results from JSON file.
    
    Args:
        filename (str): Input filename
        
    Returns:
        dict: Training results
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Results file not found: {filename}")
    
    with open(filename, 'r') as f:
        return json.load(f)
