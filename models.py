#!/usr/bin/env python3
"""
Machine Learning Models Module for Online News Popularity Analysis

This module contains implementations of various machine learning algorithms
including Logistic Regression, Naive Bayes, Neural Network, and SVM.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# PyTorch imports (optional, will be imported when needed)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy classes to avoid NameError
    class nn:
        class Module:
            pass
        class Linear:
            pass
        class ReLU:
            pass
        class Sigmoid:
            pass
        class BCELoss:
            pass
    class optim:
        class Adam:
            pass


class LogisticRegression:
    """Implementation of Logistic Regression from scratch."""
    
    def __init__(self, train_data, test_data, learning_rate=0.01, epsilon=1e-6, max_iteration=1000):
        """
        Initialize Logistic Regression model.
        
        Args:
            train_data (pd.DataFrame): Training data
            test_data (pd.DataFrame): Test data
            learning_rate (float): Learning rate for gradient descent
            epsilon (float): Convergence threshold
            max_iteration (int): Maximum number of iterations
        """
        self.X_train = train_data.drop('popularity', axis=1)
        self.y_train = train_data['popularity']
        self.X_test = test_data.drop('popularity', axis=1)
        self.y_test = test_data['popularity']
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.max_iteration = max_iteration
        self.w = None
        
    def add_X0(self, X):
        """Add bias term to features."""
        return np.column_stack([np.ones(X.shape[0]), X])
    
    def train_scaling(self, X):
        """Scale continuous features for training."""
        columns = X.columns
        self.numeric_cols = []
        for col in columns:
            if X[col].nunique() > 2:
                self.numeric_cols.append(col)
        
        self.mean = np.mean(X[self.numeric_cols], axis=0)
        self.std = np.std(X[self.numeric_cols], axis=0)
        X[self.numeric_cols] = (X[self.numeric_cols] - self.mean) / self.std
        X = self.add_X0(X)
        return X
    
    def test_scaling(self, X):
        """Scale continuous features for testing."""
        X[self.numeric_cols] = (X[self.numeric_cols] - self.mean) / self.std
        X = self.add_X0(X)
        return X
    
    def sigmoid(self, z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def predict(self, X):
        """Make predictions."""
        sig = self.sigmoid(X.dot(self.w))
        return np.around(sig)
    
    def cost_function(self, X, y):
        """Calculate log loss."""
        sig = self.sigmoid(X.dot(self.w))
        loss = y * np.log(sig + 1e-15) + (1 - y) * np.log(1 - sig + 1e-15)
        cost = -1 * loss.sum() / len(y)
        return cost
    
    def cost_derivative(self, X, y):
        """Calculate gradient."""
        sig = self.sigmoid(X.dot(self.w))
        gradient = (sig - y).dot(X)
        return gradient
    
    def gradient_descent(self, X, y):
        """Perform gradient descent optimization."""
        errors = []
        prev_error = float("inf")
        self.w = np.ones(X.shape[1])
        
        for iteration in tqdm(range(self.max_iteration), desc="Training"):
            self.w -= self.learning_rate * self.cost_derivative(X, y)
            curr_error = self.cost_function(X, y)
            errors.append(curr_error)
            
            if abs(prev_error - curr_error) < self.epsilon:
                print("Model converged early")
                break
            else:
                prev_error = curr_error
        
        return errors
    
    def fit(self):
        """Train the model."""
        print("Training Logistic Regression...")
        self.X_train_scaled = self.train_scaling(self.X_train)
        self.X_test_scaled = self.test_scaling(self.X_test)
        
        errors = self.gradient_descent(self.X_train_scaled, self.y_train)
        print("Training completed.")
        return errors


class NaiveBayes:
    """Implementation of Naive Bayes from scratch."""
    
    def __init__(self, train_data, test_data):
        """Initialize Naive Bayes model."""
        self.X_train_cont, self.X_train_cat = self.extract_features(train_data)
        self.y_train = train_data['popularity'].to_numpy()
        self.X_test_cont, self.X_test_cat = self.extract_features(test_data)
        self.y_test = test_data['popularity'].to_numpy()
        self.prior = None
        self.cont_distribution = None
        self.cat_counters = None
    
    def extract_features(self, data):
        """Separate continuous and categorical features."""
        X_cont = []
        X_cat = []
        for col in data.columns:
            if data[col].nunique() <= 2:
                X_cat.append(data[col].to_numpy().reshape(-1, 1))
            else:
                X_cont.append(data[col].to_numpy().reshape(-1, 1))
        return np.hstack(X_cont), np.hstack(X_cat)
    
    def gaussian_fit(self, X):
        """Calculate mean and standard deviation."""
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return mean, std
    
    def single_feature_counter(self, x_j):
        """Count unique values in a column."""
        value, counts = np.unique(x_j, return_counts=True)
        count_dict = defaultdict(int, zip(value, counts))
        return count_dict
    
    def fit(self):
        """Train the Naive Bayes model."""
        print("Training Naive Bayes...")
        self.classes, counts = np.unique(self.y_train, return_counts=True)
        self.prior = counts / sum(counts)
        
        # Fit Gaussian for continuous features
        self.cont_distribution = {}
        for class_k in self.classes:
            X_cont = self.X_train_cont[self.y_train == class_k]
            self.cont_distribution[class_k] = np.apply_along_axis(self.gaussian_fit, 0, X_cont)
        
        # Fit discrete for categorical features
        self.cat_counters = {}
        for class_k in self.classes:
            X_cat = self.X_train_cat[self.y_train == class_k]
            self.cat_counters[class_k] = np.apply_along_axis(self.single_feature_counter, 0, X_cat)
        
        print("Training completed.")
    
    def gaussian(self, X, mean, std):
        """Calculate Gaussian probability."""
        return np.exp(-(X - mean) ** 2 / (2 * std ** 2)) / (np.sqrt(2 * np.pi) * std)
    
    def laplace_smoothing(self, feature_value, feature_count_dict):
        """Apply Laplace smoothing."""
        return (feature_count_dict[feature_value] + 1) / (sum(feature_count_dict.values()) + len(feature_count_dict))
    
    def predict(self, X_cont, X_cat):
        """Make predictions."""
        y_pred = []
        for i in range(len(X_cont)):
            scores = {}
            for class_k in self.classes:
                # Continuous likelihood
                cont_likelihood = np.prod([
                    self.gaussian(X_cont[i, j], self.cont_distribution[class_k][0][j], 
                                self.cont_distribution[class_k][1][j])
                    for j in range(X_cont.shape[1])
                ])
                
                # Categorical likelihood
                cat_likelihood = np.prod([
                    self.laplace_smoothing(X_cat[i, j], self.cat_counters[class_k][j])
                    for j in range(X_cat.shape[1])
                ])
                
                posterior = cont_likelihood * cat_likelihood * self.prior[class_k]
                scores[class_k] = posterior
            
            y_pred.append(max(scores, key=scores.get))
        
        return np.array(y_pred)


class FeedforwardNeuralNetwork:
    """PyTorch implementation of Feedforward Neural Network."""
    
    def __init__(self, input_size, hidden_size, output_size):
        """Initialize neural network."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for Neural Network")
        
        # Now we know PyTorch is available, so we can use nn.Module
        class NeuralNetwork(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(NeuralNetwork, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(hidden_size, output_size)
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, X):
                X = self.fc1(X)
                X = self.relu(X)
                X = self.fc2(X)
                X = self.sigmoid(X)
                return X
        
        self.network = NeuralNetwork(input_size, hidden_size, output_size)
    
    def forward(self, X):
        """Forward pass."""
        return self.network.forward(X)
    
    def predict(self, X):
        """Make predictions."""
        return self.network.forward(X)
    
    def calculate_metrics(self, pred, label):
        """Calculate accuracy, precision, recall, and F1-score."""
        pred = torch.round(pred).squeeze()
        label = label.squeeze()
        
        accuracy = (pred == label).sum().item() / len(label)
        
        true_positives = ((pred == 1) & (label == 1)).sum().item()
        false_positives = ((pred == 1) & (label == 0)).sum().item()
        false_negatives = ((pred == 0) & (label == 1)).sum().item()
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return accuracy, precision, recall, f1_score


class NeuralNetworkTrainer:
    """Trainer class for Neural Network."""
    
    def __init__(self, train_data, test_data, hidden_size=128, learning_rate=0.001, epochs=10):
        """Initialize trainer."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for Neural Network training")
            
        self.train_data = train_data
        self.test_data = test_data
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        # Prepare data
        self.X_train = train_data.drop('popularity', axis=1).to_numpy()
        self.y_train = train_data['popularity'].to_numpy()
        self.X_test = test_data.drop('popularity', axis=1).to_numpy()
        self.y_test = test_data['popularity'].to_numpy()
        
        # Convert to tensors
        train_dataset = TensorDataset(torch.from_numpy(self.X_train), torch.from_numpy(self.y_train))
        test_dataset = TensorDataset(torch.from_numpy(self.X_test), torch.from_numpy(self.y_test))
        
        self.train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32)
        self.test_loader = DataLoader(test_dataset, shuffle=True, batch_size=32)
        
        # Initialize model
        input_dim = self.X_train.shape[1]
        self.model = FeedforwardNeuralNetwork(input_dim, hidden_size, 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.network.to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.model.network.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
    
    def train(self):
        """Train the neural network."""
        print("Training Neural Network...")
        self.model.network.train()
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            all_preds = []
            all_labels = []
            
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.float().to(self.device), labels.float().to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model.network(inputs)
                loss = self.criterion(outputs, labels.unsqueeze(1))
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                all_preds.extend(outputs.detach().cpu().squeeze())
                all_labels.extend(labels.detach().cpu())
            
            # Calculate metrics
            all_preds = torch.tensor(all_preds)
            all_labels = torch.tensor(all_labels)
            accuracy, precision, recall, f1_score = self.model.calculate_metrics(all_preds, all_labels)
            
            print(f'Epoch {epoch+1}/{self.epochs}')
            print(f'Loss: {epoch_loss/len(self.train_loader):.4f}')
            print(f'Accuracy: {accuracy:.4f}')
            print(f'Precision: {precision:.4f}')
            print(f'Recall: {recall:.4f}')
            print(f'F1-Score: {f1_score:.4f}')
            print('-' * 50)
        
        print("Training completed.")
    
    def evaluate(self):
        """Evaluate the model on test data."""
        print("\nEvaluating Neural Network...")
        self.model.network.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.float().to(self.device), labels.float().to(self.device)
                outputs = self.model.network(inputs)
                
                all_preds.extend(outputs.cpu().squeeze())
                all_labels.extend(labels.cpu())
        
        all_preds = torch.tensor(all_preds)
        all_labels = torch.tensor(all_labels)
        accuracy, precision, recall, f1_score = self.model.calculate_metrics(all_preds, all_labels)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }


class KernelSVM:
    """Implementation of Support Vector Machine with kernel functions."""
    
    def __init__(self, train_data, test_data, kernel='linear', C=1, max_iter=10):
        """Initialize SVM model."""
        self.kernel = kernel
        self.C = C
        self.max_iter = max_iter
        
        # Convert target to -1, 1 for SVM
        train_data_copy = train_data.copy()
        test_data_copy = test_data.copy()
        train_data_copy['popularity'] = train_data_copy['popularity'].map({0: -1, 1: 1})
        test_data_copy['popularity'] = test_data_copy['popularity'].map({0: -1, 1: 1})
        
        self.X_train = train_data_copy.drop('popularity', axis=1)
        self.y_train = train_data_copy['popularity']
        self.X_test = test_data_copy.drop('popularity', axis=1)
        self.y_test = test_data_copy['popularity']
        
        # Scale features
        self.X_train = self.train_scaling(self.X_train)
        self.X_test = self.test_scaling(self.X_test)
        
        self.fit()
    
    def kernel_function(self, X1, X2=None):
        """Calculate kernel matrix."""
        if X2 is None:
            X2 = X1
        
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'polynomial':
            return (np.dot(X1, X2.T) + 1) ** 2
        elif self.kernel == 'rbf':
            gamma = 1 / X1.shape[1]
            return np.exp(-gamma * np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2, axis=2))
        else:
            raise ValueError('Invalid kernel function.')
    
    def train_scaling(self, X):
        """Scale features for training."""
        columns = X.columns
        self.numeric_cols = []
        for col in columns:
            if X[col].nunique() > 2:
                self.numeric_cols.append(col)
        
        self.mean = np.mean(X[self.numeric_cols], axis=0)
        self.std = np.std(X[self.numeric_cols], axis=0)
        X[self.numeric_cols] = (X[self.numeric_cols] - self.mean) / self.std
        return X
    
    def test_scaling(self, X):
        """Scale features for testing."""
        X[self.numeric_cols] = (X[self.numeric_cols] - self.mean) / self.std
        return X
    
    def fit(self):
        """Train SVM using SMO algorithm."""
        print("Training SVM...")
        X = self.X_train.values
        y = self.y_train.values
        
        self.n, self.d = X.shape
        
        # Kernel matrix
        self.K = self.kernel_function(X)
        
        # Initialize variables
        self.alpha = np.random.rand(self.n) * 2 * self.C - self.C
        self.b = 0
        tol = 1e-3
        passes = 0
        iter_count = 0
        
        while passes < self.n and iter_count < self.max_iter:
            num_changed_alphas = 0
            for i in range(self.n):
                Ei = np.sum(self.alpha * y * self.K[:, i]) + self.b - y[i]
                if (y[i] * Ei < -tol and self.alpha[i] < self.C) or (y[i] * Ei > tol and self.alpha[i] > 0):
                    j = np.random.choice([j for j in range(self.n) if j != i])
                    Ej = np.sum(self.alpha * y * self.K[:, j]) + self.b - y[j]
                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]
                    
                    # Compute bounds
                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[j] + self.alpha[i] - self.C)
                        H = min(self.C, self.alpha[j] + self.alpha[i])
                    
                    if L == H:
                        continue
                    
                    eta = self.K[i, i] + self.K[j, j] - 2 * self.K[i, j]
                    if eta <= 0:
                        continue
                    
                    self.alpha[j] = self.alpha[j] - y[j] * (Ei - Ej) / eta
                    self.alpha[j] = max(L, min(H, self.alpha[j]))
                    
                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    self.alpha[i] = self.alpha[i] + y[i] * y[j] * (alpha_j_old - self.alpha[j])
                    
                    # Update bias
                    b1 = self.b - Ei - y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, i] - y[j] * (self.alpha[j] - alpha_j_old) * self.K[i, j]
                    b2 = self.b - Ej - y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, j] - y[j] * (self.alpha[j] - alpha_j_old) * self.K[j, j]
                    
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                    
                    num_changed_alphas += 1
            
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
            iter_count += 1
        
        print("Training completed.")
    
    def predict(self, X_new):
        """Make predictions."""
        if self.alpha is None or self.b is None:
            raise ValueError("Model not fitted yet.")
        
        K_new = self.kernel_function(X_new.values, self.X_train.values)
        y_pred = np.sign(np.dot(self.alpha, K_new.T) + self.b)
        return y_pred
