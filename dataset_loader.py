import numpy as np
import os
import config

# Try to import sklearn for LIBSVM handling
try:
    from sklearn.datasets import load_svmlight_file
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Global context
_DATA_CONTEXT = None

def load_data(dataset_name):
    """
    Factory function to initialize data and return the gradient accessor.
    """
    global _DATA_CONTEXT
    
    if dataset_name == 'synthetic_regression':
        if _DATA_CONTEXT is None:
            print(f"Generating Synthetic Linear Regression Data (Dim={config.DIMENSION})...")
            _DATA_CONTEXT = SyntheticLinearRegression(
                num_agents=config.NUM_AGENTS,
                dimension=config.DIMENSION,
                samples_per_agent=100
            )
        return _DATA_CONTEXT.get_gradient

    elif dataset_name == 'libsvm':
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for LIBSVM. Install via 'pip install scikit-learn'")
        
        if _DATA_CONTEXT is None:
            # You must define LIBSVM_PATH in config.py or pass it here
            # For prototype, we default to a dummy filename if not in config
            path = getattr(config, 'LIBSVM_PATH', 'data/a9a.txt') 
            print(f"Loading LIBSVM Dataset from {path}...")
            
            _DATA_CONTEXT = LibSVMClassification(
                filename=path,
                num_agents=config.NUM_AGENTS,
                expected_dim=config.DIMENSION
            )
        return _DATA_CONTEXT.get_gradient
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def load_loss_func(dataset_name):

    """Returns the loss calculation function for the current dataset."""
    if _DATA_CONTEXT is None:
        # Ensure data is loaded if this is called before load_data
        load_data(dataset_name) 
    return _DATA_CONTEXT.get_loss

# ==============================================================================
# 1. Synthetic Linear Regression (Already implemented)
class SyntheticLinearRegression:
    def __init__(self, num_agents, dimension, samples_per_agent):
        self.true_weights = np.random.randn(dimension)
        self.agent_data = []
        for i in range(num_agents):
            A_i = np.random.randn(samples_per_agent, dimension)
            # Label noise (Gaussian), distinct from the heavy-tail gradient noise
            label_noise = np.random.normal(0, 0.1, samples_per_agent)
            b_i = np.dot(A_i, self.true_weights) + label_noise
            self.agent_data.append((A_i, b_i))
            
    def get_gradient(self, agent_id, params):
        # Gradient of MSE: 2/N * A^T * (Ax - b)
        A_i, b_i = self.agent_data[agent_id]
        errors = np.dot(A_i, params) - b_i
        grad = (2.0 / len(b_i)) * np.dot(A_i.T, errors)
        return grad
    
    def get_loss(self, agent_id, params):
        """Calculates Mean Squared Error (MSE)"""
        A_i, b_i = self.agent_data[agent_id]
        predictions = np.dot(A_i, params)
        errors = predictions - b_i
        return np.mean(errors**2)

# ==============================================================================
# 2. LIBSVM Classification (Logistic Regression)
class LibSVMClassification:
    """
    Handles Binary Classification using Logistic Regression on LIBSVM data.
    Minimizes Logistic Loss: f(w) = sum( log(1 + exp(-y * w^T x)) ) + lambda ||w||^2
    """
    def __init__(self, filename, num_agents, expected_dim):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"LIBSVM file not found at: {filename}. Please download a dataset (e.g., 'a9a', 'w8a') from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/")

        # 1. Load Data
        # n_features ensures fixed dimension even if file is sparse
        X_sparse, y = load_svmlight_file(filename, n_features=expected_dim)
        
        # Convert to dense (for simple prototype math)
        X = X_sparse.toarray()
        
        # 2. Preprocessing
        # Fix labels to be {-1, 1} for logistic loss formulation
        y = np.where(y <= 0, -1, 1)
        
        # Standardize features (Critical for gradient descent stability)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Add bias term (intercept) by appending 1s? 
        # For this prototype, we assume bias is part of weights or data is centered.
        
        # 3. Distribute to Agents
        # We perform a random split to simulate IID or non-IID data
        self.agent_data = []
        
        # Indices for each agent
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        chunks = np.array_split(indices, num_agents)
        
        for i in range(num_agents):
            idx = chunks[i]
            self.agent_data.append((X[idx], y[idx]))
            
        print(f"Data distributed. Agent 0 has {len(chunks[0])} samples.")

    def get_gradient(self, agent_id, params):
        """
        Computes gradient of Logistic Loss for agent_id.
        Loss_i = (1/m) * sum( log(1 + exp(-y_j * x_j . w)) ) + (lambda/2)||w||^2
        Grad_i = (1/m) * sum( -y_j * x_j * sigmoid(-y_j * x_j . w) ) + lambda * w
        """
        X_i, y_i = self.agent_data[agent_id]
        m = len(y_i)
        if m == 0: return np.zeros_like(params)
        
        # 1. Compute scores z = y * (X . w)
        # Note: params shape (D,), X_i shape (m, D)
        scores = y_i * np.dot(X_i, params)
        
        # 2. Compute sigmoid(-z) = 1 / (1 + exp(z))
        # Stable sigmoid computation
        # exp(-z) can explode if z is very negative (score is negative)
        # We use scipy's expit or a stable numpy impl
        # Sigmoid(-z)
        sigmoid_val = 1.0 / (1.0 + np.exp(scores)) 
        
        # 3. Compute Gradient component
        # -y * x * sigmoid
        # We can broadcast: (m, 1) * (m, D)
        coeffs = -y_i * sigmoid_val
        grad_sum = np.dot(coeffs, X_i) # Shape (D,)
        
        # 4. Average + Regularization
        LAMBDA = 0.001 # L2 regularization strength
        grad = (grad_sum / m) + (LAMBDA * params)
        
        return grad
    
    def get_loss(self, agent_id, params):
        """Calculates Logistic Loss + L2 Regularization"""
        X_i, y_i = self.agent_data[agent_id]
        # Robust log-sum-exp for log(1 + exp(-y * score))
        scores = y_i * np.dot(X_i, params)
        loss_val = np.logaddexp(0, -scores)
        
        # Add Regularization (Lambda = 0.001) matching the gradient function
        reg_loss = (0.001 / 2) * np.sum(params**2)
        return np.mean(loss_val) + reg_loss

# ==============================================================================
# SHAYAN:

# 3. mu-strongly convex
# 4. non-convex function    