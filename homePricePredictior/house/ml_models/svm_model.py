import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score

class SVMRegressor(BaseEstimator, RegressorMixin): 
    def __init__(self, C=1.0, epsilon=0.1, learning_rate=0.0001, n_epochs=1000):
        self.C = C
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights_ = None
        self.bias_ = None
        
    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0
        
        for epoch in range(self.n_epochs):
            y_pred = self._predict(X)
            
            # Compute gradients using epsilon-insensitive loss
            diff = y_pred - y
            loss_gradient = np.zeros_like(diff)
            mask = np.abs(diff) > self.epsilon
            loss_gradient[mask] = diff[mask]
            
            # Compute gradients with regularization
            grad_w = (1 / n_samples) * np.dot(X.T, loss_gradient) + self.C * self.weights_
            grad_b = np.mean(loss_gradient)
            
            # Update weights and bias
            self.weights_ -= self.learning_rate * grad_w
            self.bias_ -= self.learning_rate * grad_b
            
            # Clip weights to prevent numerical instability
            self.weights_ = np.clip(self.weights_, -1e6, 1e6)
            self.bias_ = np.clip(self.bias_, -1e6, 1e6)
            
            # Print progress (optional)
            if epoch % 100 == 0:
                loss = self._compute_loss(X, y)
                print(f"Epoch {epoch}: Loss = {loss:.4f}")
        
        return self
    
    def _predict(self, X):
        X = np.array(X)
        predictions = np.dot(X, self.weights_) + self.bias_
        return np.clip(predictions, -1e6, 1e6)
    
    def predict(self, X):

        check_is_fitted = getattr(self, 'weights_', None) is not None
        if not check_is_fitted:
            raise ValueError("Model needs to be fitted before making predictions.")
        return self._predict(X)
    
    def _compute_loss(self, X, y):
       
        y_pred = self._predict(X)
        diff = np.abs(y_pred - y) - self.epsilon
        loss = np.mean(np.maximum(0, diff) ** 2)
        regularization = 0.5 * self.C * np.sum(self.weights_ ** 2)
        return loss + regularization
    
    def score(self, X, y):
        return r2_score(y, self.predict(X))