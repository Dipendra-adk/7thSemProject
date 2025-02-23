import numpy as np

class SVR:
    def __init__(self, kernel='rbf', C=5.0, epsilon=0.05, learning_rate=0.01, n_epochs=2000, decay=0.99, momentum=0.9,gamma='scale'):
        self.C = C
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.decay = decay
        self.momentum = momentum
        self.w = None
        self.b = None
        self.velocity_w = None
        self.velocity_b = None
        self.kernel = kernel
        self.gamma = gamma

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0.0
        self.velocity_w = np.zeros(n_features)
        self.velocity_b = 0.0

        for epoch in range(self.n_epochs):
            y_pred = np.dot(X, self.w) + self.b
            errors = y_pred - y

            grad_loss = np.where(errors > self.epsilon, 1, 0) - np.where(errors < -self.epsilon, 1, 0)
            grad_w = self.w + self.C * np.dot(grad_loss, X) / n_samples
            grad_b = self.C * np.mean(grad_loss)

            self.velocity_w = self.momentum * self.velocity_w - self.learning_rate * grad_w
            self.velocity_b = self.momentum * self.velocity_b - self.learning_rate * grad_b

            self.w += self.velocity_w
            self.b += self.velocity_b

            self.learning_rate *= self.decay

    def predict(self, X):
        return np.dot(X, self.w) + self.b
