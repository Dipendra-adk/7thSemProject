import numpy as np

class DecisionTreeRegressor:
    def __init__(self, max_depth=10, min_samples_split=5, min_samples_leaf=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def fit(self, X, y):
        # Convert DataFrame to numpy array
        X = np.array(X)
        y = np.array(y)
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split or np.var(y) == 0:
            return np.mean(y)

        best_feature, best_threshold = self._find_best_split(X, y)
        if best_feature is None:
            return np.mean(y)

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        return {
            "feature": best_feature,
            "threshold": best_threshold,
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._build_tree(X[right_mask], y[right_mask], depth + 1),
        }

    def _find_best_split(self, X, y):
        best_mse, best_feature, best_threshold = float("inf"), None, None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                mse = self._calculate_mse(y[left_mask], y[right_mask])
                if mse < best_mse:
                    best_mse, best_feature, best_threshold = mse, feature, threshold

        return best_feature, best_threshold

    def _calculate_mse(self, left_y, right_y):
        mse = sum(np.var(group) * len(group) for group in [left_y, right_y] if len(group) > 0)
        return mse

    def predict(self, X):
        X = np.array(X)  # Ensure input is numpy array
        return np.array([self._predict_sample(x, self.tree) for x in X])

    def _predict_sample(self, x, tree):
        if not isinstance(tree, dict):
            return tree
        if x[tree["feature"]] <= tree["threshold"]:
            return self._predict_sample(x, tree["left"])
        return self._predict_sample(x, tree["right"])

    # Add score method for evaluating R^2
    def score(self, X, y):
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)

    # Implementing get_params and set_params
    def get_params(self, deep=True):
        return {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
