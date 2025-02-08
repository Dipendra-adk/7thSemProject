import numpy as np

class TreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeRegressor:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape

        if num_samples < self.min_samples_split or depth >= self.max_depth:
            return TreeNode(value=np.mean(y))

        best_feature, best_thresh, best_mse = None, None, float("inf")
        best_splits = None

        for feature_index in range(num_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = X[:, feature_index] > threshold

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]
                mse = np.var(y_left) * len(y_left) + np.var(y_right) * len(y_right)

                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature_index
                    best_thresh = threshold
                    best_splits = (X[left_mask], y_left, X[right_mask], y_right)

        if best_feature is None:
            return TreeNode(value=np.mean(y))

        left_node = self._build_tree(best_splits[0], best_splits[1], depth + 1)
        right_node = self._build_tree(best_splits[2], best_splits[3], depth + 1)
        return TreeNode(feature_index=best_feature, threshold=best_thresh, left=left_node, right=right_node)

    def _predict_sample(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict_sample(x, self.root) for x in X])
