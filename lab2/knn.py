import numpy as np


class KNN:
    def __init__(self, k=5, task="classification", distance="euclidean"):
        self.k = k
        self.task = task
        self.distance = distance

        if task not in ["classification", "regression"]:
            raise ValueError("task должен быть classification или regression")

        if distance not in ["euclidean", "manhattan"]:
            raise ValueError("distance должен быть euclidean или manhattan")

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        return self

    def _calculate_distances(self, x):
        if self.distance == "euclidean":
            return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

        return np.sum(np.abs(self.X_train - x), axis=1)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        predictions = []

        for x in X:
            distances = self._calculate_distances(x)
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_targets = self.y_train[nearest_indices]

            if self.task == "classification":
                classes, counts = np.unique(nearest_targets, return_counts=True)
                prediction = classes[np.argmax(counts)]
            else:
                prediction = np.mean(nearest_targets)

            predictions.append(prediction)

        return np.array(predictions)
