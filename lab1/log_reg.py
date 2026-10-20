import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.1, epochs=1000, threshold=0.5):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.threshold = threshold

        self.weights = None
        self.bias = 0
        self.loss_history = []

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500) # чтобы переполнения не было
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0
        self.loss_history = []

        for _ in range(self.epochs):
            z = X @ self.weights + self.bias
            probabilities = self._sigmoid(z)

            safe_probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)

            loss = -np.mean(
                y * np.log(safe_probabilities) + (1 - y) * np.log(1 - safe_probabilities)
            )

            self.loss_history.append(loss)

            error = probabilities - y

            dw = X.T @ error / n_samples
            db = np.mean(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)

        z = X @ self.weights + self.bias
        return self._sigmoid(z)


    def predict(self, X):
        probabilities = self.predict_proba(X)

        return (probabilities >= self.threshold).astype(int)