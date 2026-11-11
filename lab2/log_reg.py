import numpy as np


class LogisticRegression:
    def __init__(
        self,
        learning_rate=0.1,
        epochs=1000,
        threshold=0.5,
        regularization="none",
        lambda_=0.0,
        l1_ratio=0.5,
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.threshold = threshold
        self.regularization = regularization
        self.lambda_ = lambda_
        self.l1_ratio = l1_ratio

        self.weights = None
        self.bias = 0
        self.loss_history = []

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _regularization(self):
        reg_loss = 0
        reg_gradient = np.zeros_like(self.weights)

        if self.regularization == "l1":
            reg_loss = self.lambda_ * np.sum(np.abs(self.weights))
            reg_gradient = self.lambda_ * np.sign(self.weights)

        elif self.regularization == "l2":
            reg_loss = self.lambda_ * np.sum(self.weights ** 2)
            reg_gradient = 2 * self.lambda_ * self.weights

        elif self.regularization == "elasticnet":
            l1 = self.l1_ratio * np.sum(np.abs(self.weights))
            l2 = (1 - self.l1_ratio) * np.sum(self.weights ** 2)
            reg_loss = self.lambda_ * (l1 + l2)

            reg_gradient = self.lambda_ * (
                self.l1_ratio * np.sign(self.weights)
                + 2 * (1 - self.l1_ratio) * self.weights
            )

        return reg_loss, reg_gradient

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

            reg_loss, reg_gradient = self._regularization()
            loss = -np.mean(
                y * np.log(safe_probabilities)
                + (1 - y) * np.log(1 - safe_probabilities)
            ) + reg_loss
            self.loss_history.append(loss)

            error = probabilities - y
            dw = X.T @ error / n_samples + reg_gradient
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
