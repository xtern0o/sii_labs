import numpy as np

class LinearSVM:
    """Linear SVM with lambda"""
    def __init__(
        self,
        learning_rate=0.01,
        epochs=1000,
        lambda_=0.01,
        batch_size=32,
        seed=67,
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_ = lambda_
        self.batch_size = batch_size
        self.seed = seed

        self.weights = None
        self.bias = 0
        self.hinge_loss_history = []
        self.loss_history = []


    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if not np.all(np.isin(y, [-1, 1])):
            raise ValueError("y должен содержать только -1 и +1")

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0
        self.hinge_loss_history = []
        self.loss_history = []

        batch_size = self.batch_size
        if batch_size is None:
            batch_size = n_samples

        rng = np.random.default_rng(self.seed)

        for _ in range(self.epochs):

            # перемешиваем индексы train перед каждой эпохой
            indices = rng.permutation(n_samples)

            # mini batches
            for start in range(0, n_samples, batch_size):
                batch_indices = indices[start:start + batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                scores = X_batch @ self.weights + self.bias     # s(x) = Xw + b
                margins = y_batch * scores
                inside_margin = margins < 1

                # градиент регуляризации (производная lambda/2 * ||w||^2)
                dw = self.lambda_ * self.weights
                db = 0

                # только объекты, нарушающие margin 
                if np.any(inside_margin):
                    dw -= (X_batch[inside_margin].T @ y_batch[inside_margin] / len(X_batch))
                    db = -np.sum(y_batch[inside_margin]) / len(X_batch)

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            scores = self.decision_function(X)
            hinge_loss = np.mean(np.maximum(0, 1 - y * scores))
            regularization = self.lambda_ / 2 * np.sum(self.weights ** 2)

            self.hinge_loss_history.append(hinge_loss)
            self.loss_history.append(hinge_loss + regularization)

        return self

    def decision_function(self, X):
        """Непрерывный score"""
        X = np.asarray(X, dtype=float)
        return X @ self.weights + self.bias

    def predict(self, X):
        """Применяет 0 threshold к decision_function"""
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, -1)
                
                