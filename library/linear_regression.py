import numpy as np

from .utils import func


class LinearRegression:
    def __init__(self, learning_rate=0.001, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.b_0 = 0.0
        self.b_1 = 0.0
        self.loss_history = []
        self.sum_squared_errors = 0.0

    def fit(self, x, y):
        x = np.array(x)
        y = np.array(y)
        n = len(x)
        for _ in range(self.epochs):
            # Vectorized prediction
            y_pred = self.b_1 * x + self.b_0
            errors = y_pred - y

            # Vectorized gradients
            grad_b0 = (2 / n) * np.sum(errors)
            grad_b1 = (2 / n) * np.sum(errors * x)

            # Update weights
            self.b_0 -= self.learning_rate * grad_b0
            self.b_1 -= self.learning_rate * grad_b1

            self.loss_history.append(np.mean(errors ** 2))
        self.coeff = (self.b_0, self.b_1)

    def fit_ols(self, x, y):
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)

        mean_x, mean_y = np.mean(x_arr), np.mean(y_arr)
        numerator = np.sum((x_arr - mean_x) * (y_arr - mean_y))
        denominator = np.sum((x_arr - mean_x) ** 2)

        self.b_1 = numerator / denominator
        self.b_0 = mean_y - self.b_1 * mean_x
        self.coeff = (self.b_0, self.b_1)

    def score(self, x, y):
        predictions = [self.predict(xi) for xi in x]

        ss_res = sum((yi - y_hat) ** 2 for yi, y_hat in zip(y, predictions))

        y_mean = sum(y) / len(y)
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)

        return 1 - (ss_res / ss_tot)

    def predict(self, x):
        if isinstance(x, (list, tuple, np.ndarray)):
            return [float(func(xi, b=self.b_0, m=self.b_1)) for xi in x]
        return float(func(x, b=self.b_0, m=self.b_1))
