import numpy as np

from .utils import func


class LinearRegression:
    def __init__(self, learning_rate=0.001, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.b_0 = 0.0
        self.b_1 = 0.0
        self.coefficients = (self.b_0, self.b_1)
        self.loss_history = []
        self.sum_squared_errors = 0.0
        self.r_squared = None

    def fit(self, x, y):
        for _epoch in range(self.epochs):
            total_0 = 0.0
            total_1 = 0.0
            error_sq = 0.0
            for i in range(len(x)):
                error = func(x[i], b=self.b_0, m=self.b_1) - y[i]
                error_sq += error**2
                total_0 += error * 1
                total_1 += error * x[i]

            self.b_0 -= self.learning_rate * (total_0 / len(x))
            self.b_1 -= self.learning_rate * (total_1 / len(x))
            self.loss_history.append(error_sq / len(x))
            self.sum_squared_errors = error_sq

    def score(self, y):
        ss_total = sum((yi - np.mean(y)) ** 2 for yi in y)
        if ss_total == 0:
            self.r_squared = 0.0
        else:
            self.r_squared = 1 - (self.sum_squared_errors / ss_total)
        return self.r_squared

    def predict(self, x):
        return func(x, b=self.b_0, m=self.b_1)
