import matplotlib.pyplot as plt
import numpy as np

x = [1, 2, 3, 4, 5, 6, 7, 8]
y = [2.1, 3.9, 6.2, 7.8, 10.3, 11.7, 13.9, 16.2]

def func(x, m=0, b=0):
    return m * x + b


class LinearRegression:
    def __init__(self, learning_rate=0.001, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.b_0 = 0
        self.b_1 = 0
        self.coefficients = (self.b_0, self.b_1)
        self.loss_history = []
        self.sum_squared_errors = 0
        self.r_squared = self.score(y)

    def fit(self, x, y):
        for _epoch in range(self.epochs):
            total_0 = 0
            total_1 = 0
            error_sq = 0
            for i in range(len(x)):
                error = func(x[i], b=self.b_0, m=self.b_1) - y[i]
                error_sq += error ** 2
                total_0 += error * 1
                total_1 += error * x[i]

            self.b_0 -= (self.learning_rate * (total_0 / len(x)))
            self.b_1 -= (self.learning_rate * (total_1 / len(x)))
            self.loss_history.append(error_sq / len(x))
            self.sum_squared_errors = error_sq

    def score(self, y):
        ss_total = sum((yi - np.mean(y)) ** 2 for yi in y)
        self.r_squared = 1 - (self.sum_squared_errors / ss_total)
        return self.r_squared

    def predict(self, x):
        return func(x, b=self.b_0, m=self.b_1)

model = LinearRegression(learning_rate=0.01, epochs=2000)
model.fit(x, y)
print(f"Intercept (b_0): {model.b_0:.4f}, Slope (b_1): {model.b_1:.4f}")
