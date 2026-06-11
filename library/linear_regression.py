"""
From-scratch implementations of Linear Regression and Multiple Linear Regression.

These classes are built using Python and NumPy, demonstrating both gradient descent
and closed-form analytical solutions (Ordinary Least Squares/Normal Equation).
"""

import numpy as np
from typing import Union, List, Tuple, Optional
from .utils import func


class LinearRegression:
    """
    Simple Linear Regression model supporting both Gradient Descent and
    Ordinary Least Squares (OLS) closed-form estimation.

    Parameters
    ----------
    learning_rate : float, default=0.001
        The step size used for updating parameters during gradient descent.
    epochs : int, default=1000
        The number of training iterations for gradient descent.

    Attributes
    ----------
    b_0 : float
        The intercept of the regression line (bias). Default is 0.0.
    b_1 : float
        The slope of the regression line (weight). Default is 0.0.
    loss_history : list of float
        A list storing the Mean Squared Error (MSE) loss value at each epoch.
    coeff_history : list of tuple of (float, float)
        A history of (intercept, slope) tuples recorded at each epoch or step.
    coeff : tuple of (float, float)
        The final trained coefficients (b_0, b_1).
    """

    def __init__(self, learning_rate: float = 0.001, epochs: int = 1000) -> None:
        self.learning_rate: float = learning_rate
        self.epochs: int = epochs
        self.b_0: float = 0.0
        self.b_1: float = 0.0
        self.loss_history: List[float] = []
        self.coeff_history: List[Tuple[float, float]] = []
        self.coeff: Tuple[float, float] = (0.0, 0.0)

    def fit(self, x: Union[List[float], np.ndarray], y: Union[List[float], np.ndarray]) -> None:
        """
        Fit the model coefficients using Batch Gradient Descent.

        Parameters
        ----------
        x : array-like of shape (n_samples,)
            Training input features.
        y : array-like of shape (n_samples,)
            Target values.
        """
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        n = len(x_arr)
        
        self.loss_history = []
        self.coeff_history = []

        for _ in range(self.epochs):
            # Vectorized prediction
            y_pred = self.b_1 * x_arr + self.b_0
            errors = y_pred - y_arr

            # Vectorized gradients
            grad_b0 = (2.0 / n) * np.sum(errors)
            grad_b1 = (2.0 / n) * np.sum(errors * x_arr)

            # Update weights
            self.b_0 -= self.learning_rate * grad_b0
            self.b_1 -= self.learning_rate * grad_b1

            self.loss_history.append(float(np.mean(errors ** 2)))
            self.coeff_history.append((self.b_0, self.b_1))
            
        self.coeff = (self.b_0, self.b_1)

    def fit_ols(self, x: Union[List[float], np.ndarray], y: Union[List[float], np.ndarray]) -> None:
        """
        Fit the model coefficients using Ordinary Least Squares (OLS) closed-form equations.

        Parameters
        ----------
        x : array-like of shape (n_samples,)
            Training input features.
        y : array-like of shape (n_samples,)
            Target values.
        """
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        mean_x, mean_y = np.mean(x_arr), np.mean(y_arr)
        numerator = np.sum((x_arr - mean_x) * (y_arr - mean_y))
        denominator = np.sum((x_arr - mean_x) ** 2)

        if denominator == 0.0:
            self.b_1 = 0.0
        else:
            self.b_1 = float(numerator / denominator)
            
        self.b_0 = float(mean_y - self.b_1 * mean_x)
        self.coeff = (self.b_0, self.b_1)
        self.coeff_history = [(self.b_0, self.b_1)]

    def predict(self, x: Union[float, List[float], np.ndarray]) -> Union[float, List[float]]:
        """
        Predict target values using the linear model: y = b_1 * x + b_0.

        Parameters
        ----------
        x : float or array-like
            Input features to predict.

        Returns
        -------
        float or list of float
            Predicted target values matching the type/shape of the input.
        """
        if isinstance(x, (list, tuple, np.ndarray)):
            return [float(func(xi, b=self.b_0, m=self.b_1)) for xi in x]
        return float(func(x, b=self.b_0, m=self.b_1))

    def score(self, x: Union[List[float], np.ndarray], y: Union[List[float], np.ndarray]) -> float:
        """
        Calculate the Coefficient of Determination (R^2 score) of the prediction.

        Parameters
        ----------
        x : array-like of shape (n_samples,)
            Test input features.
        y : array-like of shape (n_samples,)
            True target values.

        Returns
        -------
        float
            R^2 score. A value of 1.0 indicates perfect fit.
        """
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        y_pred = np.array(self.predict(x_arr), dtype=float)

        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)

        if ss_tot == 0.0:
            return 0.0
        return float(1.0 - (ss_res / ss_tot))


class MultipleLinearRegression:
    """
    Multiple Linear Regression model using the closed-form Normal Equation.

    Attributes
    ----------
    weights : np.ndarray or None
        Coefficients for the input features (excluding the intercept).
    intercept : float or None
        The intercept of the regression hyperplane (bias).
    """

    def __init__(self) -> None:
        self.weights: Optional[np.ndarray] = None
        self.intercept: Optional[float] = None

    def fit(self, X: Union[List[List[float]], np.ndarray], y: Union[List[float], np.ndarray]) -> None:
        """
        Fit the multiple linear model using the Normal Equation:
        theta = (X^T * X)^(-1) * X^T * y

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input features.
        y : array-like of shape (n_samples,)
            Target values.
        """
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        
        # Add bias term (column of 1s) to feature matrix
        X_b = np.c_[np.ones(X_arr.shape[0]), X_arr]
        
        # Normal Equation solver
        theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y_arr
        self.intercept = float(theta[0])
        self.weights = theta[1:]

    def predict(self, X: Union[List[List[float]], np.ndarray]) -> np.ndarray:
        """
        Predict target values for multiple features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features to predict.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted target values.

        Raises
        ------
        ValueError
            If the model has not been fitted prior to prediction.
        """
        X_arr = np.asarray(X, dtype=float)
        if self.weights is None or self.intercept is None:
            raise ValueError("Model must be fitted before calling predict.")
        return X_arr @ self.weights + self.intercept

    def score(self, X: Union[List[List[float]], np.ndarray], y: Union[List[float], np.ndarray]) -> float:
        """
        Calculate the Coefficient of Determination (R^2 score) of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test input features.
        y : array-like of shape (n_samples,)
            True target values.

        Returns
        -------
        float
            R^2 score. A value of 1.0 indicates perfect fit.
        """
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        preds = self.predict(X_arr)
        ss_res = np.sum((y_arr - preds) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        
        if ss_tot == 0.0:
            return 0.0
        return float(1.0 - (ss_res / ss_tot))
