"""
Utility functions and datasets for training and testing regression models.

This module provides original toy datasets, synthetic data generators for simple
and multiple linear regression, and subsets of classic datasets (Iris and Housing).
"""

import numpy as np
from typing import Tuple, List, Optional, Union

# Toy dataset for simple linear regression demonstration (8 data points)
x: List[float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
y: List[float] = [2.1, 3.9, 6.2, 7.8, 10.3, 11.7, 13.9, 16.2]

# Test dataset for simple linear regression validation
x_test: List[float] = [7.0, 8.0]
y_test: List[float] = [13.9, 16.2]

# Sample multiple features data (5 samples, 2 features)
x_multi_test: np.ndarray = np.array([[1.0, 4.0], [2.0, 1.0], [3.0, 0.0], [4.0, 2.0], [5.0, 5.0]])
y_multi_test: np.ndarray = np.array([0.0, 9.0, 14.0, 13.0, 10.0])


def make_simple_regression(
    n_samples: int = 100,
    slope: float = 2.0,
    intercept: float = 1.0,
    noise: float = 1.0,
    random_seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for simple linear regression.

    Generates data according to: y = slope * x + intercept + gaussian_noise.

    Parameters
    ----------
    n_samples : int, default=100
        Number of data samples to generate.
    slope : float, default=2.0
        The true coefficient (slope) of the data.
    intercept : float, default=1.0
        The true intercept (y-intercept) of the data.
    noise : float, default=1.0
        Standard deviation of the gaussian noise added to the target.
    random_seed : int or None, default=42
        Seed for the random number generator to ensure reproducibility.

    Returns
    -------
    x_arr : np.ndarray of shape (n_samples,)
        Generated input features.
    y_arr : np.ndarray of shape (n_samples,)
        Generated target values.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    x_arr = np.random.uniform(-10.0, 10.0, n_samples)
    noise_arr = np.random.normal(0.0, noise, n_samples)
    y_arr = slope * x_arr + intercept + noise_arr

    return x_arr, y_arr


def make_multiple_regression(
    n_samples: int = 100,
    n_features: int = 3,
    weights: Optional[Union[List[float], np.ndarray]] = None,
    intercept: float = 1.0,
    noise: float = 1.0,
    random_seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for multiple linear regression.

    Generates data according to: y = X @ weights + intercept + gaussian_noise.

    Parameters
    ----------
    n_samples : int, default=100
        Number of data samples to generate.
    n_features : int, default=3
        Number of features to generate. Ignored if `weights` is specified.
    weights : array-like of shape (n_features,) or None, default=None
        True coefficients for each feature. If None, weights are generated randomly.
    intercept : float, default=1.0
        The true intercept of the data.
    noise : float, default=1.0
        Standard deviation of the gaussian noise added to the target.
    random_seed : int or None, default=42
        Seed for the random number generator.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
        Generated input feature matrix.
    y : np.ndarray of shape (n_samples,)
        Generated target values.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if weights is None:
        actual_weights = np.random.uniform(1.0, 5.0, n_features)
    else:
        actual_weights = np.asarray(weights, dtype=float)
        n_features = len(actual_weights)

    X = np.random.uniform(-5.0, 5.0, size=(n_samples, n_features))
    noise_arr = np.random.normal(0.0, noise, n_samples)
    y = X @ actual_weights + intercept + noise_arr

    return X, y


def load_iris_regression() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a small slice of the classic Iris dataset for simple linear regression.

    Features (X) : Petal Width (cm)
    Target (y)   : Petal Length (cm)

    Returns
    -------
    X : np.ndarray of shape (40,)
        Petal Width features.
    y : np.ndarray of shape (40,)
        Petal Length target values.
    """
    # 40 real samples from Setosa, Versicolor, and Virginica classes
    iris_raw = [
        [0.2, 1.4],
        [0.2, 1.4],
        [0.2, 1.3],
        [0.2, 1.5],
        [0.2, 1.4],
        [0.4, 1.7],
        [0.3, 1.4],
        [0.2, 1.5],
        [0.2, 1.4],
        [0.1, 1.5],
        [0.2, 1.5],
        [0.2, 1.6],
        [0.1, 1.4],
        [0.1, 1.1],
        [0.2, 1.2],
        [0.4, 1.5],
        [0.4, 1.3],
        [0.3, 1.4],
        [0.3, 1.7],
        [0.3, 1.5],
        [1.4, 4.7],
        [1.5, 4.5],
        [1.5, 4.9],
        [1.3, 4.0],
        [1.5, 4.6],
        [1.3, 4.5],
        [1.6, 4.7],
        [1.0, 3.3],
        [1.3, 4.6],
        [1.4, 3.9],
        [2.5, 6.0],
        [1.9, 5.1],
        [2.1, 5.9],
        [1.8, 5.6],
        [2.2, 5.8],
        [2.1, 6.6],
        [1.7, 4.5],
        [1.8, 4.8],
        [1.8, 5.4],
        [2.5, 5.7],
    ]
    data = np.array(iris_raw)
    X = data[:, 0]  # Petal Width
    y = data[:, 1]  # Petal Length
    return X, y


def load_housing_regression() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a toy housing price dataset for multiple linear regression.

    Features (X) : Matrix of shape (20, 3) representing:
                   - Square Footage (in hundreds of sq ft)
                   - Number of Bedrooms
                   - Age of the property (in years)
    Target (y)   : Array of shape (20,) representing house price (in thousands of dollars)

    Returns
    -------
    X : np.ndarray of shape (20, 3)
        Feature matrix.
    y : np.ndarray of shape (20,)
        Target values (House Prices).
    """
    housing_raw = [
        [15.0, 3, 10, 350.0],
        [20.0, 4, 5, 480.0],
        [12.0, 2, 15, 270.0],
        [18.0, 3, 8, 410.0],
        [25.0, 4, 2, 590.0],
        [14.0, 3, 20, 290.0],
        [22.0, 4, 12, 450.0],
        [16.0, 3, 14, 330.0],
        [30.0, 5, 1, 720.0],
        [10.0, 2, 30, 180.0],
        [19.0, 3, 6, 435.0],
        [21.0, 4, 4, 495.0],
        [13.0, 2, 18, 285.0],
        [17.0, 3, 11, 365.0],
        [24.0, 4, 3, 560.0],
        [11.0, 2, 25, 210.0],
        [28.0, 5, 2, 680.0],
        [15.5, 3, 9, 360.0],
        [20.5, 4, 7, 470.0],
        [23.0, 4, 5, 520.0],
    ]
    data = np.array(housing_raw)
    X = data[:, :3]  # [Square Footage, Bedrooms, Age]
    y = data[:, 3]  # Price
    return X, y
