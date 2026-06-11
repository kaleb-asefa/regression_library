"""
From-Scratch Linear Regression Library.

A educational, lightweight machine learning package implementing simple and multiple
linear regression algorithms from scratch using NumPy.

Features:
- `LinearRegression`: Supports gradient descent fitting and analytical closed-form (OLS) fitting.
- `MultipleLinearRegression`: Normal equation solver for multidimensional features.
- `data_blog.animate`: Visualization utilities to animate gradient descent fitting and plot regression lines.
- Synthetic and Classic dataset generators/loaders for testing and comparative benchmarking.
"""

from .data import (
    x,
    x_test,
    y,
    y_test,
    x_multi_test,
    y_multi_test,
    make_simple_regression,
    make_multiple_regression,
    load_iris_regression,
    load_housing_regression,
)
from .linear_regression import LinearRegression, MultipleLinearRegression

__all__ = [
    "LinearRegression",
    "MultipleLinearRegression",
    "x",
    "y",
    "x_test",
    "y_test",
    "x_multi_test",
    "y_multi_test",
    "make_simple_regression",
    "make_multiple_regression",
    "load_iris_regression",
    "load_housing_regression",
]
