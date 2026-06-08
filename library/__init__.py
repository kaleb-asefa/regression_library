"""
Small regression library package.

Expose the main regression classes at package level.
"""

from .data import x, x_test, y, y_test, x_multi_test, y_multi_test
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
]
