"""
Small regression library package.

Expose the main `LinearRegression` class at package level.
"""

from .data import x, x_test, y, y_test
from .linear_regression import LinearRegression

__all__ = ["LinearRegression", "x", "y"]
