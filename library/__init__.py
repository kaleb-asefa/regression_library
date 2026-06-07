"""
Small regression library package.

Expose the main `LinearRegression` class at package level.
"""

from .data import x, y
from .linear_regression import LinearRegression

__all__ = ["LinearRegression", "x", "y"]
