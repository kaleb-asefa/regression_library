"""
Utility functions for linear regression operations.
"""

def func(x: float, m: float = 0.0, b: float = 0.0) -> float:
    """
    Compute the linear function y = m * x + b.

    Parameters
    ----------
    x : float
        The input feature value.
    m : float, default=0.0
        The slope of the line (weight).
    b : float, default=0.0
        The y-intercept (bias).

    Returns
    -------
    float
        The calculated function value.
    """
    return m * x + b
