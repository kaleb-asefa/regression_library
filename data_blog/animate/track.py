"""
Visualization and animation utilities for monitoring regression fitting.

This module provides functions to plot fitted regression lines and create real-time
animations of the gradient descent optimization process, tracking how loss decreases.
"""

import os
from typing import Union, List, Optional
import matplotlib

# Headless environment detection: use Agg backend if no DISPLAY is present
if "DISPLAY" not in os.environ and not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from data_blog import LinearRegression, x, y


def plot_regression_line(
    x: Union[List[float], np.ndarray],
    y: Union[List[float], np.ndarray],
    model: LinearRegression
) -> None:
    """
    Plot a static 2D scatter plot of the data along with the fitted regression line.

    Parameters
    ----------
    x : array-like of shape (n_samples,)
        Input feature values.
    y : array-like of shape (n_samples,)
        True target values.
    model : LinearRegression
        A fitted simple linear regression model containing `b_0` and `b_1`.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    plt.figure()
    plt.scatter(x_arr, y_arr, color="blue", label="Data Points")

    # Generate points along x for the line
    x_line = np.linspace(min(x_arr), max(x_arr), 100)
    y_line = model.b_1 * x_line + model.b_0

    plt.plot(x_line, y_line, color="red", label="Regression Line")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Regression Fit")
    plt.legend()
    plt.show()


def animate_regression_fitting(
    x: Union[List[float], np.ndarray],
    y: Union[List[float], np.ndarray],
    model: LinearRegression,
    save_path: Optional[str] = None,
    fps: int = 10,
    interval: Optional[int] = None,
    max_frames: int = 100,
    pause_seconds: float = 2.0,
    show_plot: bool = True,
) -> FuncAnimation:
    """
    Create an animation showing the regression line and loss history over epochs.

    Requires that the provided `model` has been fitted and contains `coeff_history`
    and `loss_history`.

    Parameters
    ----------
    x : array-like of shape (n_samples,)
        Input feature values.
    y : array-like of shape (n_samples,)
        True target values.
    model : LinearRegression
        A fitted simple linear regression model with training history.
    save_path : str or None, default=None
        Optional file path (e.g., 'fitting.gif') to save the animation using Pillow.
    fps : int, default=10
        Frames per second for saved GIF or display timing (ignored if `interval` is provided).
    interval : int or None, default=None
        Delay between frames in milliseconds. If None, calculated as `1000 / fps`.
    max_frames : int, default=100
        Maximum number of animated frames sampled across fitting history.
    pause_seconds : float, default=2.0
        Duration in seconds to linger on the final fitted frame before repeating.
    show_plot : bool, default=True
        Whether to call `plt.show()` if `save_path` is None.

    Returns
    -------
    FuncAnimation
        The matplotlib FuncAnimation object.
    """
    if not hasattr(model, "coeff_history") or not model.coeff_history:
        raise ValueError(
            "Model has not been fitted or does not have coefficient history."
        )

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if interval is None:
        interval = int(1000 / max(1, fps))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle("Gradient Descent Fitting Process", fontsize=14, fontweight="bold")

    # Left subplot: Data Points and Regression Line
    ax1.scatter(x_arr, y_arr, color="#1f77b4", edgecolors="k", alpha=0.8, s=40, label="Data Points")
    (line,) = ax1.plot([], [], color="#d62728", lw=2.5, label="Fitting Line")
    x_margin = (max(x_arr) - min(x_arr)) * 0.1 if len(x_arr) > 1 else 1.0
    y_margin = (max(y_arr) - min(y_arr)) * 0.1 if len(y_arr) > 1 else 1.0
    ax1.set_xlim(min(x_arr) - x_margin, max(x_arr) + x_margin)
    ax1.set_ylim(min(y_arr) - y_margin, max(y_arr) + y_margin)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Model Fit Step-by-Step")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left")

    # Overlay text box for metrics
    metrics_text = ax1.text(
        0.03,
        0.65,
        "",
        transform=ax1.transAxes,
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.9),
    )

    # Right subplot: Loss Curve
    (loss_line,) = ax2.plot([], [], color="#9467bd", lw=2, label="MSE Loss")
    (loss_dot,) = ax2.plot([], [], "ro", markersize=6, label="Current Epoch")
    ax2.set_xlim(0, len(model.loss_history))
    max_loss = max(model.loss_history) if model.loss_history else 1.0
    ax2.set_ylim(0, max_loss * 1.1)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss (MSE)")
    ax2.set_title("Loss Convergence Curve")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right")

    # Sample history indices for comfortable viewing pace
    history_len = len(model.coeff_history)
    if history_len > max_frames:
        sampled_indices = np.linspace(0, history_len - 1, max_frames, dtype=int).tolist()
    else:
        sampled_indices = list(range(history_len))

    # Add pause/lagging on the final fitted frame
    pause_frames = int(pause_seconds * (1000 / interval)) if pause_seconds > 0 else 0
    frame_indices = sampled_indices + [sampled_indices[-1]] * pause_frames

    # X values for smooth line visualization across plot range
    x_line_range = np.linspace(min(x_arr) - x_margin, max(x_arr) + x_margin, 100)

    def init():
        line.set_data([], [])
        loss_line.set_data([], [])
        loss_dot.set_data([], [])
        metrics_text.set_text("")
        return line, loss_line, loss_dot, metrics_text

    def update(frame_idx):
        idx = frame_indices[frame_idx]
        b_0, b_1 = model.coeff_history[idx]
        current_loss = model.loss_history[idx]

        y_line_pred = b_1 * x_line_range + b_0
        line.set_data(x_line_range, y_line_pred)

        loss_line.set_data(range(idx + 1), model.loss_history[: idx + 1])
        loss_dot.set_data([idx], [current_loss])

        text_str = (
            f"Epoch: {idx + 1}/{history_len}\n"
            f"Loss (MSE): {current_loss:.4f}\n"
            f"b_0 (bias) : {b_0:.4f}\n"
            f"b_1 (slope): {b_1:.4f}"
        )
        metrics_text.set_text(text_str)

        return line, loss_line, loss_dot, metrics_text

    plt.tight_layout()

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        init_func=init,
        interval=interval,
        repeat_delay=int(pause_seconds * 1000) if pause_seconds > 0 else None,
        blit=True,
    )

    if save_path:
        anim.save(save_path, writer="pillow", fps=1000 / interval)
        plt.close(fig)
    elif show_plot:
        plt.show()

    return anim


if __name__ == "__main__":
    # Test execution when run as a standalone script
    print("Fitting model...")
    model = LinearRegression(learning_rate=0.01, epochs=100)
    model.fit(x, y)
    print("Generating static plot...")
    plot_regression_line(x, y, model)
    print("Generating interactive animation...")
    animate_regression_fitting(x, y, model, show_plot=False)

