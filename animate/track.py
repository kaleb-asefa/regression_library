import os
import matplotlib

# Headless detection: use Agg backend if no DISPLAY is present
if "DISPLAY" not in os.environ and not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from library import LinearRegression, x, y


def plot_regression_line(x, y, model):
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
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


def animate_regression_fitting(x, y, model, save_path=None):
    if not hasattr(model, "coeff_history") or not model.coeff_history:
        raise ValueError(
            "Model has not been fitted or does not have coefficient history."
        )

    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left subplot: Data Points and Regression Line
    ax1.scatter(x_arr, y_arr, color="blue", label="Data Points")
    (line,) = ax1.plot([], [], color="red", lw=2, label="Fitting Line")
    ax1.set_xlim(min(x_arr) - 1, max(x_arr) + 1)
    ax1.set_ylim(min(y_arr) - 2, max(y_arr) + 2)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Linear Regression Fit")
    ax1.legend()

    # Right subplot: Loss Curve
    (loss_line,) = ax2.plot([], [], color="purple", lw=2, label="MSE Loss")
    ax2.set_xlim(0, len(model.loss_history))
    ax2.set_ylim(0, max(model.loss_history) * 1.1 if model.loss_history else 1.0)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss (MSE)")
    ax2.set_title("Loss Curve History")
    ax2.legend()

    def init():
        line.set_data([], [])
        loss_line.set_data([], [])
        return line, loss_line

    # Downsample history to keep the animation fast and smooth (max 100 frames)
    history_len = len(model.coeff_history)
    max_frames = 100
    if history_len > max_frames:
        indices = np.linspace(0, history_len - 1, max_frames, dtype=int)
    else:
        indices = list(range(history_len))

    def update(frame_idx):
        idx = indices[frame_idx]
        b_0, b_1 = model.coeff_history[idx]
        y_pred = b_1 * x_arr + b_0
        line.set_data(x_arr, y_pred)
        loss_line.set_data(range(idx + 1), model.loss_history[: idx + 1])
        return line, loss_line

    anim = FuncAnimation(
        fig,
        update,
        frames=len(indices),
        init_func=init,
        interval=50,
        blit=True,
    )

    if save_path:
        anim.save(save_path, writer="pillow")
    else:
        plt.tight_layout()
        plt.show()

    return anim


if __name__ == "__main__":
    # Test execution when run as a standalone script
    print("Fitting model...")
    model = LinearRegression(learning_rate=0.01, epochs=1000)
    model.fit(x, y)
    print("Generating static plot...")
    plot_regression_line(x, y, model)
    print("Generating interactive animation...")
    animate_regression_fitting(x, y, model)
