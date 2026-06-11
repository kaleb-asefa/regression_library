import sys

from animate.track import animate_regression_fitting
from library import (
    LinearRegression,
    MultipleLinearRegression,
    load_housing_regression,
    load_iris_regression,
    make_multiple_regression,
    make_simple_regression,
    x,
    y,
)


def main() -> None:
    print("=" * 50)
    print("Welcome to the From-Scratch Linear Regression Library!")
    print("=" * 50)
    print("Select a dataset to train your model:")
    print("1. Original Simple Dataset (8 points)")
    print("2. Synthetic Simple Regression (Dynamic generator with custom noise)")
    print("3. Classic Iris Subset (Petal Width -> Petal Length)")
    print("4. Classic Housing Dataset (Square Footage, Bedrooms, Age -> Price)")
    print("=" * 50)

    try:
        choice = input("Enter your choice (1-4): ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting...")
        return

    if choice == "1":
        print("\n--- Training on Original Simple Dataset ---")
        model = LinearRegression(learning_rate=0.01, epochs=2000)
        model.fit(x, y)
        print(f"Intercept (b_0): {model.b_0:.4f}, Slope (b_1): {model.b_1:.4f}")
        print(f"R-squared Score: {model.score(x, y):.4f}")

    elif choice == "2":
        print("\n--- Training on Synthetic Simple Regression ---")
        try:
            noise_val = input("Enter noise level (e.g. 0.5, 1.5, 3.0) [default: 1.0]: ").strip()
            noise = float(noise_val) if noise_val else 1.0
        except ValueError:
            noise = 1.0

        X, y_data = make_simple_regression(
            n_samples=100, slope=3.0, intercept=-2.0, noise=noise
        )

        print("\nFitting Gradient Descent Model...")
        model_gd = LinearRegression(learning_rate=0.01, epochs=10)
        model_gd.fit(X, y_data)
        print(f"[GD] Intercept: {model_gd.b_0:.4f}, Slope: {model_gd.b_1:.4f}")
        print(f"[GD] R-squared: {model_gd.score(X, y_data):.4f}")

        print("\nFitting OLS (Closed-Form) Model...")
        model_ols = LinearRegression()
        model_ols.fit_ols(X, y_data)
        print(f"[OLS] Intercept: {model_ols.b_0:.4f}, Slope: {model_ols.b_1:.4f}")
        print(f"[OLS] R-squared: {model_ols.score(X, y_data):.4f}")

        print("\nGenerating animation of Gradient Descent fitting process...")
        animate_regression_fitting(X, y_data, model_gd)

    elif choice == "3":
        print("\n--- Training on Classic Iris Subset ---")
        X, y_data = load_iris_regression()
        model = LinearRegression()
        model.fit_ols(X, y_data)
        print("Features: Petal Width -> Target: Petal Length")
        print(f"Intercept: {model.b_0:.4f}, Slope (Coefficient): {model.b_1:.4f}")
        print(f"R-squared Score: {model.score(X, y_data):.4f}")

    elif choice == "4":
        print("\n--- Training on Classic Housing Dataset ---")
        X, y_data = load_housing_regression()
        model = MultipleLinearRegression()
        model.fit(X, y_data)
        print("Features: [Square Footage, Bedrooms, Age] -> Target: House Price")
        print(f"Intercept (Bias): {model.intercept:.4f}")
        print(f"Weights (Coefficients): {model.weights}")
        print(f"R-squared Score: {model.score(X, y_data):.4f}")

    else:
        print("Invalid choice! Exiting...")


if __name__ == "__main__":
    main()
