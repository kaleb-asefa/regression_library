import numpy as np

# Backwards compatibility: Original simple test data
x = [1, 2, 3, 4, 5, 6, 7, 8]
y = [2.1, 3.9, 6.2, 7.8, 10.3, 11.7, 13.9, 16.2]
x_test = [7, 8]
y_test = [13.9, 16.2]

x_multi_test = np.array([[1, 4], [2, 1], [3, 0], [4, 2], [5, 5]])

y_multi_test = np.array([0, 9, 14, 13, 10])


# 1. Synthetic Single Regression Generator
def make_simple_regression(
    n_samples=100, slope=2.0, intercept=1.0, noise=1.0, random_seed=42
):
    """Generates synthetic data for simple linear regression: y = slope * x + intercept + noise"""
    if random_seed is not None:
        np.random.seed(random_seed)

    x_arr = np.random.uniform(-10.0, 10.0, n_samples)
    noise_arr = np.random.normal(0.0, noise, n_samples)
    y_arr = slope * x_arr + intercept + noise_arr

    return x_arr, y_arr


# 2. Synthetic Multiple Regression Generator
def make_multiple_regression(
    n_samples=100, n_features=3, weights=None, intercept=1.0, noise=1.0, random_seed=42
):
    """Generates synthetic data for multiple linear regression: y = X @ weights + intercept + noise"""
    if random_seed is not None:
        np.random.seed(random_seed)

    if weights is None:
        # Generate default weights
        weights = np.random.uniform(1.0, 5.0, n_features)
    else:
        weights = np.asarray(weights)
        n_features = len(weights)

    X = np.random.uniform(-5.0, 5.0, size=(n_samples, n_features))
    noise_arr = np.random.normal(0.0, noise, n_samples)
    y = X @ weights + intercept + noise_arr

    return X, y


# 3. Real Classic Dataset: Iris Subset (Petal Width -> Petal Length)
def load_iris_regression():
    """Loads a slice of the classic Iris dataset.
    Features (X): Petal Width (cm)
    Target (y): Petal Length (cm)
    This dataset has a strong linear correlation.
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


# 4. Real Synthetic Dataset: Housing Prices (Square Footage, Bedrooms, Age -> Price)
def load_housing_regression():
    """Loads a small classic-style housing dataset.
    Features (X): [Square Footage (hundreds), Bedrooms, Age (years)]
    Target (y): Price (thousands of dollars)
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
