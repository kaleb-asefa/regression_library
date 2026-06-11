# Regression Library

A lightweight, from-scratch linear and multiple regression library implemented in Python with NumPy. Perfect for educational demonstration, portfolios, and quick testing of regression mechanics.

## Features

- **Simple Linear Regression**: Supports Batch Gradient Descent with complete loss and coefficient history, and closed-form OLS estimation.
- **Multiple Linear Regression**: Analytical Normal Equation solver for multidimensional datasets.
- **Visualization & Animation**: Real-time animation of gradient descent fitting and static regression line plotting with Matplotlib.
- **Dataset Loaders**: Pre-packaged synthetic generators and subsets of classic real datasets (Iris and Housing).

## Quick Start

### Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Interactive Demo

To explore interactive fits on various datasets and watch the live training animation:

```bash
python main.py
```

### Library Usage

You can easily import and train models in your own scripts:

```python
from library import LinearRegression, x, y

# Instantiate and fit using Gradient Descent
model = LinearRegression(learning_rate=0.01, epochs=1000)
model.fit(x, y)

print(f"Intercept: {model.b_0:.4f}")
print(f"Slope: {model.b_1:.4f}")
print(f"R-squared Score: {model.score(x, y):.4f}")
```
