import math

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import r2_score

from data_blog import (
    LinearRegression,
    MultipleLinearRegression,
    x,
    x_multi_test,
    x_test,
    y,
    y_multi_test,
    y_test,
    make_simple_regression,
    make_multiple_regression,
    load_iris_regression,
    load_housing_regression,
)


@pytest.fixture
def model():
    model = LinearRegression(learning_rate=0.01, epochs=1000)
    model_ols = LinearRegression()
    model_ols.fit_ols(x, y)
    model.fit(x, y)
    yield model, model_ols
    del model

def test_fit_changes_coefficients():
    model = LinearRegression(learning_rate=0.01, epochs=10)
    initial_b0, initial_b1 = model.b_0, model.b_1
    model.fit(x, y)
    assert not (
        math.isclose(model.b_0, initial_b0) and math.isclose(model.b_1, initial_b1)
    )

def test_score_calculates_r_squared(model):
    model, _ = model
    r_squared = model.score(x, y)
    assert isinstance(r_squared, float)
    assert 0 <= r_squared <= 1

def test_predict_returns_expected_values(model):
    model, _ = model
    predictions = model.predict(x)
    assert len(predictions) == len(x)
    assert all(isinstance(pred, (int, float)) for pred in predictions)

def test_coefficients_close_to_expected(model):
    model, _ = model
    expected_b0 = 0.0
    expected_b1 = 2.0
    assert math.isclose(model.b_0, expected_b0, abs_tol=0.1)
    assert math.isclose(model.b_1, expected_b1, abs_tol=0.1)
    assert model.coeff == (model.b_0, model.b_1)

def test_r_square_to_sklearn(model):
    model, _ = model
    sklearn_model = SklearnLinearRegression()
    sklearn_model.fit(np.array(x).reshape(-1, 1), y)
    sklearn_r2 = sklearn_model.score(np.array(x).reshape(-1, 1), y)
    model_r2 = model.score(x, y)
    assert math.isclose(model_r2, sklearn_r2, abs_tol=0.01)

def test_predict_on_test_data(model):
    model, _ = model
    predictions = model.predict(x_test)
    assert len(predictions) == len(x_test)
    assert all(isinstance(pred, (int, float)) for pred in predictions)
    r2 = r2_score(y_test, predictions)
    assert r2 > 0.9

def test_ols_coefficients_close_to_expected(model):
    model, model_ols = model
    expected_b0 = 0.0
    expected_b1 = 2.0
    assert math.isclose(model_ols.b_0, expected_b0, abs_tol=0.1)
    assert math.isclose(model_ols.b_1, expected_b1, abs_tol=0.1)
    assert model_ols.coeff == (model_ols.b_0, model_ols.b_1)

def test_ols_r_square_to_sklearn(model):
    model, model_ols = model
    sklearn_model = SklearnLinearRegression()
    sklearn_model.fit(np.array(x).reshape(-1, 1), y)
    sklearn_r2 = sklearn_model.score(np.array(x).reshape(-1, 1), y)
    model_ols_r2 = model_ols.score(x, y)
    assert math.isclose(model_ols_r2, sklearn_r2, abs_tol=0.01)

def test_ols_predict_on_test_data(model):
    model, model_ols = model
    predictions = model_ols.predict(x_test)
    assert len(predictions) == len(x_test)
    assert all(isinstance(pred, (int, float)) for pred in predictions)
    r2 = r2_score(y_test, predictions)
    assert r2 > 0.9


def test_multiple_regression_vs_sklearn():
    model = MultipleLinearRegression()
    model.fit(x_multi_test, y_multi_test)

    sklearn_model = SklearnLinearRegression()
    sklearn_model.fit(x_multi_test, y_multi_test)

    assert math.isclose(model.intercept, sklearn_model.intercept_, abs_tol=1e-5)
    assert np.allclose(model.weights, sklearn_model.coef_, atol=1e-5)

    preds = model.predict(x_multi_test)
    sklearn_preds = sklearn_model.predict(x_multi_test)
    assert np.allclose(preds, sklearn_preds, atol=1e-5)

    score = model.score(x_multi_test, y_multi_test)
    sklearn_score = sklearn_model.score(x_multi_test, y_multi_test)
    assert math.isclose(score, sklearn_score, abs_tol=1e-5)


def test_plot_regression_line():
    from unittest.mock import patch
    from data_blog.animate import plot_regression_line

    model = LinearRegression(learning_rate=0.01, epochs=10)
    model.fit(x, y)

    with patch("matplotlib.pyplot.show") as mock_show:
        plot_regression_line(x, y, model)
        mock_show.assert_called_once()


def test_animate_regression_fitting():
    from unittest.mock import patch
    from data_blog.animate import animate_regression_fitting

    model = LinearRegression(learning_rate=0.01, epochs=10)
    model.fit(x, y)

    with patch("matplotlib.pyplot.show") as mock_show:
        anim = animate_regression_fitting(x, y, model)
        assert anim is not None
        mock_show.assert_called_once()


def test_make_simple_regression():
    X, y = make_simple_regression(n_samples=50, slope=3.0, intercept=-2.0, noise=0.5, random_seed=123)
    assert len(X) == 50
    assert len(y) == 50
    
    # Fit OLS model on the generated data
    model = LinearRegression()
    model.fit_ols(X, y)
    
    # Should be close to slope 3.0 and intercept -2.0
    assert math.isclose(model.b_1, 3.0, abs_tol=0.2)
    assert math.isclose(model.b_0, -2.0, abs_tol=0.2)


def test_make_multiple_regression():
    X, y = make_multiple_regression(n_samples=100, n_features=4, weights=[1.5, -2.0, 0.5, 3.0], intercept=10.0, noise=0.1, random_seed=42)
    assert X.shape == (100, 4)
    assert len(y) == 100
    
    model = MultipleLinearRegression()
    model.fit(X, y)
    
    assert math.isclose(model.intercept, 10.0, abs_tol=0.1)
    assert np.allclose(model.weights, [1.5, -2.0, 0.5, 3.0], atol=0.1)


def test_load_iris_regression():
    X, y = load_iris_regression()
    assert len(X) == 40
    assert len(y) == 40
    
    model = LinearRegression()
    model.fit_ols(X, y)
    
    score = model.score(X, y)
    assert score > 0.90  # Iris petal width vs length has a strong correlation


def test_load_housing_regression():
    X, y = load_housing_regression()
    assert X.shape == (20, 3)
    assert len(y) == 20
    
    model = MultipleLinearRegression()
    model.fit(X, y)
    
    score = model.score(X, y)
    assert score > 0.95  # Our synthetic housing data fits very well


