import math

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import r2_score

from library import (
    LinearRegression,
    MultipleLinearRegression,
    x,
    x_multi_test,
    x_test,
    y,
    y_multi_test,
    y_test,
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
    from animate import plot_regression_line

    model = LinearRegression(learning_rate=0.01, epochs=10)
    model.fit(x, y)

    with patch("matplotlib.pyplot.show") as mock_show:
        plot_regression_line(x, y, model)
        mock_show.assert_called_once()


def test_animate_regression_fitting():
    from unittest.mock import patch
    from animate import animate_regression_fitting

    model = LinearRegression(learning_rate=0.01, epochs=10)
    model.fit(x, y)

    with patch("matplotlib.pyplot.show") as mock_show:
        anim = animate_regression_fitting(x, y, model)
        assert anim is not None
        mock_show.assert_called_once()

