import math

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import r2_score

from library import LinearRegression, x, x_test, y, y_test


@pytest.fixture
def model():
    model = LinearRegression(learning_rate=0.01, epochs=1000)
    model.fit(x, y)
    yield model
    del model

def test_fit_changes_coefficients():
    model = LinearRegression(learning_rate=0.01, epochs=10)
    initial_b0, initial_b1 = model.b_0, model.b_1
    model.fit(x, y)
    assert not (
        math.isclose(model.b_0, initial_b0) and math.isclose(model.b_1, initial_b1)
    )

def test_score_calculates_r_squared(model):
    r_squared = model.score(y)
    assert 0 <= r_squared <= 1

def test_predict_returns_expected_values(model):
    predictions = model.predict(x)
    assert len(predictions) == len(x)
    assert all(isinstance(pred, (int, float)) for pred in predictions)

def test_coefficients_close_to_expected(model):
    expected_b0 = 0.0
    expected_b1 = 2.0
    assert math.isclose(model.b_0, expected_b0, abs_tol=0.1)
    assert math.isclose(model.b_1, expected_b1, abs_tol=0.1)
    assert model.coeff == (model.b_0, model.b_1)

def test_r_square_to_sklearn(model):
    sklearn_model = SklearnLinearRegression()
    sklearn_model.fit(np.array(x).reshape(-1, 1), y)
    sklearn_r2 = sklearn_model.score(np.array(x).reshape(-1, 1), y)
    model_r2 = model.score(y)
    assert math.isclose(model_r2, sklearn_r2, abs_tol=0.01)

def test_predict_on_test_data(model):
    predictions = model.predict(x_test)
    assert len(predictions) == len(x_test)
    assert all(isinstance(pred, (int, float)) for pred in predictions)
    r2 = r2_score(y_test, predictions)
    assert r2 > 0.9

