import math

from library import LinearRegression, x, y


def test_fit_changes_coefficients():
    model = LinearRegression(learning_rate=0.01, epochs=100)
    initial_b0, initial_b1 = model.b_0, model.b_1
    model.fit(x, y)
    assert not (
        math.isclose(model.b_0, initial_b0) and math.isclose(model.b_1, initial_b1)
    )
