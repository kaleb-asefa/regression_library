from library import LinearRegression, x, y, x_test, y_test


def main() -> None:
    model = LinearRegression(learning_rate=0.01, epochs=2000)
    model.fit(x, y)
    print(f"Intercept (b_0): {model.b_0:.4f}, Slope (b_1): {model.b_1:.4f}")
    print(f"predictions: {model.predict(x_test)}")


if __name__ == "__main__":
    main()
