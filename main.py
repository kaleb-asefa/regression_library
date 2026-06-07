from regression_library import x, y, LinearRegression


def main() -> None:
    model = LinearRegression(learning_rate=0.01, epochs=2000)
    model.fit(x, y)
    print(f"Intercept (b_0): {model.b_0:.4f}, Slope (b_1): {model.b_1:.4f}")


if __name__ == "__main__":
    main()
