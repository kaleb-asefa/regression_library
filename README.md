# Regression Library

Small linear regression library suitable for portfolio projects and demos.

Usage:

```bash
python main.py
```

Or import as a package:

```python
from regression_library import LinearRegression, x, y

model = LinearRegression(learning_rate=0.01, epochs=1000)
model.fit(x, y)
print(model.b_0, model.b_1)
```
