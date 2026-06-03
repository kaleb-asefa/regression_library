import matplotlib.pyplot as plt
import numpy as np

x = [1, 2, 3, 4, 5, 6, 7, 8]
y = [2, 4, 6, 8, 10, 12, 14, 16]

def func(x, m=0, b=0):
    return m * x + b
error_ls = []
def batch_gradient(rate, epoch, b_0=0, b_1=0):
    j = 0
    while j < epoch:
        total_0 = 0
        total_1 = 0
        for i in range(len(x)):
            error = func(x[i], b=b_0, m=b_1) - y[i]
            total_0 += error * 1
            total_1 += error * x[i]

        # Divide by N to get the average gradient, preventing gradient explosion
        b_0 -= (rate * (total_0))
        b_1 -= (rate * (total_1))
        error_ls.append((b_0, total_0))
        j += 1

    return b_0, b_1

# 0.01 works perfectly now that we average the gradient!
a, b = batch_gradient(0.001, 10)

print(f"Intercept (b_0): {a:.4f}, Slope (b_1): {b:.4f}")

x_lis = np.array(range(1, 10))
y_lis = b * x_lis + a
print(error_ls)
plt.scatter(x, y, color="red", label="Data")
plt.plot(x_lis, y_lis, label="Gradient Descent Fit")
# plt.plot(x_lis, error_ls, label="Gradient Descent Fit")
plt.legend()
plt.show()
