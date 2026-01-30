"""Basic Savitzky-Golay smoothing example"""
import numpy as np
import matplotlib.pyplot as plt
from savgol_filter import smooth_nonuniform_simple

# Generate noisy data
np.random.seed(42)
x = np.array([0.0, 0.5, 1.2, 2.0, 3.1, 4.0, 5.2, 6.0, 7.1, 8.0, 9.0])
y_true = np.sin(x)
y_noisy = y_true + 0.1 * np.random.randn(len(x))

# Apply smoothing
y_smooth = smooth_nonuniform_simple(x, y_noisy, window_size=5, polynomial_degree=2)

# Plot (Fix Issue 3: visual instead of print)
plt.figure(figsize=(10, 6))
plt.plot(x, y_true, 'k-', linewidth=2, label='True signal')
plt.plot(x, y_noisy, 'o', alpha=0.5, label='Noisy data')
plt.plot(x, y_smooth, 'r-', linewidth=2, label='Smoothed')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('Savitzky-Golay Smoothing')
plt.savefig('basic_smoothing.png', dpi=150)
print("Plot saved as basic_smoothing.png")
