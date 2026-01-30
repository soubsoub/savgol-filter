"""Derivatives estimation example"""
import numpy as np
import matplotlib.pyplot as plt
from savgol_filter import FilterConfig, smooth_with_derivatives

# Generate data
x = np.linspace(0, 2*np.pi, 50)
y = np.sin(x) + 0.1 * np.random.randn(len(x))

# Compute smoothed values and first derivative
config = FilterConfig(window_size=11, polynomial_degree=5)
y_smooth, dy_dx = smooth_with_derivatives(x, y, config, (0, 1))

# True derivatives
dy_true = np.cos(x)

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(x, y, 'o', alpha=0.5, label='Noisy data')
ax1.plot(x, y_smooth, 'r-', linewidth=2, label='Smoothed')
ax1.plot(x, np.sin(x), 'k--', linewidth=1, label='True')
ax1.set_ylabel('y')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_title('Function Smoothing')

ax2.plot(x, dy_dx, 'r-', linewidth=2, label='Estimated dy/dx')
ax2.plot(x, dy_true, 'k--', linewidth=1, label='True dy/dx')
ax2.set_xlabel('x')
ax2.set_ylabel('dy/dx')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_title('First Derivative')

plt.tight_layout()
plt.savefig('derivatives.png', dpi=150)
print("Plot saved as derivatives.png")
