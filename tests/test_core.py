"""
Test suite for Savitzky-Golay filter
Fixes Issues 5 & 6: Uses pytest with specific error message validation
"""

import pytest
import numpy as np
from savgol_filter import (
    FilterConfig,
    smooth_nonuniform,
    smooth_nonuniform_simple,
    smooth_with_derivatives,
    estimate_noise_level,
    should_use_uniform_approximation,
)


class TestBasicSmoothing:
    """Test basic smoothing functionality"""
    
    def test_simple_smoothing(self):
        """Basic smoothing should work without errors"""
        x = np.linspace(0, 10, 20)
        y = np.sin(x) + 0.1 * np.random.randn(len(x))
        
        y_smooth = smooth_nonuniform_simple(x, y, window_size=5, polynomial_degree=2)
        
        assert len(y_smooth) == len(y)
        assert not np.any(np.isnan(y_smooth))
    
    def test_polynomial_reproduction(self):
        """Degree-2 filter should reproduce quadratics exactly"""
        x = np.linspace(0, 10, 20)
        y = 2 + 3*x + 0.5*x**2
        
        config = FilterConfig(window_size=7, polynomial_degree=2)
        y_smooth = smooth_nonuniform(x, y, config)
        
        # Interior points should match
        np.testing.assert_allclose(y_smooth[3:-3], y[3:-3], rtol=1e-10)
    
    def test_constant_signal(self):
        """Constant signal should remain constant"""
        x = np.linspace(0, 10, 20)
        y = np.full(len(x), 5.0)
        
        y_smooth = smooth_nonuniform_simple(x, y, window_size=5, polynomial_degree=2)
        
        np.testing.assert_allclose(y_smooth, y, rtol=1e-12)


class TestDerivatives:
    """Test derivative estimation"""
    
    def test_first_derivative(self):
        """First derivative of sin(x) should approximate cos(x)"""
        x = np.linspace(0, 2*np.pi, 50)
        y = np.sin(x)
        
        config = FilterConfig(window_size=11, polynomial_degree=5)
        y_smooth, dy_dx = smooth_with_derivatives(x, y, config, (0, 1))
        
        dy_true = np.cos(x)
        
        # Check interior points
        valid = ~np.isnan(dy_dx)
        np.testing.assert_allclose(dy_dx[valid], dy_true[valid], atol=0.05)
    
    def test_second_derivative(self):
        """Test second derivative computation"""
        x = np.linspace(0, 2*np.pi, 50)
        y = x**2
        
        config = FilterConfig(window_size=9, polynomial_degree=4)
        _, _, d2y = smooth_with_derivatives(x, y, config, (0, 1, 2))
        
        # Second derivative of x^2 is 2
        valid = ~np.isnan(d2y)
        np.testing.assert_allclose(d2y[valid], 2.0, atol=0.1)


class TestAutoDetection:
    """Test auto-detection of uniform spacing"""
    
    def test_uniform_detection(self):
        """Uniform spacing triggers scipy optimization"""
        x_uniform = np.linspace(0, 10, 50)
        x_nonuniform = np.array([0, 0.5, 1.5, 3, 5, 8, 12, 17, 23, 30])
        y = np.sin(x_uniform[:10])
        
        assert should_use_uniform_approximation(x_uniform, y, 7)
        assert not should_use_uniform_approximation(x_nonuniform, y, 7)
    
    def test_nonuniform_detection(self):
        """Nonuniform spacing should use custom algorithm"""
        x = np.array([0, 0.5, 1.5, 3, 5, 7, 10, 14, 19, 25])
        y = np.sin(x)
        
        use_uniform = should_use_uniform_approximation(x, y, window_size=5)
        assert not use_uniform


class TestNoiseEstimation:
    """Test noise level estimation"""
    
    def test_noise_accuracy(self):
        """Noise estimator should be reasonably accurate"""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        
        for true_noise in [0.01, 0.05, 0.1]:
            y = np.sin(x) + true_noise * np.random.randn(len(x))
            estimated = estimate_noise_level(y)
            
            assert abs(estimated - true_noise) / true_noise < 0.5


class TestBoundaryInterpolation:
    """Test boundary interpolation"""
    
    def test_with_interpolation(self):
        """Boundary interpolation should not produce NaN"""
        x = np.linspace(0, 10, 20)
        y = 2 + 3*x + 0.5*x**2
        
        config = FilterConfig(window_size=7, polynomial_degree=2, interpolate_boundaries=True)
        y_smooth = smooth_nonuniform(x, y, config)
        
        assert not np.any(np.isnan(y_smooth))
    
    def test_without_interpolation(self):
        """Without interpolation, boundaries should copy original"""
        x = np.linspace(0, 10, 20)
        y = np.random.randn(len(x))
        
        config = FilterConfig(window_size=7, polynomial_degree=2, interpolate_boundaries=False)
        y_smooth = smooth_nonuniform(x, y, config)
        
        # First 3 and last 3 should match original
        np.testing.assert_array_equal(y_smooth[:3], y[:3])
        np.testing.assert_array_equal(y_smooth[-3:], y[-3:])


class TestErrorHandling:
    """Test error handling with specific message validation (Fix Issue 6)"""
    
    def test_size_mismatch(self):
        """Size mismatch should raise ValueError with specific message"""
        x = np.linspace(0, 10, 20)
        y = np.sin(x)[:-1]
        
        with pytest.raises(ValueError, match="x and y must have same length"):
            smooth_nonuniform_simple(x, y, 5, 2)
    
    def test_even_window(self):
        """Even window size should raise ValueError"""
        x = np.linspace(0, 10, 20)
        y = np.sin(x)
        
        with pytest.raises(ValueError, match="window_size must be odd"):
            smooth_nonuniform_simple(x, y, 6, 2)
    
    def test_insufficient_data(self):
        """Too few data points should raise ValueError"""
        x = np.array([0, 1, 2])
        y = np.array([0, 1, 2])
        
        with pytest.raises(ValueError, match="Data size .* must be >= window_size"):
            smooth_nonuniform_simple(x, y, 5, 2)
    
    def test_non_monotonic(self):
        """Non-increasing x should raise ValueError"""
        x = np.array([0, 2, 1, 3, 4, 5])
        y = np.sin(x)
        
        with pytest.raises(ValueError, match="x values must be strictly increasing"):
            smooth_nonuniform_simple(x, y, 5, 2)
    
    def test_degree_too_high(self):
        """Polynomial degree >= window size should raise ValueError"""
        x = np.linspace(0, 10, 20)
        y = np.sin(x)
        
        with pytest.raises(ValueError, match="polynomial_degree .* must be less than window_size"):
            FilterConfig(window_size=5, polynomial_degree=5)
    
    def test_derivative_order_too_high(self):
        """Derivative order >= polynomial order should raise ValueError"""
        x = np.linspace(0, 10, 20)
        y = np.sin(x)
        config = FilterConfig(window_size=7, polynomial_degree=2)
        
        with pytest.raises(ValueError, match="Cannot compute derivative of order"):
            smooth_with_derivatives(x, y, config, (3,))


class TestEdgeCases:
    """Test edge cases"""
    
    def test_minimum_window(self):
        """Smallest valid window should work"""
        x = np.linspace(0, 10, 10)
        y = np.sin(x)
        
        y_smooth = smooth_nonuniform_simple(x, y, window_size=3, polynomial_degree=1)
        assert len(y_smooth) == len(y)
    
    def test_large_dataset(self):
        """Large dataset should complete in reasonable time"""
        np.random.seed(42)
        x = np.sort(np.random.uniform(0, 100, 1000))
        y = np.sin(x) + 0.1 * np.random.randn(len(x))
        
        y_smooth = smooth_nonuniform_simple(x, y, window_size=11, polynomial_degree=3)
        assert len(y_smooth) == len(y)
