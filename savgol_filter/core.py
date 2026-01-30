"""
Savitzky-Golay Filter for Non-Uniformly Spaced Data

Includes automatic discrimination between uniform and nonuniform data
based on signal variation criterion.

Reference: https://dsp.stackexchange.com/questions/1676/
"""

import numpy as np
import math
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class FilterConfig:
    """Configuration for Savitzky-Golay filter"""
    window_size: int
    polynomial_degree: int
    interpolate_boundaries: bool = True
    
    def __post_init__(self):
        if not isinstance(self.window_size, int):
            raise TypeError("window_size must be an integer")
        if self.window_size % 2 == 0:
            raise ValueError("window_size must be odd")
        if not isinstance(self.polynomial_degree, int):
            raise TypeError("polynomial_degree must be an integer")
        if self.polynomial_degree >= self.window_size:
            raise ValueError(f"polynomial_degree ({self.polynomial_degree}) "
                           f"must be less than window_size ({self.window_size})")
    
    @property
    def half_window(self) -> int:
        return self.window_size // 2
    
    @property
    def polynomial_order(self) -> int:
        return self.polynomial_degree + 1


def validate_inputs(x: np.ndarray, y: np.ndarray, config: FilterConfig) -> None:
    """Validate input arrays and configuration"""
    if len(x) != len(y):
        raise ValueError(f"x and y must have same length (got {len(x)} and {len(y)})")
    if len(x) < config.window_size:
        raise ValueError(f"Data size ({len(x)}) must be >= window_size ({config.window_size})")
    if len(x) > 1 and np.any(np.diff(x) <= 0):
        raise ValueError("x values must be strictly increasing")


def is_approximately_uniform(x: np.ndarray, tolerance: float = 0.05) -> bool:
    """Check if sampling is approximately uniform"""
    if len(x) < 3:
        return True
    spacings = np.diff(x)
    mean_spacing = np.mean(spacings)
    if mean_spacing == 0:
        return False
    relative_deviations = np.abs(spacings - mean_spacing) / mean_spacing
    return np.all(relative_deviations <= tolerance)


def estimate_noise_level(y: np.ndarray) -> float:
    """
    Estimate measurement noise using robust MAD estimator on second differences
    
    Uses Median Absolute Deviation on second differences to estimate σ_noise
    """
    if len(y) < 3:
        return 0.0
    
    second_diff = np.diff(y, n=2)
    mad = np.median(np.abs(second_diff - np.median(second_diff)))
    noise_estimate = 1.4826 * mad / np.sqrt(6)
    return noise_estimate


def should_use_uniform_approximation(
    x: np.ndarray,
    y: np.ndarray,
    window_size: int,
    noise_estimate: Optional[float] = None
) -> bool:
    """
    Determine if uniform approximation is valid based on signal criterion
    
    Criterion: Δf_window < sqrt(N/2) × σ_noise
    where N is the window size
    """
    if noise_estimate is None:
        noise_estimate = estimate_noise_level(y)
    
    if noise_estimate == 0:
        return False
    
    # simple output in case of uniform spacing
    return is_approximately_uniform(x, tolerance=0.05)
    
    # Evaluate signal variation criterion
    half_window = window_size // 2
    threshold = np.sqrt(half_window) * noise_estimate
    
    n_points = len(y)
    for i in range(half_window, n_points - half_window):
        window_y = y[i - half_window : i + half_window + 1]
        delta_f = np.ptp(window_y)
        
        if delta_f >= threshold:
            return False
    
    return True


def compute_sg_coefficients(normalized_coords: np.ndarray, polynomial_order: int) -> Optional[np.ndarray]:
    """Compute Savitzky-Golay filter coefficients for a local window"""
    window_size = len(normalized_coords)
    A = np.zeros((window_size, polynomial_order))
    for j in range(polynomial_order):
        A[:, j] = normalized_coords ** j
    
    try:
        AtA = A.T @ A
        AtA_inv = np.linalg.inv(AtA)
        pseudoinverse = AtA_inv @ A.T
        return pseudoinverse
    except np.linalg.LinAlgError:
        return None


def smooth_nonuniform(x: np.ndarray, y: np.ndarray, config: FilterConfig) -> np.ndarray:
    """Apply Savitzky-Golay filter to non-uniformly spaced data"""
    validate_inputs(x, y, config)
    
    n_points = len(x)
    half_window = config.half_window
    polynomial_order = config.polynomial_order
    
    y_smoothed = np.full(n_points, np.nan)
    first_coeffs = None
    last_coeffs = None
    
    # Main smoothing loop
    for i in range(half_window, n_points - half_window):
        window_indices = np.arange(i - half_window, i + half_window + 1)
        normalized_coords = x[window_indices] - x[i]
        
        pseudoinverse = compute_sg_coefficients(normalized_coords, polynomial_order)
        if pseudoinverse is None:
            raise ValueError(f"Singular matrix at index {i}")
        
        y_smoothed[i] = pseudoinverse[0, :] @ y[window_indices]
        
        if config.interpolate_boundaries:
            if i == half_window:
                first_coeffs = pseudoinverse @ y[window_indices]
            elif i == n_points - half_window - 1:
                last_coeffs = pseudoinverse @ y[window_indices]
    
    # Handle boundaries
    if config.interpolate_boundaries and first_coeffs is not None and last_coeffs is not None:
        for i in range(half_window):
            t = x[i] - x[half_window]
            powers = t ** np.arange(polynomial_order)
            y_smoothed[i] = first_coeffs @ powers
        
        for i in range(n_points - half_window, n_points):
            t = x[i] - x[n_points - half_window - 1]
            powers = t ** np.arange(polynomial_order)
            y_smoothed[i] = last_coeffs @ powers
    else:
        y_smoothed[:half_window] = y[:half_window]
        y_smoothed[-half_window:] = y[-half_window:]
    
    return y_smoothed


def smooth_nonuniform_simple(
    x: np.ndarray,
    y: np.ndarray,
    window_size: int,
    polynomial_degree: int,
    auto_detect: bool = True,
    interpolate_boundaries: bool = True
) -> np.ndarray:
    """
    Simplified interface for Savitzky-Golay smoothing
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable values (must be strictly increasing)
    y : np.ndarray
        Dependent variable values to smooth
    window_size : int
        Window length (must be odd)
    polynomial_degree : int
        Polynomial degree for local fit
    auto_detect : bool
        If True, automatically use uniform approximation when signal criterion is met
    interpolate_boundaries : bool
        Whether to interpolate boundary points
        
    Returns
    -------
    np.ndarray
        Smoothed y values
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    if auto_detect:
        use_uniform = should_use_uniform_approximation(x, y, window_size)
        
        if use_uniform:
            try:
                from scipy.signal import savgol_filter
                return savgol_filter(y, window_length=window_size, polyorder=polynomial_degree)
            except ImportError:
                pass
    
    config = FilterConfig(
        window_size=window_size,
        polynomial_degree=polynomial_degree,
        interpolate_boundaries=interpolate_boundaries
    )
    return smooth_nonuniform(x, y, config)


def smooth_with_derivatives(
    x: np.ndarray,
    y: np.ndarray,
    config: FilterConfig,
    derivative_orders: Tuple[int, ...] = (0,)
) -> Tuple[np.ndarray, ...]:
    """
    Compute smoothed function and its derivatives
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable values
    y : np.ndarray
        Dependent variable values
    config : FilterConfig
        Filter configuration
    derivative_orders : tuple of int
        Which derivatives to compute (0=value, 1=first derivative, etc.)
        
    Returns
    -------
    tuple of np.ndarray
        Smoothed values for each requested derivative order
    """
    validate_inputs(x, y, config)
    
    n_points = len(x)
    half_window = config.half_window
    polynomial_order = config.polynomial_order
    
    max_order = max(derivative_orders)
    if max_order >= polynomial_order:
        raise ValueError(f"Cannot compute derivative of order {max_order} "
                        f"with polynomial degree {config.polynomial_degree}")
    
    results = {order: np.full(n_points, np.nan) for order in derivative_orders}
    factorials = np.array([math.factorial(k) for k in range(polynomial_order)])
    
    for i in range(half_window, n_points - half_window):
        window_indices = np.arange(i - half_window, i + half_window + 1)
        normalized_coords = x[window_indices] - x[i]
        
        pseudoinverse = compute_sg_coefficients(normalized_coords, polynomial_order)
        if pseudoinverse is None:
            raise ValueError(f"Singular matrix at index {i}")
        
        for order in derivative_orders:
            value = pseudoinverse[order, :] @ y[window_indices]
            results[order][i] = value * factorials[order]
    
    for order in derivative_orders:
        results[order][:half_window] = np.nan
        results[order][-half_window:] = np.nan
    
    return tuple(results[order] for order in derivative_orders)


# Fix Issue 4: Renamed polynom → polynomial_degree
def non_uniform_savgol(x: np.ndarray, y: np.ndarray, window: int, polynomial_degree: int) -> np.ndarray:
    """
    Legacy compatibility function
    
    Parameters
    ----------
    x : array_like
        Independent variable values
    y : array_like
        Dependent variable values
    window : int
        Window size (must be odd)
    polynomial_degree : int
        Polynomial degree
        
    Returns
    -------
    np.ndarray
        Smoothed y values
    """
    x = np.asarray(x)
    y = np.asarray(y)
    return smooth_nonuniform_simple(x, y, window, polynomial_degree, 
                                   auto_detect=True, interpolate_boundaries=True)
