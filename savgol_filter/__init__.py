"""Savitzky-Golay Filter for Non-Uniformly Spaced Data"""

__version__ = "1.0.0"
__author__ = "Julio Montecinos"

from .core import (
    FilterConfig,
    smooth_nonuniform,
    smooth_nonuniform_simple,
    smooth_with_derivatives,
    estimate_noise_level,
    should_use_uniform_approximation,
    is_approximately_uniform,
    non_uniform_savgol,
)

__all__ = [
    "FilterConfig",
    "smooth_nonuniform",
    "smooth_nonuniform_simple",
    "smooth_with_derivatives",
    "estimate_noise_level",
    "should_use_uniform_approximation",
    "is_approximately_uniform",
    "non_uniform_savgol",
]
