# Savitzky-Golay Filter

Production-quality Savitzky-Golay filter for non-uniformly spaced data with auto-detection.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from savgol_filter import smooth_nonuniform_simple
import numpy as np

x = np.array([0.0, 0.5, 1.2, 2.0, 3.1, 4.0])
y = np.sin(x) + 0.1 * np.random.randn(len(x))

y_smooth = smooth_nonuniform_simple(x, y, window_size=5, polynomial_degree=2)
```

## Features

- **Non-uniform spacing**: Handles irregularly sampled data
- **Auto-detection**: Automatically uses optimized scipy for uniform data
- **Derivatives**: Compute smoothed derivatives up to arbitrary order
- **Boundary handling**: Polynomial extrapolation at boundaries
- **Type-safe**: Full type hints and input validation

## Examples

See `examples/` directory:
- `example_basic.py` - Basic smoothing
- `example_derivatives.py` - Derivative estimation

## Testing

```bash
pytest tests/ -v --cov=savgol_filter
```

## License

MIT License - Copyright (c) 2026 Julio Montecinos
