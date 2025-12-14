from __future__ import annotations

import numpy as np


def hill_activation(x: np.ndarray, k: float, n: float) -> np.ndarray:
    x = np.clip(x, 0.0, None)
    return (x**n) / (k**n + x**n)


def hill_repression(x: np.ndarray, k: float, n: float) -> np.ndarray:
    return 1.0 - hill_activation(x, k, n)
