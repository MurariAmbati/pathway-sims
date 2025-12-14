from __future__ import annotations

import numpy as np


def neighbor_anticorrelation(x: np.ndarray, neighbor_mean_x: np.ndarray) -> float:
    """pearson corr(x, mean(neighbors(x))) as a single scalar.

    for lateral inhibition patterns, this tends to be negative.
    """

    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(neighbor_mean_x, dtype=float).reshape(-1)

    x = x - np.mean(x)
    y = y - np.mean(y)

    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom == 0.0:
        return 0.0
    return float(np.sum(x * y) / denom)


def field_summary(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float).reshape(-1)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }
