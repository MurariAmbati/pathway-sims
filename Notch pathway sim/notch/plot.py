from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .types import Grid


def _reshape_to_grid(grid: Grid, x: np.ndarray) -> np.ndarray:
    return x.reshape(grid.rows, grid.cols)


def plot_field(grid: Grid, field: np.ndarray, *, title: str, ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 4))

    im = ax.imshow(_reshape_to_grid(grid, field), cmap="viridis", interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_ode_snapshot(grid: Grid, out: dict[str, np.ndarray], *, idx: int = -1):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plot_field(grid, out["n"][idx], title="notch (n)", ax=axes[0])
    plot_field(grid, out["d"][idx], title="delta (d)", ax=axes[1])
    plot_field(grid, out["icd"][idx], title="nicd (icd)", ax=axes[2])
    fig.tight_layout()
    return fig
