from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .math import hill_activation
from .neighbors import grid_adjacency, neighbor_mean
from .types import Array, Grid


@dataclass
class BooleanParams:
    """simple logical notch-delta model.

    state per cell i:
    - d_i in {0,1}
    - icd_i in {0,1}

    update (synchronous):
    - icd <- 1 if neighbor delta mean above threshold
    - d <- 1 if icd == 0 and (optional) intrinsic bias passes

    this is intentionally coarse but useful for quick patterning intuition.
    """

    trans_threshold: float = 0.35
    bias: float = 0.0
    noise: float = 0.0


def simulate_boolean(
    *,
    grid: Grid,
    adjacency: list[list[int]] | None = None,
    steps: int = 50,
    params: BooleanParams | None = None,
    seed: int = 0,
    d0: Array | None = None,
) -> dict[str, Array]:
    p = params or BooleanParams()
    rng = np.random.default_rng(seed)

    if adjacency is None:
        adjacency = grid_adjacency(grid)
    n_cells = len(adjacency)
    if n_cells != grid.n:
        raise ValueError(f"adjacency size {n_cells} does not match grid cells {grid.n}")

    if d0 is None:
        d = (rng.random(n_cells) < 0.5).astype(int)
    else:
        d = np.asarray(d0, dtype=int).reshape(-1)
        if d.shape[0] != n_cells:
            raise ValueError(f"d0 must have length {n_cells}, got {d.shape[0]}")
        d = (d > 0).astype(int)

    icd = np.zeros(n_cells, dtype=int)

    d_hist = np.zeros((steps + 1, n_cells), dtype=int)
    i_hist = np.zeros((steps + 1, n_cells), dtype=int)
    d_hist[0] = d
    i_hist[0] = icd

    for t in range(1, steps + 1):
        d_bar = neighbor_mean(d.astype(float), adjacency)
        trans = hill_activation(d_bar, k=p.trans_threshold, n=8.0)

        icd_next = (trans > 0.5).astype(int)

        eta = 0.0
        if p.noise > 0:
            eta = p.noise * rng.standard_normal(n_cells)

        d_drive = (p.bias + eta) > 0.0
        d_next = ((icd_next == 0) & d_drive).astype(int)

        d, icd = d_next, icd_next
        d_hist[t] = d
        i_hist[t] = icd

    return {"d": d_hist, "icd": i_hist}
