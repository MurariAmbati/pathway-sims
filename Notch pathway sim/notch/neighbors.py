from __future__ import annotations

import numpy as np

from .types import Array, Grid


def _wrap_index(i: int, n: int) -> int:
    return i % n


def _reflect_index(i: int, n: int) -> int:
    if i < 0:
        return -i - 1
    if i >= n:
        return 2 * n - i - 1
    return i


def grid_adjacency(grid: Grid) -> list[list[int]]:
    """return adjacency list for a 2d grid.

    nodes are flattened row-major: idx = r * cols + c.

    topology:
    - von_neumann: 4-neighborhood
    - moore: 8-neighborhood

    boundary:
    - periodic: wrap-around
    - reflect: mirror at edges
    """

    rows, cols = grid.rows, grid.cols
    nbrs: list[list[int]] = [[] for _ in range(grid.n)]

    if grid.topology == "von_neumann":
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif grid.topology == "moore":
        deltas = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]
    else:
        raise ValueError(f"unknown topology: {grid.topology}")

    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            for dr, dc in deltas:
                rr, cc = r + dr, c + dc
                if grid.boundary == "periodic":
                    rr = _wrap_index(rr, rows)
                    cc = _wrap_index(cc, cols)
                elif grid.boundary == "reflect":
                    rr = _reflect_index(rr, rows)
                    cc = _reflect_index(cc, cols)
                else:
                    raise ValueError(f"unknown boundary: {grid.boundary}")

                j = rr * cols + cc
                if j != i:
                    nbrs[i].append(j)

    return nbrs


def neighbor_mean(values: Array, adjacency: list[list[int]]) -> Array:
    """mean of neighbor values for each node."""
    out = np.empty_like(values, dtype=float)
    for i, js in enumerate(adjacency):
        if not js:
            out[i] = values[i]
        else:
            out[i] = float(np.mean(values[js]))
    return out
