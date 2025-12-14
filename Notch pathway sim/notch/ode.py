from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from .math import hill_activation, hill_repression
from .neighbors import grid_adjacency, neighbor_mean
from .types import Array, Grid


@dataclass
class ODEParams:
    """notch-delta lateral inhibition (minimal but expressive).

    state per cell i:
    - n_i: notch receptor level
    - d_i: delta ligand level
    - i_i: intracellular notch (nicd) activity

    coupling:
    - delta from neighbors (mean) activates notch signaling
    - nicd represses delta production

    the structure is chosen to reproduce classic salt-and-pepper patterning.
    """

    # production baselines
    pn: float = 1.0
    pd: float = 1.0

    # degradation
    gn: float = 1.0
    gd: float = 1.0
    gi: float = 1.0

    # trans-activation strength for nicd generation
    alpha: float = 5.0

    # notch production feedback from nicd (optional)
    beta: float = 0.0

    # hill parameters
    k_trans: float = 0.5
    n_trans: float = 2.0

    k_rep: float = 0.5
    n_rep: float = 2.0

    k_fb: float = 0.5
    n_fb: float = 2.0

    # tiny epsilon to avoid exact symmetry
    eps: float = 1e-3


def _rhs_factory(*, n_cells: int, adjacency: list[list[int]], p: ODEParams) -> Callable[[float, Array], Array]:

    def rhs(_t: float, y: Array) -> Array:
        y = y.reshape(3, n_cells)
        n, d, icd = y[0], y[1], y[2]

        d_bar = neighbor_mean(d, adjacency)

        trans = hill_activation(d_bar, p.k_trans, p.n_trans)
        rep = hill_repression(icd, p.k_rep, p.n_rep)
        fb = hill_activation(icd, p.k_fb, p.n_fb)

        dn = p.pn + p.beta * fb - p.gn * n
        dd = p.pd * rep - p.gd * d
        dicd = p.alpha * n * trans - p.gi * icd

        return np.concatenate([dn, dd, dicd], axis=0)

    return rhs


def simulate_ode(
    *,
    grid: Grid,
    adjacency: list[list[int]] | None = None,
    t_span: tuple[float, float] = (0.0, 40.0),
    dt: float = 0.2,
    params: ODEParams | None = None,
    seed: int = 0,
    y0: Array | None = None,
    method: str = "RK45",
) -> dict[str, Array]:
    """simulate the ode model on a grid.

    returns a dict with keys:
    - t: (T,)
    - n, d, icd: (T, cells)
    """

    p = params or ODEParams()
    rng = np.random.default_rng(seed)

    if adjacency is None:
        adjacency = grid_adjacency(grid)
    n_cells = len(adjacency)

    if n_cells != grid.n:
        raise ValueError(f"adjacency size {n_cells} does not match grid cells {grid.n}")

    if y0 is None:
        base = np.ones((3, n_cells), dtype=float)
        noise = p.eps * rng.standard_normal(size=(3, n_cells))
        y0 = (base + noise).reshape(-1)
    else:
        y0 = np.asarray(y0, dtype=float).reshape(-1)
        if y0.shape[0] != 3 * n_cells:
            raise ValueError(f"y0 must have length {3*n_cells}, got {y0.shape[0]}")

    rhs = _rhs_factory(n_cells=n_cells, adjacency=adjacency, p=p)
    t_eval = np.arange(t_span[0], t_span[1] + 1e-12, dt)

    sol = solve_ivp(rhs, t_span=t_span, y0=y0, t_eval=t_eval, method=method)
    if not sol.success:
        raise RuntimeError(f"ode solve failed: {sol.message}")

    y = sol.y.T.reshape(-1, 3, n_cells)
    return {
        "t": sol.t,
        "n": y[:, 0, :],
        "d": y[:, 1, :],
        "icd": y[:, 2, :],
    }
