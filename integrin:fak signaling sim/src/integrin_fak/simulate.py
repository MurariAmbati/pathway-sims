from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from .model import Params, STATE_NAMES, fluxes, rhs


def simulate(
    params: Params,
    t_end: float = 30.0,
    n_points: int = 600,
    y0: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """run the ode and return (states_df, flux_df)."""
    if y0 is None:
        # i, t, f, s, p, e, r
        y0 = np.array([0.05, 0.02, 0.02, 0.01, 0.01, 0.01, 0.05], dtype=float)

    t_eval = np.linspace(0.0, float(t_end), int(n_points))

    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y, params),
        t_span=(0.0, float(t_end)),
        y0=y0,
        t_eval=t_eval,
        method="LSODA",
        rtol=1e-6,
        atol=1e-9,
    )
    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")

    states = pd.DataFrame(sol.y.T, columns=list(STATE_NAMES))
    states.insert(0, "time", sol.t)

    flux_rows = []
    for row in sol.y.T:
        fx = fluxes(row, params)
        flux_rows.append(fx)
    flux = pd.DataFrame(flux_rows)
    flux.insert(0, "time", sol.t)

    # attach params as metadata-like columns for sweeps if needed
    for k, v in asdict(params).items():
        states[k] = v
        flux[k] = v

    return states, flux


def param_sweep_heatmap(
    base: Params,
    ecm_values: np.ndarray,
    force_values: np.ndarray,
    t_end: float = 30.0,
    n_points: int = 400,
    metric: str = "f",
) -> pd.DataFrame:
    """grid sweep returning a tidy dataframe with final-state metric."""
    out = []
    for ecm in ecm_values:
        for force in force_values:
            p = Params(**{**asdict(base), "ecm": float(ecm), "force": float(force)})
            states, _ = simulate(p, t_end=t_end, n_points=n_points)
            val = float(states[metric].iloc[-1])
            out.append({"ecm": float(ecm), "force": float(force), "value": val})
    return pd.DataFrame(out)


def phase_plane(states: pd.DataFrame, x: str = "i", y: str = "f") -> pd.DataFrame:
    """return a small df for plotting phase-plane trajectories."""
    return states[["time", x, y]].copy()
