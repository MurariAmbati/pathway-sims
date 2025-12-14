from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from .model import STATE_ORDER, egfr_ode, pack_state, unpack_state
from .params import Params, default_initial_state, default_params


def simulate(
    *,
    params: Params | None = None,
    initial_state: dict[str, float] | None = None,
    t_end: float = 60.0,
    dt: float = 0.2,
    method: Literal["RK45", "BDF", "LSODA"] = "LSODA",
) -> pd.DataFrame:
    p = params or default_params()
    x0 = initial_state or default_initial_state(p)

    y0 = pack_state(x0)
    t_eval = np.arange(0.0, t_end + dt, dt, dtype=float)

    sol = solve_ivp(
        fun=lambda t, y: egfr_ode(t, y, p),
        t_span=(0.0, float(t_end)),
        y0=y0,
        t_eval=t_eval,
        method=method,
        rtol=1e-6,
        atol=1e-9,
    )

    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")

    out = pd.DataFrame(sol.y.T, columns=STATE_ORDER)
    out.insert(0, "t", sol.t)
    out["ligand"] = float(p.ligand)

    # derived metrics
    out["prolif_index"] = np.clip(out["ERK_p"] ** 1.2, 0.0, 1.0)
    out["survival_index"] = np.clip(out["AKT_p"] ** 1.2, 0.0, 1.0)
    return out


def with_ligand(params: Params, ligand: float) -> Params:
    return replace(params, ligand=float(ligand))


def summary_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Convenience summary for a simulated time course."""
    metrics = {}
    for col in ["ERK_p", "AKT_p", "RAS_GTP", "PI3K_act", "prolif_index", "survival_index"]:
        metrics[f"{col}_peak"] = float(df[col].max())
        metrics[f"{col}_steady"] = float(df[col].iloc[-1])
    metrics["t_end"] = float(df["t"].iloc[-1])
    metrics["ligand"] = float(df["ligand"].iloc[0])
    return metrics
