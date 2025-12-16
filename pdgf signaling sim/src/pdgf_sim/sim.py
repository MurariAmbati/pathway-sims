from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from .model import PDGFParams, STATE_NAMES, default_initial_state, dict_to_params, pdgf_ode


def simulate(
    params: PDGFParams,
    y0: np.ndarray,
    t_end_min: float = 240.0,
    n_points: int = 800,
    method: str = "LSODA",
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> pd.DataFrame:
    t_eval = np.linspace(0.0, float(t_end_min), int(n_points))

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        return pdgf_ode(t, y, params)

    sol = solve_ivp(
        rhs,
        t_span=(0.0, float(t_end_min)),
        y0=np.asarray(y0, dtype=float),
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    data = {"t_min": sol.t}
    for i, name in enumerate(STATE_NAMES):
        data[name] = sol.y[i, :]

    df = pd.DataFrame(data)
    return df


def normalize_for_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["L", "R", "C", "D", "Dp"]:
        if col in out:
            out[col] = out[col].clip(lower=0.0)
    for col in ["ERK", "AKT"]:
        if col in out:
            out[col] = out[col].clip(lower=0.0, upper=1.0)
    if "P" in out:
        out["P"] = out["P"].clip(lower=0.0)
    return out


def param_sweep_2d(
    base_params: PDGFParams,
    y0: np.ndarray,
    t_end_min: float,
    sweep_x: Tuple[str, Sequence[float]],
    sweep_y: Tuple[str, Sequence[float]],
    metric: str = "P_final",
    n_points: int = 600,
) -> pd.DataFrame:
    x_name, x_vals = sweep_x
    y_name, y_vals = sweep_y

    base_dict = asdict(base_params)

    rows: List[Dict[str, float]] = []
    for xv in x_vals:
        for yv in y_vals:
            d = dict(base_dict)
            d[x_name] = float(xv)
            d[y_name] = float(yv)
            params = dict_to_params(d)

            df = simulate(params=params, y0=y0, t_end_min=t_end_min, n_points=n_points)
            df = normalize_for_display(df)

            if metric == "P_final":
                m = float(df["P"].iloc[-1])
            elif metric == "ERK_auc":
                m = float(np.trapz(df["ERK"].to_numpy(), df["t_min"].to_numpy()))
            elif metric == "AKT_auc":
                m = float(np.trapz(df["AKT"].to_numpy(), df["t_min"].to_numpy()))
            elif metric == "Dp_auc":
                m = float(np.trapz(df["Dp"].to_numpy(), df["t_min"].to_numpy()))
            else:
                raise ValueError(f"Unknown metric: {metric}")

            rows.append({x_name: float(xv), y_name: float(yv), metric: m})

    return pd.DataFrame(rows)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")
