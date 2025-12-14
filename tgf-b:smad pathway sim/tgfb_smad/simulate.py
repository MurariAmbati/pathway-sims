from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from tgfb_smad.ode import Inputs, STATE_ORDER, enforce_physical_constraints, pack_state, rhs


@dataclass
class SimulationResult:
    t: np.ndarray
    y: np.ndarray
    df: pd.DataFrame


def simulate_timecourse(
    *,
    ligand: float,
    mapk: float,
    pi3k: float,
    params: Dict[str, float],
    y0: Dict[str, float],
    t_end: float,
    n_points: int,
    method: str = "LSODA",
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> SimulationResult:
    t = np.linspace(0.0, float(t_end), int(n_points))
    y0_vec = pack_state(y0)

    u = Inputs(ligand=float(ligand), mapk=float(mapk), pi3k=float(pi3k))

    def f(tt: float, yy: np.ndarray) -> np.ndarray:
        return rhs(tt, yy, params, u)

    sol = solve_ivp(
        f,
        (float(t[0]), float(t[-1])),
        y0_vec,
        t_eval=t,
        method=method,
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    y = np.vstack([enforce_physical_constraints(row, params) for row in sol.y.T])
    df = pd.DataFrame(y, columns=STATE_ORDER)
    df.insert(0, "t", sol.t)

    return SimulationResult(t=sol.t, y=y, df=df)


def dose_response(
    *,
    dose_min: float,
    dose_max: float,
    n_doses: int,
    logspace: bool,
    mapk: float,
    pi3k: float,
    params: Dict[str, float],
    y0: Dict[str, float],
    t_end: float,
    n_points: int,
) -> pd.DataFrame:
    dose_min = float(dose_min)
    dose_max = float(dose_max)
    n_doses = int(n_doses)

    if n_doses < 2:
        raise ValueError("n_doses must be >= 2")

    if logspace:
        # include 0 dose explicitly, then logspace for the positive range
        eps = 1e-4
        pos = np.geomspace(max(eps, dose_min if dose_min > 0 else eps), max(dose_max, eps), n_doses - 1)
        doses = np.concatenate([[0.0], pos])
    else:
        doses = np.linspace(dose_min, dose_max, n_doses)

    rows = []
    for d in doses:
        sim = simulate_timecourse(
            ligand=float(d),
            mapk=mapk,
            pi3k=pi3k,
            params=params,
            y0=y0,
            t_end=t_end,
            n_points=n_points,
        )
        end = sim.df.iloc[-1]
        rows.append(
            {
                "dose": float(d),
                "C_n": float(end["C_n"]),
                "Smad7": float(end["Smad7"]),
                "G_prog": float(end["G_prog"]),
                "E_prog": float(end["E_prog"]),
                "F_prog": float(end["F_prog"]),
            }
        )

    return pd.DataFrame(rows)


def crosstalk_grid(
    *,
    ligand: float,
    mapk_values: np.ndarray,
    pi3k_values: np.ndarray,
    params: Dict[str, float],
    y0: Dict[str, float],
    t_end: float,
    n_points: int,
    readout: str = "E_prog",
) -> pd.DataFrame:
    """Evaluate an end-time readout over a MAPK×PI3K grid.

    Returns a tidy dataframe with columns: mapk, pi3k, value.
    """

    mapk_values = np.asarray(mapk_values, dtype=float)
    pi3k_values = np.asarray(pi3k_values, dtype=float)

    if readout not in {"C_n", "Smad7", "G_prog", "E_prog", "F_prog"}:
        raise ValueError(f"Unsupported readout: {readout}")

    rows = []
    for m in mapk_values:
        for a in pi3k_values:
            sim = simulate_timecourse(
                ligand=float(ligand),
                mapk=float(m),
                pi3k=float(a),
                params=params,
                y0=y0,
                t_end=t_end,
                n_points=n_points,
            )
            end = sim.df.iloc[-1]
            rows.append({"mapk": float(m), "pi3k": float(a), "value": float(end[readout])})

    return pd.DataFrame(rows)
