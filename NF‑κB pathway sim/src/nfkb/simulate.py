from __future__ import annotations

from dataclasses import asdict
from typing import Callable, Literal

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from .model import NFkBParams, nfkb_rhs


def _make_time_grid(t_end: float, dt: float) -> np.ndarray:
    if t_end <= 0:
        raise ValueError("t_end must be > 0")
    if dt <= 0:
        raise ValueError("dt must be > 0")
    n = int(np.floor(t_end / dt))
    if n < 2:
        raise ValueError("t_end/dt too small")
    return np.linspace(0.0, n * dt, n + 1)


def simulate_deterministic(
    p: NFkBParams,
    x0: np.ndarray,
    ikk_fn: Callable[[float], float],
    *,
    t_end: float,
    dt: float,
    method: Literal["RK45", "Radau", "BDF"] = "RK45",
) -> pd.DataFrame:
    t_eval = _make_time_grid(t_end, dt)

    def rhs(t: float, x: np.ndarray) -> np.ndarray:
        return nfkb_rhs(t, x, p, float(ikk_fn(t)))

    sol = solve_ivp(
        rhs,
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        y0=np.asarray(x0, dtype=float),
        t_eval=t_eval,
        method=method,
        rtol=1e-6,
        atol=1e-9,
    )

    if not sol.success:
        raise RuntimeError(f"ode solve failed: {sol.message}")

    df = pd.DataFrame(
        {
            "t": sol.t,
            "nn": sol.y[0],
            "im": sol.y[1],
            "i": sol.y[2],
            "ikk": np.array([float(ikk_fn(float(t))) for t in sol.t], dtype=float),
        }
    )

    for k, v in asdict(p).items():
        df.attrs[k] = v

    return df


def simulate_stochastic(
    p: NFkBParams,
    x0: np.ndarray,
    ikk_fn: Callable[[float], float],
    *,
    t_end: float,
    dt: float,
    sigma: float = 0.02,
    seed: int | None = 0,
) -> pd.DataFrame:
    # euler-maruyama with additive noise (simple, fast, exploratory).
    t = _make_time_grid(t_end, dt)
    x = np.zeros((t.size, 3), dtype=float)
    x[0] = np.asarray(x0, dtype=float)

    rng = np.random.default_rng(seed)

    for k in range(t.size - 1):
        ikk = float(ikk_fn(float(t[k])))
        drift = nfkb_rhs(float(t[k]), x[k], p, ikk)
        noise = sigma * rng.normal(size=3) * np.sqrt(dt)
        x[k + 1] = x[k] + drift * dt + noise
        x[k + 1] = np.maximum(x[k + 1], 0.0)

    df = pd.DataFrame(
        {
            "t": t,
            "nn": x[:, 0],
            "im": x[:, 1],
            "i": x[:, 2],
            "ikk": np.array([float(ikk_fn(float(tt))) for tt in t], dtype=float),
        }
    )

    df.attrs.update({"sigma": sigma, "seed": seed})
    for k, v in asdict(p).items():
        df.attrs[k] = v

    return df
