from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from .ode import DEFAULT_STATE, STATE_NAMES, rhs
from .outputs import add_derived_outputs
from .params import Inputs, ModelParams
from .variants import Variant, apply_variant


def simulate(
    *,
    variant: Variant = Variant.baseline,
    params: Optional[ModelParams] = None,
    inputs: Inputs = Inputs(),
    t_end: float = 120.0,
    n_points: int = 600,
    y0: Optional[np.ndarray] = None,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    include_derived: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Simulate the ODE and return (timeseries_df, metadata).

    DataFrame columns include `t` and all state names.
    """

    if params is None:
        params = ModelParams()
    params = apply_variant(params, variant)

    if y0 is None:
        y0 = DEFAULT_STATE.as_vector()

    t_end = float(t_end)
    if t_end <= 0:
        raise ValueError("t_end must be > 0")

    n_points = int(n_points)
    if n_points < 2:
        raise ValueError("n_points must be >= 2")

    t_eval = np.linspace(0.0, t_end, n_points)

    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y, params, inputs),
        t_span=(0.0, t_end),
        y0=np.asarray(y0, dtype=float),
        t_eval=t_eval,
        method="LSODA",
        rtol=float(rtol),
        atol=float(atol),
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    data = {"t": sol.t}
    for i, name in enumerate(STATE_NAMES):
        data[name] = sol.y[i, :]

    df = pd.DataFrame(data)
    if include_derived:
        df = add_derived_outputs(df)

    meta: Dict[str, object] = {
        "variant": str(variant.value),
        "params": asdict(params),
        "inputs": asdict(inputs),
        "t_end": t_end,
        "n_points": n_points,
        "solver": "LSODA",
        "rtol": float(rtol),
        "atol": float(atol),
        "include_derived": bool(include_derived),
    }
    return df, meta
