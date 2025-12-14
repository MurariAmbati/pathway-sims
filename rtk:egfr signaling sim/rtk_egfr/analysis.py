from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from .params import Params, default_initial_state, default_params
from .sim import simulate, summary_metrics


def dose_response(
    *,
    doses: np.ndarray,
    params: Params | None = None,
    t_end: float = 60.0,
    dt: float = 0.2,
    metric: str = "ERK_p_peak",
) -> pd.DataFrame:
    p0 = params or default_params()
    rows: list[dict[str, float]] = []

    for dose in doses:
        p = replace(p0, ligand=float(dose))
        df = simulate(params=p, initial_state=default_initial_state(p), t_end=t_end, dt=dt)
        m = summary_metrics(df)
        rows.append({"dose": float(dose), metric: float(m[metric])})

    out = pd.DataFrame(rows).sort_values("dose").reset_index(drop=True)
    out["log10_dose"] = np.log10(out["dose"] + 1e-12)
    return out


def parameter_sweep(
    *,
    name: str,
    values: np.ndarray,
    params: Params | None = None,
    t_end: float = 60.0,
    dt: float = 0.2,
    metric: str = "ERK_p_peak",
) -> pd.DataFrame:
    p0 = params or default_params()

    if not hasattr(p0, name):
        raise KeyError(f"unknown parameter: {name}")

    rows: list[dict[str, float]] = []
    for v in values:
        p = replace(p0, **{name: float(v)})
        df = simulate(params=p, initial_state=default_initial_state(p), t_end=t_end, dt=dt)
        m = summary_metrics(df)
        rows.append({name: float(v), metric: float(m[metric])})

    return pd.DataFrame(rows)


def local_sensitivity(
    *,
    params: Params | None = None,
    t_end: float = 60.0,
    dt: float = 0.2,
    metric: str = "ERK_p_peak",
    rel_step: float = 0.1,
) -> pd.DataFrame:
    """One-at-a-time local sensitivities using a symmetric relative perturbation.

    Returns a dataframe with columns: parameter, base, up, down, sensitivity.
    sensitivity is a unitless finite-difference approximation of d log(metric) / d log(param).
    """

    p0 = params or default_params()
    p_dict = {k: float(v) for k, v in asdict(p0).items()}

    base_df = simulate(params=p0, initial_state=default_initial_state(p0), t_end=t_end, dt=dt)
    base = summary_metrics(base_df)[metric]
    base = float(base)

    rows: list[dict[str, float | str]] = []

    for k, v in p_dict.items():
        if k in {"clamp_eps"}:
            continue
        if v == 0.0:
            continue

        up_p = replace(p0, **{k: float(v * (1.0 + rel_step))})
        dn_p = replace(p0, **{k: float(v * (1.0 - rel_step))})

        up = float(summary_metrics(simulate(params=up_p, initial_state=default_initial_state(up_p), t_end=t_end, dt=dt))[metric])
        dn = float(summary_metrics(simulate(params=dn_p, initial_state=default_initial_state(dn_p), t_end=t_end, dt=dt))[metric])

        # d log y / d log x
        y_ratio = max(up, 1e-12) / max(dn, 1e-12)
        x_ratio = (v * (1.0 + rel_step)) / (v * (1.0 - rel_step))
        sens = float(np.log(y_ratio) / np.log(x_ratio))

        rows.append(
            {
                "parameter": k,
                "base_param": float(v),
                "base_metric": base,
                "metric_up": up,
                "metric_down": dn,
                "sensitivity": sens,
            }
        )

    out = pd.DataFrame(rows).sort_values("sensitivity", ascending=False).reset_index(drop=True)
    return out
