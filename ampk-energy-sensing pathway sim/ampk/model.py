from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


def _hill(x: np.ndarray | float, k: float, n: float) -> np.ndarray | float:
    x = np.asarray(x)
    x = np.clip(x, 0.0, None)
    return (x**n) / (k**n + x**n)


@dataclass(frozen=True)
class ModelParams:
    t_end: float = 12.0
    n_points: int = 2000

    # energy dynamics
    k_prod: float = 0.55
    k_cons: float = 0.65
    k_relax: float = 0.08
    e_set: float = 0.65
    alpha_ampk_supply: float = 0.8
    alpha_mtor_demand: float = 0.6

    # ampk activation by low energy (1-e)
    k_ampk: float = 0.25
    n_ampk: float = 3.0
    k_ampk_relax: float = 1.2

    # mtorc1 activation by nutrients and energy; inhibited by ampk
    k_mtor_nutr: float = 0.35
    n_mtor_nutr: float = 3.0
    k_mtor_energy: float = 0.35
    n_mtor_energy: float = 3.0
    beta_ampk_inhib_mtor: float = 2.5
    k_mtor_relax: float = 0.9

    # ulk1: activated by ampk; inhibited by mtor
    k_ulk_ampk: float = 0.35
    n_ulk_ampk: float = 3.0
    beta_mtor_inhib_ulk: float = 2.8
    k_ulk_relax: float = 0.9

    # autophagy flux downstream of ulk1
    k_auto: float = 0.35
    n_auto: float = 2.0
    k_auto_relax: float = 0.8


@dataclass(frozen=True)
class ModelState:
    e: float = 0.65  # energy charge proxy, 0..1
    ampk: float = 0.15
    mtorc1: float = 0.55
    ulk1: float = 0.20
    autophagy: float = 0.15


@dataclass(frozen=True)
class Inputs:
    nutrient: float = 0.6  # 0..1
    demand: float = 0.5  # 0..1

    # optional events
    stress_pulse_on: bool = False
    stress_pulse_t0: float = 4.0
    stress_pulse_dur: float = 1.0
    stress_pulse_amp: float = 0.5

    nutrient_step_on: bool = False
    nutrient_step_t0: float = 6.0
    nutrient_step_to: float = 0.2


def _make_input_functions(inp: Inputs) -> Tuple[Callable[[float], float], Callable[[float], float]]:
    nutrient0 = float(np.clip(inp.nutrient, 0.0, 1.0))
    demand0 = float(np.clip(inp.demand, 0.0, 1.0))

    def nutrient_t(t: float) -> float:
        if inp.nutrient_step_on and t >= inp.nutrient_step_t0:
            return float(np.clip(inp.nutrient_step_to, 0.0, 1.0))
        return nutrient0

    def demand_t(t: float) -> float:
        d = demand0
        if inp.stress_pulse_on and inp.stress_pulse_t0 <= t <= (inp.stress_pulse_t0 + inp.stress_pulse_dur):
            d = d + inp.stress_pulse_amp
        return float(np.clip(d, 0.0, 2.0))

    return nutrient_t, demand_t


def simulate(
    params: ModelParams,
    inputs: Inputs,
    state0: Optional[ModelState] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    state0 = state0 or ModelState()

    nutrient_t, demand_t = _make_input_functions(inputs)

    y0 = np.array(
        [
            float(state0.e),
            float(state0.ampk),
            float(state0.mtorc1),
            float(state0.ulk1),
            float(state0.autophagy),
        ],
        dtype=float,
    )

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        e, a, m, u, g = y
        n = nutrient_t(t)
        d = demand_t(t)

        e = float(np.clip(e, 0.0, 1.0))
        a = float(np.clip(a, 0.0, 1.0))
        m = float(np.clip(m, 0.0, 1.0))
        u = float(np.clip(u, 0.0, 1.0))
        g = float(np.clip(g, 0.0, 1.0))

        # energy: production increases with nutrients and ampk (catabolic tone),
        # consumption increases with demand and mtorc1 (anabolic drive).
        prod = params.k_prod * n * (1.0 + params.alpha_ampk_supply * a) * (1.0 - e)
        cons = params.k_cons * d * (1.0 + params.alpha_mtor_demand * m) * e
        relax = params.k_relax * (params.e_set - e)
        de = prod - cons + relax

        # ampk: activated by energy stress (1-e)
        energy_stress = 1.0 - e
        a_target = float(_hill(energy_stress, params.k_ampk, params.n_ampk))
        da = params.k_ampk_relax * (a_target - a)

        # mtorc1: activated by nutrient and energy, inhibited by ampk
        m_nutr = float(_hill(n, params.k_mtor_nutr, params.n_mtor_nutr))
        m_energy = float(_hill(e, params.k_mtor_energy, params.n_mtor_energy))
        m_target = (m_nutr * m_energy) / (1.0 + params.beta_ampk_inhib_mtor * a)
        dm = params.k_mtor_relax * (m_target - m)

        # ulk1: activated by ampk, inhibited by mtorc1
        u_target = float(_hill(a, params.k_ulk_ampk, params.n_ulk_ampk)) / (1.0 + params.beta_mtor_inhib_ulk * m)
        du = params.k_ulk_relax * (u_target - u)

        # autophagy flux: downstream of ulk1
        g_target = float(_hill(u, params.k_auto, params.n_auto))
        dg = params.k_auto_relax * (g_target - g)

        return np.array([de, da, dm, du, dg], dtype=float)

    t_eval = np.linspace(0.0, float(params.t_end), int(params.n_points))

    sol = solve_ivp(
        rhs,
        t_span=(0.0, float(params.t_end)),
        y0=y0,
        t_eval=t_eval,
        method="BDF",
        rtol=1e-6,
        atol=1e-9,
    )
    if not sol.success:
        raise RuntimeError(f"ode solver failed: {sol.message}")

    e, a, m, u, g = sol.y
    t = sol.t

    nutrient_series = np.array([nutrient_t(float(tt)) for tt in t], dtype=float)
    demand_series = np.array([demand_t(float(tt)) for tt in t], dtype=float)

    df = pd.DataFrame(
        {
            "t": t,
            "nutrient": nutrient_series,
            "demand": demand_series,
            "energy": np.clip(e, 0.0, 1.0),
            "ampk": np.clip(a, 0.0, 1.0),
            "mtorc1": np.clip(m, 0.0, 1.0),
            "ulk1": np.clip(u, 0.0, 1.0),
            "autophagy": np.clip(g, 0.0, 1.0),
        }
    )

    # summary metrics
    dt = float(params.t_end) / max(int(params.n_points) - 1, 1)
    summary = {
        "mean_energy": float(df["energy"].mean()),
        "mean_ampk": float(df["ampk"].mean()),
        "mean_mtorc1": float(df["mtorc1"].mean()),
        "mean_autophagy": float(df["autophagy"].mean()),
        "autophagy_auc": float(df["autophagy"].sum() * dt),
        "min_energy": float(df["energy"].min()),
    }

    return df, summary
