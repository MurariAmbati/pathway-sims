from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class Params:
    # upstream drive
    gef: float = 1.0
    gap: float = 1.0

    # rhoa kinetics
    k_on_rhoa: float = 1.2
    k_off_rhoa: float = 1.0

    # rhoa -> rock
    k_rock_on: float = 2.0
    k_rock_off: float = 1.5

    # rhoa -> mdia
    k_mdia_on: float = 1.6
    k_mdia_off: float = 1.0

    # rock -> limk
    k_limk_on: float = 1.8
    k_limk_off: float = 1.2

    # limk -> cofilin inhibition
    k_cof_inhib: float = 2.0
    k_cof_recover: float = 0.8

    # actin polymerization / depolymerization
    k_factin_poly: float = 2.2
    k_factin_depoly: float = 1.2

    # rock -> mlc, mlcp -> mlc
    k_mlc_phos: float = 2.0
    k_mlc_dephos: float = 1.5

    # contractility readout weights
    w_factin: float = 0.6
    w_mlc: float = 0.8


def _hill(x: float, k: float = 0.5, n: float = 2.0) -> float:
    x = max(0.0, min(1.0, float(x)))
    return (x**n) / (k**n + x**n)


def _ode(t: float, y: np.ndarray, p: Params, perturb: Dict[str, float]) -> np.ndarray:
    # state variables are normalized 0..1
    rhoa, rock, mdia, limk, cofilin, factin, mlc = y

    # apply simple multiplicative perturbations (0..2) to nodes
    rhoa *= perturb.get("rhoa", 1.0)
    rock *= perturb.get("rock", 1.0)
    mdia *= perturb.get("mdia", 1.0)
    limk *= perturb.get("limk", 1.0)
    cofilin *= perturb.get("cofilin", 1.0)
    factin *= perturb.get("factin", 1.0)
    mlc *= perturb.get("mlc", 1.0)

    # rhoa activation: gef drives on, gap increases off
    rhoa_on = p.k_on_rhoa * p.gef * (1.0 - rhoa)
    rhoa_off = p.k_off_rhoa * p.gap * rhoa
    drhoa = rhoa_on - rhoa_off

    # rock effector
    rock_on = p.k_rock_on * _hill(rhoa) * (1.0 - rock)
    rock_off = p.k_rock_off * rock
    drock = rock_on - rock_off

    # mdia effector
    mdia_on = p.k_mdia_on * _hill(rhoa) * (1.0 - mdia)
    mdia_off = p.k_mdia_off * mdia
    dmdia = mdia_on - mdia_off

    # limk downstream of rock
    limk_on = p.k_limk_on * _hill(rock) * (1.0 - limk)
    limk_off = p.k_limk_off * limk
    dlimk = limk_on - limk_off

    # cofilin is active when high; limk inhibits
    cof_inhib = p.k_cof_inhib * _hill(limk) * cofilin
    cof_recover = p.k_cof_recover * (1.0 - cofilin)
    dcofilin = cof_recover - cof_inhib

    # factin increases with mdia and with reduced severing (high cofilin -> more severing -> less factin)
    poly = p.k_factin_poly * (_hill(mdia) + 0.35 * _hill(rock)) * (1.0 - factin)
    sever = 0.9 * (1.0 - cofilin)  # low cofilin activity means less severing
    depoly = p.k_factin_depoly * (0.35 + 0.65 * sever) * factin
    dfactin = poly - depoly

    # mlc phosphorylation promoted by rock, dephosphorylation baseline+mlcp proxy
    phos = p.k_mlc_phos * _hill(rock) * (1.0 - mlc)
    dephos = p.k_mlc_dephos * (0.35 + 0.65 * (1.0 - _hill(rock))) * mlc
    dmlc = phos - dephos

    return np.array([drhoa, drock, dmdia, dlimk, dcofilin, dfactin, dmlc], dtype=float)


def simulate(
    params: Params,
    t_end: float = 60.0,
    n_points: int = 600,
    y0: Tuple[float, float, float, float, float, float, float] = (0.10, 0.05, 0.05, 0.05, 0.85, 0.20, 0.10),
    perturb: Dict[str, float] | None = None,
) -> Dict[str, np.ndarray]:
    perturb = perturb or {}

    t_eval = np.linspace(0.0, float(t_end), int(n_points))
    y0_arr = np.array(y0, dtype=float)

    sol = solve_ivp(
        fun=lambda t, y: _ode(t, y, params, perturb),
        t_span=(0.0, float(t_end)),
        y0=y0_arr,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-9,
    )

    if not sol.success:
        raise RuntimeError(f"simulation failed: {sol.message}")

    rhoa, rock, mdia, limk, cofilin, factin, mlc = sol.y

    contractility = np.clip(params.w_factin * factin + params.w_mlc * mlc, 0.0, 1.0)

    # a simple non-monotonic migration score: high at moderate tension
    migration = np.clip(factin * (1.0 - 0.8 * contractility), 0.0, 1.0)

    return {
        "t": sol.t,
        "rhoa": rhoa,
        "rock": rock,
        "mdia": mdia,
        "limk": limk,
        "cofilin": cofilin,
        "factin": factin,
        "mlc": mlc,
        "contractility": contractility,
        "migration": migration,
    }
