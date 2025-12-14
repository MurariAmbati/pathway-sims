from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .params import Params


STATE_ORDER: list[str] = [
    "R",
    "RL",
    "R2",
    "RAS_GTP",
    "RAF_act",
    "MEK_p",
    "ERK_p",
    "PI3K_act",
    "AKT_p",
]


def pack_state(state: dict[str, float]) -> np.ndarray:
    return np.array([float(state[k]) for k in STATE_ORDER], dtype=float)


def unpack_state(y: np.ndarray) -> dict[str, float]:
    return {k: float(v) for k, v in zip(STATE_ORDER, y, strict=True)}


def _clamp01(x: float, eps: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    if abs(x) < eps:
        return 0.0
    return x


def egfr_ode(t: float, y: np.ndarray, params: Params) -> np.ndarray:
    """ODE system.

    receptor species are concentrations, downstream species are fractions in [0,1].
    ligand is treated as a constant input (params.ligand).
    """

    # unpack
    R, RL, R2, ras, raf, mek, erk, pi3k, akt = y

    # enforce physical bounds softly (avoid solver blow-ups)
    R = max(R, 0.0)
    RL = max(RL, 0.0)
    R2 = max(R2, 0.0)

    ras = _clamp01(float(ras), params.clamp_eps)
    raf = _clamp01(float(raf), params.clamp_eps)
    mek = _clamp01(float(mek), params.clamp_eps)
    erk = _clamp01(float(erk), params.clamp_eps)
    pi3k = _clamp01(float(pi3k), params.clamp_eps)
    akt = _clamp01(float(akt), params.clamp_eps)

    L = max(params.ligand, 0.0)

    # receptor module
    bind = params.k_on * L * R
    unbind = params.k_off * RL

    dim = params.k_dim * (RL**2)
    undim = params.k_undim * R2

    dR = -bind + unbind
    dRL = bind - unbind - 2.0 * dim + 2.0 * undim
    dR2 = dim - undim - params.k_r2_deact * R2

    # downstream: use R2 as input (scaled)
    r2_signal = R2 / (params.r_tot + 1e-9)

    # negative feedback: erk boosts effective gap rate
    eff_gap = params.k_ras_gap * (1.0 + params.feedback_strength * erk)

    ras_act_drive = params.k_ras_act * r2_signal
    # michaelis-menten-like saturation on inactive fraction
    ras_inactive = 1.0 - ras
    dRAS = ras_act_drive * (ras_inactive / (params.ras_km + ras_inactive)) - eff_gap * ras

    dRAF = params.k_raf_act * ras * (1.0 - raf) - params.k_raf_deact * raf
    dMEK = params.k_mek_phos * raf * (1.0 - mek) - params.k_mek_dephos * mek
    dERK = params.k_erk_phos * mek * (1.0 - erk) - params.k_erk_dephos * erk

    pi3k_drive = r2_signal + params.pi3k_ras_crosstalk * ras
    dPI3K = params.k_pi3k_act * pi3k_drive * (1.0 - pi3k) - params.k_pi3k_deact * pi3k

    dAKT = params.k_akt_phos * pi3k * (1.0 - akt) - params.k_akt_dephos * akt

    return np.array([dR, dRL, dR2, dRAS, dRAF, dMEK, dERK, dPI3K, dAKT], dtype=float)


def params_to_dict(params: Params) -> dict[str, float]:
    return {k: float(v) for k, v in asdict(params).items()}
