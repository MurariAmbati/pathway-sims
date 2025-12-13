from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .params import Inputs, ModelParams


STATE_NAMES: Tuple[str, ...] = (
    "R",  # receptor active
    "Is",  # IRS1 inhibited fraction (functional I = 1-Is)
    "P",  # PI3K active
    "PIP3",  # PIP3 fraction (PIP2 = 1-PIP3)
    "Ap",  # active AKT
    "F",  # active FOXO (AKT-inhibited)
    "Ti",  # TSC inhibited fraction (active TSC = 1-Ti)
    "M1",  # mTORC1 active
    "S",  # S6K active
    "M2",  # mTORC2 active
    "X",  # ERK active (optional)
    "K",  # AMPK active (optional)
    "Au",  # autophagy activity
)


@dataclass(frozen=True)
class State:
    R: float = 0.0
    Is: float = 0.0
    P: float = 0.0
    PIP3: float = 0.0
    Ap: float = 0.0
    F: float = 1.0
    Ti: float = 0.0
    M1: float = 0.0
    S: float = 0.0
    M2: float = 0.0
    X: float = 0.0
    K: float = 0.0
    Au: float = 0.0

    def as_vector(self) -> np.ndarray:
        return np.array(
            [
                self.R,
                self.Is,
                self.P,
                self.PIP3,
                self.Ap,
                self.F,
                self.Ti,
                self.M1,
                self.S,
                self.M2,
                self.X,
                self.K,
                self.Au,
            ],
            dtype=float,
        )


DEFAULT_STATE = State()


def clamp01(x: np.ndarray) -> np.ndarray:
    return np.minimum(1.0, np.maximum(0.0, x))


def rhs(t: float, y: np.ndarray, params: ModelParams, inputs: Inputs) -> np.ndarray:
    """Right-hand side for the ODE system.

    Notes:
    - States are treated as normalized activities in [0, 1].
    - The solver may transiently step outside [0, 1]; downstream rates use clamped states.
    """

    y = clamp01(y)
    (
        R,
        Is,
        P,
        PIP3,
        Ap,
        F,
        Ti,
        M1,
        S,
        M2,
        X,
        K,
        Au,
    ) = y

    ligand = float(np.clip(inputs.ligand, 0.0, 1.0))
    erk_input = float(np.clip(inputs.erk_input, 0.0, 1.0))
    ampk_input = float(np.clip(inputs.ampk_input, 0.0, 1.0))

    pi3k_activity = float(np.clip(inputs.pi3k_activity, 0.0, 1.0))
    akt_activity = float(np.clip(inputs.akt_activity, 0.0, 1.0))
    mtorc1_activity = float(np.clip(inputs.mtorc1_activity, 0.0, 1.0))

    I = 1.0 - Is

    # Receptor
    dR = params.k_R_on * ligand * (1.0 - R) - params.k_R_off * R

    # ERK and AMPK (optional inputs)
    dX = params.k_ERK_on * erk_input * (1.0 - X) - params.k_ERK_off * X
    dK = params.k_AMPK_on * ampk_input * (1.0 - K) - params.k_AMPK_off * K

    # IRS1 inhibition: feedback/crosstalk terms can be toggled via params
    inhib_drive = params.k_S6K_to_IRS * S + params.k_ERK_to_IRS * X
    rescue_drive = params.k_FOXO_to_IRS_rescue * F
    dIs = inhib_drive * (1.0 - Is) - (params.k_IRS_deinhib + rescue_drive) * Is

    # PI3K activation by R and functional IRS1
    dP = (
        pi3k_activity
        * (params.k_PI3K_on * R * I * (1.0 - P))
        - params.k_PI3K_off * P
    )

    # PIP3: PI3K production vs PTEN-like removal; PIP2 = 1 - PIP3
    PIP2 = 1.0 - PIP3
    dPIP3 = pi3k_activity * (params.v_PI3K * P * PIP2) - params.v_PTEN * PIP3

    # mTORC2 (kept simple: receptor-proximal)
    dM2 = (
        params.k_mTORC2_on * R * (1.0 - M2)
        - (params.k_mTORC2_off + params.k_S6K_to_mTORC2_inhib * S) * M2
    )

    # AKT phosphorylation from PIP3 (PDK1-like) and mTORC2
    akt_on = (params.v_PDK1 * PIP3 + params.v_mTORC2_to_AKT * M2) * (1.0 - Ap)
    akt_off = params.v_AKT_dephos * Ap
    dAp = akt_activity * akt_on - akt_off

    # FOXO: activated in absence of AKT; inhibited by AKT
    foxo_on = params.k_FOXO_on * (1.0 - Ap) * (1.0 - F)
    foxo_off = (params.k_FOXO_off + params.k_AKT_to_FOXO_inhib * Ap) * F
    dF = foxo_on - foxo_off

    # TSC inhibition by AKT and ERK (ERK term is crosstalk toggle)
    tsc_inhib_drive = params.k_AKT_to_TSC * Ap + params.k_ERK_to_TSC * X
    dTi = tsc_inhib_drive * (1.0 - Ti) - params.k_TSC_recover * Ti

    # mTORC1 activation gated by active TSC (1-Ti), inhibited by AMPK via alpha_AMPK
    ampk_gate = max(0.0, 1.0 - params.alpha_AMPK * K)
    m1_on = params.k_mTORC1_on * (1.0 - Ti) * ampk_gate * (1.0 - M1)
    m1_off = params.k_mTORC1_off * M1
    dM1 = mtorc1_activity * m1_on - m1_off

    # S6K from mTORC1
    dS = params.k_S6K_on * M1 * (1.0 - S) - params.k_S6K_off * S

    # Autophagy: promoted by AMPK, suppressed by mTORC1
    aut_drive = max(0.0, params.beta_AUT_AMPK * K - params.beta_AUT_mTORC1 * M1)
    dAu = params.k_AUT_on * aut_drive * (1.0 - Au) - params.k_AUT_off * Au

    return np.array(
        [dR, dIs, dP, dPIP3, dAp, dF, dTi, dM1, dS, dM2, dX, dK, dAu], dtype=float
    )


def state_index(name: str) -> int:
    return STATE_NAMES.index(name)


def initial_state_from_dict(values: dict) -> np.ndarray:
    y0 = DEFAULT_STATE.as_vector().copy()
    for key, value in values.items():
        if key not in STATE_NAMES:
            raise KeyError(f"Unknown state '{key}'. Valid: {list(STATE_NAMES)}")
        y0[state_index(key)] = float(value)
    return y0
