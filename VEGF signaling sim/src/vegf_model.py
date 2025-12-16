from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class VegfParams:
    """parameter set for a compact vegf-a/vegfr2 signaling + outputs model.

    this is a phenomenological ode model intended for interactive exploration.
    variables are normalized (dimensionless) unless noted.
    """

    # ligand & receptor interaction
    kon: float = 3.0
    koff: float = 1.0
    kact: float = 2.0
    kdeact: float = 0.6

    # vegfr1 acts primarily as a decoy (binds vegf, reduces availability to vegfr2)
    kon_r1: float = 4.0
    koff_r1: float = 0.6

    # receptor trafficking
    kint: float = 0.35
    krec: float = 0.10

    # nrp1 co-receptor: increases effective vegfr2 activation when present
    nrp1_act_gain: float = 0.6

    # downstream modules (erk, akt, src)
    k_erk_on: float = 2.5
    k_erk_off: float = 1.2
    k_akt_on: float = 2.0
    k_akt_off: float = 0.9
    k_src_on: float = 2.2
    k_src_off: float = 1.0

    # cross-talk / feedback
    erk_feedback_on_vegfr2: float = 0.35
    akt_feedback_on_internalization: float = 0.25

    # eNOS / no (akt -> enos -> no)
    k_no_prod: float = 1.8
    k_no_decay: float = 0.8

    # permeability proxy (src + no drive; relaxes back)
    k_perm_up: float = 1.4
    k_perm_down: float = 0.7

    # angiogenesis proxy (erk + akt drive; relaxes back)
    k_ang_up: float = 1.0
    k_ang_down: float = 0.25


STATE_NAMES = [
    "vegf_free",
    "vegfr2_free",
    "vegfr1_free",
    "nrp1_free",
    "vegf_bound",
    "vegf_bound_r1",
    "pvegfr2",
    "vegfr2_internal",
    "perk",
    "pakt",
    "psrc",
    "no",
    "permeability",
    "angiogenesis",
]


def initial_state(
    vegf: float = 1.0,
    vegfr2: float = 1.0,
    vegfr1: float = 0.6,
    nrp1: float = 0.8,
) -> np.ndarray:
    x0 = np.zeros(len(STATE_NAMES), dtype=float)
    x0[0] = max(0.0, vegf)
    x0[1] = max(0.0, vegfr2)
    x0[2] = max(0.0, vegfr1)
    x0[3] = max(0.0, nrp1)
    return x0


def _rhs(t: float, x: np.ndarray, p: VegfParams) -> np.ndarray:
    (
        vegf_free,
        vegfr2_free,
        vegfr1_free,
        nrp1_free,
        vegf_bound,
        vegf_bound_r1,
        pvegfr2,
        vegfr2_internal,
        perk,
        pakt,
        psrc,
        no,
        perm,
        ang,
    ) = x

    # binding and activation
    bind_r2 = p.kon * vegf_free * vegfr2_free
    unbind_r2 = p.koff * vegf_bound

    # vegfr1 decoy binding
    bind_r1 = p.kon_r1 * vegf_free * vegfr1_free
    unbind_r1 = p.koff_r1 * vegf_bound_r1

    # erk negative feedback attenuates activation effectiveness (bounded)
    act_gain = 1.0 / (1.0 + p.erk_feedback_on_vegfr2 * perk)

    # nrp1 increases effective activation (proxy for co-receptor facilitation)
    nrp1_gain = 1.0 + p.nrp1_act_gain * (nrp1_free / (1.0 + nrp1_free))
    act = p.kact * act_gain * nrp1_gain * vegf_bound
    deact = p.kdeact * pvegfr2

    # internalization: increased by akt feedback (interpretable as endocytosis machinery)
    int_gain = 1.0 + p.akt_feedback_on_internalization * pakt
    internalize = p.kint * int_gain * pvegfr2
    recycle = p.krec * vegfr2_internal

    # downstream phosphorylation (hill-like saturation via (1 - state))
    erk_on = p.k_erk_on * pvegfr2 * (1.0 - perk)
    erk_off = p.k_erk_off * perk

    akt_on = p.k_akt_on * pvegfr2 * (1.0 - pakt)
    akt_off = p.k_akt_off * pakt

    src_on = p.k_src_on * pvegfr2 * (1.0 - psrc)
    src_off = p.k_src_off * psrc

    # akt -> no
    no_prod = p.k_no_prod * pakt * (1.0 - no)
    no_decay = p.k_no_decay * no

    # permeability: driven by src and no
    perm_drive = 0.7 * psrc + 0.3 * no
    perm_up = p.k_perm_up * perm_drive * (1.0 - perm)
    perm_down = p.k_perm_down * perm

    # angiogenesis: driven by erk and akt (sprouting/growth proxy)
    ang_drive = 0.6 * perk + 0.4 * pakt
    ang_up = p.k_ang_up * ang_drive * (1.0 - ang)
    ang_down = p.k_ang_down * ang

    d = np.zeros_like(x)

    # ligand/receptor pools
    d[0] = -(bind_r2 + bind_r1) + (unbind_r2 + unbind_r1)
    d[1] = -bind_r2 + unbind_r2 + recycle
    d[2] = -bind_r1 + unbind_r1
    d[3] = 0.0
    d[4] = bind_r2 - unbind_r2 - act  # bound r2 complex consumed by activation
    d[5] = bind_r1 - unbind_r1

    # receptor activation and trafficking
    d[6] = act - deact - internalize
    d[7] = internalize - recycle

    # downstream
    d[8] = erk_on - erk_off
    d[9] = akt_on - akt_off
    d[10] = src_on - src_off

    # outputs
    d[11] = no_prod - no_decay
    d[12] = perm_up - perm_down
    d[13] = ang_up - ang_down

    return d


def simulate(
    *,
    t_end: float = 60.0,
    n_steps: int = 600,
    x0: np.ndarray | None = None,
    params: VegfParams | None = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    if params is None:
        params = VegfParams()
    if x0 is None:
        x0 = initial_state()

    t_eval = np.linspace(0.0, float(t_end), int(n_steps))

    sol = solve_ivp(
        fun=lambda t, y: _rhs(t, y, params),
        t_span=(0.0, float(t_end)),
        y0=np.asarray(x0, dtype=float),
        t_eval=t_eval,
        method="LSODA",
        rtol=1e-6,
        atol=1e-9,
    )

    if not sol.success:
        raise RuntimeError(f"ode integration failed: {sol.message}")

    series = {name: sol.y[i, :].copy() for i, name in enumerate(STATE_NAMES)}
    return sol.t.copy(), series
