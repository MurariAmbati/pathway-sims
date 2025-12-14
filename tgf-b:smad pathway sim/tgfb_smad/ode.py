from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


STATE_ORDER: List[str] = [
    "R_free",
    "RL",
    "R_star",
    "Ri",
    "S23_c",
    "pS23_c",
    "pS23_n",
    "S4_c",
    "S4_n",
    "C_n",
    "Smad7_m",
    "Smad7",
    "MAPK",
    "PI3K",
    "G_prog",
    "E_prog",
    "F_prog",
]


@dataclass(frozen=True)
class Inputs:
    ligand: float
    mapk: float
    pi3k: float


def _hill(x: float, K: float, n: float) -> float:
    x = max(0.0, float(x))
    K = max(1e-12, float(K))
    n = max(1e-6, float(n))
    xn = x**n
    return xn / (K**n + xn)


def pack_state(y_dict: Dict[str, float]) -> np.ndarray:
    return np.array([float(y_dict[k]) for k in STATE_ORDER], dtype=float)


def unpack_state(y: np.ndarray) -> Dict[str, float]:
    return {k: float(y[i]) for i, k in enumerate(STATE_ORDER)}


def rhs(t: float, y: np.ndarray, params: Dict[str, float], u: Inputs) -> np.ndarray:
    """ODE right-hand side.

    Model structure:
    - Ligand activates receptor (R*) inhibited by Smad7.
    - Active receptor phosphorylates Smad2/3 in cytosol.
    - pSmad2/3 shuttles to nucleus; Smad4 shuttles.
    - Nuclear complex C_n forms from pS23_n + S4_n and dissociates.
    - C_n induces Smad7 (negative feedback) and downstream programs (G/E/F).
    - MAPK reduces pS23 nuclear import and increases dephosphorylation.
    - PI3K synergizes with C_n to drive EMT/fibrosis programs.
    """

    (
        R_free,
        RL,
        R_star,
        Ri,
        S23_c,
        pS23_c,
        pS23_n,
        S4_c,
        S4_n,
        C_n,
        Smad7_m,
        Smad7,
        MAPK,
        PI3K,
        G_prog,
        E_prog,
        F_prog,
    ) = y

    # inputs
    ligand = max(0.0, float(u.ligand))
    mapk_drive = float(np.clip(u.mapk, 0.0, 1.0))
    pi3k_drive = float(np.clip(u.pi3k, 0.0, 1.0))

    # Smad7 inhibition of receptor activation
    inhib = 1.0 / (1.0 + (Smad7 / params["K_inhib_smad7"]) ** params["n_inhib_smad7"])

    # receptor module (explicit free receptor + homeostasis + Smad7 ubiquitination)
    R_total = R_free + RL + R_star + Ri
    homeo = params["k_R_homeo"] * (params["R_tot"] - R_total)
    homeo = float(max(-params["R_tot"], min(params["R_tot"], homeo)))

    k_act_eff = params["k_act"] * inhib
    int_boost = 1.0 + params["beta_smad7_int"] * (Smad7 / (params["K_smad7_int"] + Smad7 + 1e-12))
    k_int_star_eff = params["k_int_star"] * int_boost

    ub = _hill(Smad7, params["K_ub_smad7"], params["n_ub_smad7"])
    ub_rl = params["k_ub_rl"] * ub * RL
    ub_star = params["k_ub_star"] * ub * R_star

    bind = params["k_bind"] * ligand * R_free
    unbind = params["k_unbind"] * RL

    dR_free = (
        homeo
        - params["k_R_deg"] * R_free
        - bind
        + unbind
        + params["k_rec"] * Ri
    )

    dRL = bind - unbind - k_act_eff * RL - params["k_int_rl"] * RL - params["k_R_deg"] * RL - ub_rl
    dR_star = k_act_eff * RL - params["k_deact"] * R_star - k_int_star_eff * R_star - params["k_R_deg"] * R_star - ub_star
    dRi = params["k_int_rl"] * RL + k_int_star_eff * R_star - params["k_rec"] * Ri - params["k_R_deg"] * Ri

    # allow deactivation to return to RL
    dRL += params["k_deact"] * R_star

    # MAPK/PI3K dynamics (bounded relaxation to a ligand+drive target)
    mapk_target = float(np.clip(mapk_drive + params["mapk_tgfb_gain"] * R_star, 0.0, 1.0))
    pi3k_target = float(np.clip(pi3k_drive + params["pi3k_tgfb_gain"] * R_star, 0.0, 1.0))

    dMAPK = params["k_mapk_relax"] * (mapk_target - MAPK)
    dPI3K = params["k_pi3k_relax"] * (pi3k_target - PI3K)

    # MAPK modifies effective shuttling + dephos
    imp_scale = 1.0 / (1.0 + params["alpha_mapk_imp"] * params["k_linker_phos_mapk"] * MAPK)
    dp_c_eff = params["k_dp_c"] * (1.0 + params["alpha_mapk_dp"] * params["k_linker_phos_mapk"] * MAPK)
    dp_n_eff = params["k_dp_n"] * (1.0 + params["alpha_mapk_dp"] * params["k_linker_phos_mapk"] * MAPK)

    # phosphorylation/dephosphorylation
    dpS23_c = params["k_p"] * R_star * S23_c - dp_c_eff * pS23_c
    dS23_c = -params["k_p"] * R_star * S23_c + dp_c_eff * pS23_c

    # pS23 cytosol <-> nucleus shuttling
    jp_in = params["k_imp_pS23"] * imp_scale * pS23_c
    jp_out = params["k_exp_pS23"] * pS23_n
    dpS23_c += -jp_in + jp_out
    dpS23_n = jp_in - jp_out

    # nuclear dephosphorylation returns to cytosolic unphosphorylated pool via export (phenomenological)
    # (keeps mass balance stable without adding explicit nuclear unphosphorylated state)
    dpS23_n += -dp_n_eff * pS23_n
    dS23_c += dp_n_eff * pS23_n

    # Smad4 shuttling
    j4_in = params["k_imp_S4"] * S4_c
    j4_out = params["k_exp_S4"] * S4_n
    dS4_c = -j4_in + j4_out
    dS4_n = j4_in - j4_out

    # nuclear complex formation/dissociation
    j_on = params["k_on"] * pS23_n * S4_n
    j_off = params["k_off"] * C_n
    dpS23_n += -j_on + j_off
    dS4_n += -j_on + j_off
    dC_n = j_on - j_off

    # Smad7 induction (mRNA -> protein)
    smad7_drive = _hill(C_n, params["K_smad7"], params["n_smad7"])
    dSmad7_m = params["k_smad7_txn"] * smad7_drive - params["k_smad7_mdeg"] * Smad7_m
    dSmad7 = params["k_smad7_tln"] * Smad7_m - params["k_smad7_deg"] * Smad7

    # downstream programs
    prog_drive = _hill(C_n, params["K_prog"], params["n_prog"])

    # PI3K synergy: multiplicative boost for EMT/fibrosis (kept bounded)
    synergy = 1.0 + params["k_pi3k_synergy"] * PI3K

    dG = params["k_G_prod"] * prog_drive - params["k_prog_deg"] * G_prog
    dE = params["k_E_prod"] * prog_drive * synergy - params["k_prog_deg"] * E_prog
    dF = params["k_F_prod"] * prog_drive * synergy - params["k_prog_deg"] * F_prog

    return np.array(
        [
            dR_free,
            dRL,
            dR_star,
            dRi,
            dS23_c,
            dpS23_c,
            dpS23_n,
            dS4_c,
            dS4_n,
            dC_n,
            dSmad7_m,
            dSmad7,
            dMAPK,
            dPI3K,
            dG,
            dE,
            dF,
        ],
        dtype=float,
    )


def enforce_physical_constraints(y: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    y = np.array(y, dtype=float, copy=True)
    y[y < 0.0] = 0.0

    # Soft-bound signaling nodes
    for name in ("MAPK", "PI3K"):
        try:
            idx = STATE_ORDER.index(name)
            y[idx] = float(np.clip(y[idx], 0.0, 1.0))
        except ValueError:
            pass

    return y


def mass_balance_checks(y: np.ndarray, params: Dict[str, float]) -> Tuple[float, float]:
    """Returns (S23_total_like, S4_total_like) to help sanity check conserved pools."""
    state = unpack_state(y)
    s23_total_like = state["S23_c"] + state["pS23_c"] + state["pS23_n"] + state["C_n"]
    s4_total_like = state["S4_c"] + state["S4_n"] + state["C_n"]
    return s23_total_like / max(1e-12, params["S23_tot"]), s4_total_like / max(1e-12, params["S4_tot"])
