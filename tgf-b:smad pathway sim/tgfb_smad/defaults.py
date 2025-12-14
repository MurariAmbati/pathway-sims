from __future__ import annotations

from typing import Dict


def default_parameters() -> Dict[str, float]:
    """Default parameters in consistent time units (minutes).

    This is a phenomenological-but-mechanistic model, tuned for stable interactive simulation.
    """

    return {
        # receptor / ligand (surface binding → activation → internalization → recycling)
        # R_tot is treated as a homeostatic set-point for total receptor abundance.
        "R_tot": 1.0,
        "k_bind": 1.4,  # ligand binding to free receptor (per min per ligand unit)
        "k_unbind": 0.7,
        "k_act": 0.35,  # activation of ligand-bound complex
        "k_deact": 0.22,
        "k_int_rl": 0.10,  # internalization of ligand-bound (inactive) complex
        "k_int_star": 0.14,  # internalization of active receptor
        "k_rec": 0.06,  # recycling back to surface
        "k_R_homeo": 0.03,  # homeostatic synthesis toward R_tot
        "k_R_deg": 0.008,  # basal receptor degradation (all receptor states)
        # Smad7 inhibition of receptor activation and promotion of internalization
        "K_inhib_smad7": 0.6,
        "n_inhib_smad7": 2.0,
        "beta_smad7_int": 1.2,
        "K_smad7_int": 0.6,
        # Smad7-mediated ubiquitination (e.g., SMURF) removing receptor from the system
        "k_ub_rl": 0.06,
        "k_ub_star": 0.10,
        "K_ub_smad7": 0.6,
        "n_ub_smad7": 2.0,
        # SMAD phosphorylation/dephosphorylation
        "S23_tot": 1.0,
        "S4_tot": 1.0,
        "k_p": 0.9,
        "k_dp_c": 0.30,
        "k_dp_n": 0.22,
        # cytosol-nucleus shuttling
        "k_imp_pS23": 0.18,
        "k_exp_pS23": 0.10,
        "k_imp_S4": 0.06,
        "k_exp_S4": 0.06,
        # complex formation (nucleus)
        "k_on": 2.2,
        "k_off": 0.9,
        # Smad7 transcription/translation/turnover
        "k_smad7_txn": 0.40,
        "K_smad7": 0.25,
        "n_smad7": 2.0,
        "k_smad7_mdeg": 0.18,
        "k_smad7_tln": 0.60,
        "k_smad7_deg": 0.06,
        # MAPK dynamics + crosstalk
        "k_mapk_relax": 0.25,
        "mapk_tgfb_gain": 0.55,
        "k_linker_phos_mapk": 1.0,
        "alpha_mapk_imp": 0.8,
        "alpha_mapk_dp": 0.6,
        # PI3K dynamics + synergy
        "k_pi3k_relax": 0.25,
        "pi3k_tgfb_gain": 0.35,
        "k_pi3k_synergy": 1.2,
        # downstream programs: (G = growth inhibition, E = EMT, F = fibrosis)
        "k_G_prod": 0.08,
        "k_E_prod": 0.06,
        "k_F_prod": 0.06,
        "K_prog": 0.20,
        "n_prog": 2.0,
        "k_prog_deg": 0.02,
    }


def default_initial_conditions(params: Dict[str, float]) -> Dict[str, float]:
    # receptor states (explicit)
    R_free = params["R_tot"]
    RL = 0.0
    R_star = 0.0
    Ri = 0.0

    # SMAD pools
    S23_c = params["S23_tot"]
    pS23_c = 0.0
    pS23_n = 0.0

    S4_c = params["S4_tot"]
    S4_n = 0.0

    C_n = 0.0

    Smad7_m = 0.0
    Smad7 = 0.0

    MAPK = 0.0
    PI3K = 0.0

    # programs
    G_prog = 0.0
    E_prog = 0.0
    F_prog = 0.0

    return {
        "R_free": R_free,
        "RL": RL,
        "R_star": R_star,
        "Ri": Ri,
        "S23_c": S23_c,
        "pS23_c": pS23_c,
        "pS23_n": pS23_n,
        "S4_c": S4_c,
        "S4_n": S4_n,
        "C_n": C_n,
        "Smad7_m": Smad7_m,
        "Smad7": Smad7,
        "MAPK": MAPK,
        "PI3K": PI3K,
        "G_prog": G_prog,
        "E_prog": E_prog,
        "F_prog": F_prog,
    }
