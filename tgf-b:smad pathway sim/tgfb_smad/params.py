from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class Params:
    # Receptor / ligand handling
    R_tot: float = 1.0
    k_on: float = 2.0
    k_off: float = 0.4
    k_int: float = 0.6
    k_rec: float = 0.15
    k_R_synth: float = 0.02
    k_R_deg: float = 0.01

    # SMAD core
    S_tot: float = 1.0
    S4_tot: float = 0.6
    k_phos: float = 1.2
    K_phos: float = 0.25
    k_dephos: float = 0.7

    k_bind: float = 3.0
    k_unbind: float = 0.8

    k_import0: float = 0.9
    k_export: float = 0.35

    # Negative feedback (SMAD7)
    beta_s7: float = 6.0
    k_s7_txn: float = 0.8
    K_s7: float = 0.25
    n_s7: float = 2.0
    k_s7_mdeg: float = 0.5
    k_s7_tln: float = 1.0
    k_s7_pdeg: float = 0.35

    # Gene programs driven by nuclear SMAD complex
    k_emt_txn: float = 0.6
    k_fib_txn: float = 0.75
    k_p21_txn: float = 0.55
    K_gene: float = 0.25
    n_gene: float = 2.0
    k_gene_mdeg: float = 0.25

    # MAPK (ERK) crosstalk (non-canonical + RTK)
    mapk_basal: float = 0.05
    mapk_rtk_gain: float = 0.9
    mapk_tgfb_gain: float = 0.25
    k_mapk_off: float = 0.8

    alpha_mapk_import: float = 2.5
    k_linker_deg_sp: float = 0.35
    k_linker_deg_n: float = 0.12

    # PI3K/AKT crosstalk
    pi3k_basal: float = 0.05
    pi3k_rtk_gain: float = 0.9
    pi3k_tgfb_gain: float = 0.18
    k_pi3k_off: float = 0.7

    alpha_pi3k_s7_repress: float = 1.8

    # Numerical safety
    eps: float = 1e-12


def default_params() -> Params:
    return Params()


def default_initial_conditions(params: Params | None = None) -> Dict[str, float]:
    p = params or default_params()
    return {
        "R": 0.9 * p.R_tot,
        "C": 0.0,
        "Ri": 0.0,
        "S": 0.9 * p.S_tot,
        "Sp": 0.0,
        "S4": 0.9 * p.S4_tot,
        "SpS4": 0.0,
        "N": 0.0,
        "S7_m": 0.0,
        "S7": 0.0,
        "EMT_m": 0.0,
        "FIB_m": 0.0,
        "P21_m": 0.0,
        "MAPK": p.mapk_basal,
        "PI3K": p.pi3k_basal,
    }


STATE_ORDER = [
    "R",
    "C",
    "Ri",
    "S",
    "Sp",
    "S4",
    "SpS4",
    "N",
    "S7_m",
    "S7",
    "EMT_m",
    "FIB_m",
    "P21_m",
    "MAPK",
    "PI3K",
]


def pack_state(x: Dict[str, float]) -> np.ndarray:
    return np.array([float(x[k]) for k in STATE_ORDER], dtype=float)


def unpack_state(y: np.ndarray) -> Dict[str, float]:
    return {k: float(y[i]) for i, k in enumerate(STATE_ORDER)}
