from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class PDGFParams:
    # Ligand / receptor binding
    k_on: float = 1.0  # (1/(nM*min)) PDGF + R -> C
    k_off: float = 0.2  # (1/min) C -> PDGF + R

    # Dimerization (active signaling complex)
    k_dim: float = 0.8  # (1/min) 2C -> D
    k_undim: float = 0.2  # (1/min) D -> 2C

    # Receptor phosphorylation / dephosphorylation
    k_phos: float = 1.2  # (1/min) D -> Dp
    k_dephos: float = 0.6  # (1/min) Dp -> D

    # Internalization / degradation of active receptor
    k_int: float = 0.15  # (1/min) Dp -> sink

    # Downstream nodes (coarse-grained)
    k_erk_act: float = 1.6  # (1/min)
    k_erk_deact: float = 0.7  # (1/min)

    k_akt_act: float = 1.2  # (1/min)
    k_akt_deact: float = 0.6  # (1/min)

    # Cross-talk / saturation
    erk_K: float = 0.5  # (dimensionless)
    akt_K: float = 0.5

    # Proliferation proxy (driven by ERK/AKT)
    k_prolif: float = 0.8  # (1/min)
    k_prolif_decay: float = 0.25  # (1/min)
    w_erk: float = 0.65
    w_akt: float = 0.35


STATE_NAMES: Tuple[str, ...] = (
    "L",  # ligand (extracellular PDGF)
    "R",  # free receptor
    "C",  # ligand-bound monomeric complex
    "D",  # dimerized complex
    "Dp",  # phosphorylated dimer (active)
    "ERK",  # normalized activation (0..1)
    "AKT",  # normalized activation (0..1)
    "P",  # proliferation drive proxy
)


def default_initial_state(
    ligand_nM: float = 2.0,
    receptor_nM: float = 10.0,
) -> np.ndarray:
    y0 = np.zeros(len(STATE_NAMES), dtype=float)
    y0[0] = float(ligand_nM)
    y0[1] = float(receptor_nM)
    return y0


def _hill(x: float, K: float) -> float:
    # simple saturating nonlinearity
    return float(x) / (float(K) + float(x) + 1e-12)


def pdgf_ode(t: float, y: np.ndarray, p: PDGFParams) -> np.ndarray:
    L, R, C, D, Dp, ERK, AKT, P = y

    # Binding
    v_bind = p.k_on * L * R
    v_unbind = p.k_off * C

    # Dimerization / undimerization
    v_dim = p.k_dim * C * C
    v_undim = p.k_undim * D

    # Phosphorylation cycle (active receptor)
    v_phos = p.k_phos * D
    v_dephos = p.k_dephos * Dp

    # Internalization sink
    v_int = p.k_int * Dp

    # Downstream activation driven by active receptor (Dp)
    erk_drive = _hill(Dp, p.erk_K)
    akt_drive = _hill(Dp, p.akt_K)

    dL = -v_bind + v_unbind
    dR = -v_bind + v_unbind
    dC = +v_bind - v_unbind - 2.0 * v_dim + 2.0 * v_undim
    dD = +v_dim - v_undim - v_phos + v_dephos
    dDp = +v_phos - v_dephos - v_int

    dERK = p.k_erk_act * erk_drive * (1.0 - ERK) - p.k_erk_deact * ERK
    dAKT = p.k_akt_act * akt_drive * (1.0 - AKT) - p.k_akt_deact * AKT

    # Proliferation proxy integrates weighted ERK/AKT activity
    drive = p.w_erk * ERK + p.w_akt * AKT
    dP = p.k_prolif * drive - p.k_prolif_decay * P

    return np.array([dL, dR, dC, dD, dDp, dERK, dAKT, dP], dtype=float)


def preset_params(name: str) -> PDGFParams:
    key = name.strip().lower()
    if key in {"baseline", "default"}:
        return PDGFParams()
    if key in {"fast", "strong"}:
        return PDGFParams(k_phos=1.8, k_erk_act=2.0, k_akt_act=1.6)
    if key in {"slow", "weak"}:
        return PDGFParams(k_phos=0.7, k_erk_act=1.0, k_akt_act=0.8, k_int=0.2)
    if key in {"high internalization", "internalization"}:
        return PDGFParams(k_int=0.35)
    raise ValueError(f"Unknown preset: {name}")


def params_to_dict(p: PDGFParams) -> Dict[str, float]:
    return {
        "k_on": p.k_on,
        "k_off": p.k_off,
        "k_dim": p.k_dim,
        "k_undim": p.k_undim,
        "k_phos": p.k_phos,
        "k_dephos": p.k_dephos,
        "k_int": p.k_int,
        "k_erk_act": p.k_erk_act,
        "k_erk_deact": p.k_erk_deact,
        "k_akt_act": p.k_akt_act,
        "k_akt_deact": p.k_akt_deact,
        "erk_K": p.erk_K,
        "akt_K": p.akt_K,
        "k_prolif": p.k_prolif,
        "k_prolif_decay": p.k_prolif_decay,
        "w_erk": p.w_erk,
        "w_akt": p.w_akt,
    }


def dict_to_params(d: Dict[str, float]) -> PDGFParams:
    base = params_to_dict(PDGFParams())
    base.update({k: float(v) for k, v in d.items() if k in base})
    return PDGFParams(**base)


def stoichiometry_summary() -> str:
    return (
        "States: L (ligand), R (receptor), C (bound), D (dimer), Dp (phospho), ERK, AKT, P (prolif proxy).\n"
        "Reactions: L+R <-> C; 2C <-> D; D <-> Dp; Dp -> sink; Dp drives ERK/AKT; ERK/AKT drive P."
    )
