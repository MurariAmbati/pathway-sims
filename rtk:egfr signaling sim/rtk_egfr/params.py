from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    # ligand + receptor binding / activation
    ligand: float = 1.0  # external dose (treated as constant input)
    r_tot: float = 1.0
    k_on: float = 1.0
    k_off: float = 0.2

    # dimerization / activation
    k_dim: float = 1.5
    k_undim: float = 0.15
    k_r2_deact: float = 0.25

    # ras module (fractions in [0,1])
    k_ras_act: float = 2.0
    k_ras_gap: float = 1.0
    ras_km: float = 0.15

    # raf/mek/erk cascade
    k_raf_act: float = 2.0
    k_raf_deact: float = 0.8
    k_mek_phos: float = 2.4
    k_mek_dephos: float = 1.2
    k_erk_phos: float = 3.0
    k_erk_dephos: float = 1.4

    # pi3k/akt arm
    k_pi3k_act: float = 1.8
    k_pi3k_deact: float = 0.9
    pi3k_ras_crosstalk: float = 0.35

    k_akt_phos: float = 2.2
    k_akt_dephos: float = 1.2

    # negative feedback (erk -> ras gap)
    feedback_strength: float = 1.2

    # numerical safety
    clamp_eps: float = 1e-9


def default_params() -> Params:
    return Params()


def default_initial_state(params: Params | None = None) -> dict[str, float]:
    p = params or default_params()
    # receptor species are concentrations; downstream are fractions
    return {
        "R": p.r_tot,
        "RL": 0.0,
        "R2": 0.0,
        "RAS_GTP": 0.02,
        "RAF_act": 0.02,
        "MEK_p": 0.02,
        "ERK_p": 0.02,
        "PI3K_act": 0.02,
        "AKT_p": 0.02,
    }


PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "ligand": (0.0, 10.0),
    "k_on": (0.0, 5.0),
    "k_off": (0.0, 5.0),
    "k_dim": (0.0, 5.0),
    "k_undim": (0.0, 5.0),
    "k_r2_deact": (0.0, 5.0),
    "k_ras_act": (0.0, 10.0),
    "k_ras_gap": (0.0, 10.0),
    "ras_km": (0.01, 1.0),
    "k_raf_act": (0.0, 10.0),
    "k_raf_deact": (0.0, 10.0),
    "k_mek_phos": (0.0, 10.0),
    "k_mek_dephos": (0.0, 10.0),
    "k_erk_phos": (0.0, 10.0),
    "k_erk_dephos": (0.0, 10.0),
    "k_pi3k_act": (0.0, 10.0),
    "k_pi3k_deact": (0.0, 10.0),
    "pi3k_ras_crosstalk": (0.0, 2.0),
    "k_akt_phos": (0.0, 10.0),
    "k_akt_dephos": (0.0, 10.0),
    "feedback_strength": (0.0, 5.0),
}
