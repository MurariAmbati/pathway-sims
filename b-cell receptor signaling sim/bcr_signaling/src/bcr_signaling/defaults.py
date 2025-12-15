from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Defaults:
    """default parameterization and initial conditions.

    this is a deliberately compact, interpretable bcr→nf-κb model with overlap
    to tcr-like proximal signaling (src/syk, plcγ, pkcθ/β) and canonical ikk/nf-κb.
    units are arbitrary but consistent.
    """

    t_end: float = 60.0
    n_steps: int = 600

    # input stimulus
    antigen: float = 1.0

    # proximal activation (bcr→syk)
    k_bcr_on: float = 2.0
    k_bcr_off: float = 0.8

    k_syk_act: float = 3.0
    k_syk_deact: float = 1.2

    # adaptor + plcγ2 branch
    k_plcg_act: float = 2.2
    k_plcg_deact: float = 1.0

    k_ca_in: float = 3.5
    k_ca_out: float = 2.2

    k_pkc_act: float = 2.0
    k_pkc_deact: float = 1.0

    # pi3k/akt survival branch
    k_pi3k_act: float = 1.6
    k_pi3k_deact: float = 0.9

    k_akt_act: float = 1.7
    k_akt_deact: float = 0.9

    # mapk (ras/erk) branch (activation marker)
    k_mapk_act: float = 1.4
    k_mapk_deact: float = 0.8

    # ikk and nf-κb module (canonical)
    k_ikk_act: float = 2.4
    k_ikk_deact: float = 1.1

    k_ikb_phos: float = 2.2
    k_ikb_resyn: float = 0.9

    k_nfkb_import: float = 2.0
    k_nfkb_export: float = 1.1

    # negative feedback (a20-like) induced by nuclear nf-κb
    k_a20_prod: float = 1.0
    k_a20_deg: float = 0.6

    k_a20_inhib: float = 2.0

    # phosphatase brake (shp1/ptpn6 lump)
    shp1: float = 1.0

    # pten brake on pi3k
    pten: float = 1.0


DEFAULTS = Defaults()


def initial_conditions() -> dict[str, float]:
    """baseline resting state."""

    return {
        "bcr_active": 0.0,
        "syk_active": 0.0,
        "plcg_active": 0.0,
        "ca": 0.05,
        "pkc_active": 0.0,
        "pi3k_active": 0.0,
        "akt_active": 0.0,
        "mapk_active": 0.0,
        "ikk_active": 0.0,
        "ikb": 1.0,
        "nfkb_nuclear": 0.05,
        "a20": 0.0,
    }
