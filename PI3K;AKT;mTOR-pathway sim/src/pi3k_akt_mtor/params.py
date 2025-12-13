from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Inputs:
    """External inputs (dimensionless) and drug multipliers.

    All are intended to be in [0, 1] where applicable.
    """

    ligand: float = 1.0  # RTK drive
    erk_input: float = 0.0  # ERK pathway drive (crosstalk variant)
    ampk_input: float = 0.0  # AMPK pathway drive (crosstalk variant)

    # Drug multipliers: 1.0 means no inhibition; 0.0 means fully inhibited.
    pi3k_activity: float = 1.0
    akt_activity: float = 1.0
    mtorc1_activity: float = 1.0


@dataclass(frozen=True)
class ModelParams:
    """Kinetic parameters for the PI3K/AKT/mTOR ODE model.

    Units: time is in arbitrary units; rate constants are per-time.
    State variables are normalized fractions in [0, 1].
    """

    # Receptor activation
    k_R_on: float = 0.08
    k_R_off: float = 0.05

    # IRS1 inhibition / recovery (baseline has only recovery; feedback adds inhibition terms)
    k_IRS_deinhib: float = 0.05
    k_S6K_to_IRS: float = 0.0  # enabled by feedback variant
    k_ERK_to_IRS: float = 0.0  # enabled by crosstalk variant
    k_FOXO_to_IRS_rescue: float = 0.12  # FOXO-dependent restoration of functional IRS1

    # PI3K
    k_PI3K_on: float = 0.4
    k_PI3K_off: float = 0.25

    # PIP3 balance
    v_PI3K: float = 0.8
    v_PTEN: float = 0.7

    # mTORC2
    k_mTORC2_on: float = 0.12
    k_mTORC2_off: float = 0.06
    k_S6K_to_mTORC2_inhib: float = 0.08  # negative feedback: S6K dampens mTORC2

    # AKT phosphorylation/dephosphorylation
    v_PDK1: float = 0.8
    v_mTORC2_to_AKT: float = 0.5
    v_AKT_dephos: float = 0.35

    # FOXO (active nuclear FOXO; inhibited by AKT)
    k_FOXO_on: float = 0.25
    k_AKT_to_FOXO_inhib: float = 0.9
    k_FOXO_off: float = 0.25

    # TSC inhibition / recovery
    k_AKT_to_TSC: float = 0.8
    k_ERK_to_TSC: float = 0.0  # enabled by crosstalk variant
    k_TSC_recover: float = 0.25

    # mTORC1
    k_mTORC1_on: float = 0.5
    k_mTORC1_off: float = 0.18
    alpha_AMPK: float = 0.0  # enabled by crosstalk variant

    # S6K
    k_S6K_on: float = 0.7
    k_S6K_off: float = 0.35

    # ERK (crosstalk)
    k_ERK_on: float = 0.25
    k_ERK_off: float = 0.2

    # AMPK (crosstalk)
    k_AMPK_on: float = 0.25
    k_AMPK_off: float = 0.2

    # Autophagy (metabolic branch)
    k_AUT_on: float = 0.35
    k_AUT_off: float = 0.2
    beta_AUT_AMPK: float = 1.0
    beta_AUT_mTORC1: float = 1.0

    def as_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self.__dict__.items()}
