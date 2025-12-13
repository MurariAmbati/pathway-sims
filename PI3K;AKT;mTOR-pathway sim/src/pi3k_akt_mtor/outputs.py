from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


DERIVED_COLUMNS = (
    "IRS1_func",
    "TSC_act",
    "Growth",
    "Survival",
    "Metabolism",
)


def add_derived_outputs(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived readouts for visualization/research convenience.

    Assumes the PI3K/AKT/mTOR state columns exist.
    """

    out = df.copy()

    if "Is" in out.columns:
        out["IRS1_func"] = 1.0 - out["Is"].astype(float)

    if "Ti" in out.columns:
        out["TSC_act"] = 1.0 - out["Ti"].astype(float)

    # Composite, dimensionless proxies (kept simple and transparent)
    m1 = out["M1"].astype(float) if "M1" in out.columns else 0.0
    s6k = out["S"].astype(float) if "S" in out.columns else 0.0
    akt = out["Ap"].astype(float) if "Ap" in out.columns else 0.0
    m2 = out["M2"].astype(float) if "M2" in out.columns else 0.0
    au = out["Au"].astype(float) if "Au" in out.columns else 0.0

    out["Growth"] = np.clip(0.6 * m1 + 0.4 * s6k, 0.0, 1.0)
    out["Survival"] = np.clip(0.7 * akt + 0.3 * m2 - 0.3 * au, 0.0, 1.0)
    out["Metabolism"] = np.clip(0.5 * akt + 0.5 * m1, 0.0, 1.0)

    return out


def describe_outputs() -> Dict[str, str]:
    return {
        "IRS1_func": "Functional IRS1 fraction (1 - Is)",
        "TSC_act": "Active TSC fraction (1 - Ti)",
        "Growth": "Proxy growth/translation output (M1/S6K composite)",
        "Survival": "Proxy survival output (AKT/mTORC2 minus autophagy)",
        "Metabolism": "Proxy metabolic output (AKT/mTORC1 composite)",
    }
