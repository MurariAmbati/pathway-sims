from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


STATE_ORDER = [
    "bcr_active",
    "syk_active",
    "plcg_active",
    "ca",
    "pkc_active",
    "pi3k_active",
    "akt_active",
    "mapk_active",
    "ikk_active",
    "ikb",
    "nfkb_nuclear",
    "a20",
]


@dataclass(frozen=True)
class Interventions:
    """simple toggles for perturbations.

    values are multipliers in [0, 1] unless otherwise stated.
    """

    syk_inhib: float = 0.0
    btk_inhib: float = 0.0
    pi3k_inhib: float = 0.0
    ikk_inhib: float = 0.0

    shp1_knockdown: float = 0.0
    pten_knockdown: float = 0.0


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _apply_interventions(params: dict[str, float], interventions: Interventions) -> dict[str, float]:
    p = dict(params)

    # interpret inhibitors as fractional inhibition; 1.0 means fully off
    syk_scale = 1.0 - _clamp01(interventions.syk_inhib)
    btk_scale = 1.0 - _clamp01(interventions.btk_inhib)
    pi3k_scale = 1.0 - _clamp01(interventions.pi3k_inhib)
    ikk_scale = 1.0 - _clamp01(interventions.ikk_inhib)

    # map btk inhibition to plcγ and pkc activation (btk couples bcr/syk to plcγ2)
    p["k_plcg_act"] *= (0.3 + 0.7 * btk_scale)
    p["k_pkc_act"] *= (0.4 + 0.6 * btk_scale)

    p["k_syk_act"] *= syk_scale
    p["k_pi3k_act"] *= pi3k_scale
    p["k_ikk_act"] *= ikk_scale

    # knockdowns reduce brakes
    p["shp1"] *= 1.0 - _clamp01(interventions.shp1_knockdown)
    p["pten"] *= 1.0 - _clamp01(interventions.pten_knockdown)

    return p


def odes(t: float, y: np.ndarray, p: dict[str, float]) -> np.ndarray:
    (
        bcr_active,
        syk_active,
        plcg_active,
        ca,
        pkc_active,
        pi3k_active,
        akt_active,
        mapk_active,
        ikk_active,
        ikb,
        nfkb_nuclear,
        a20,
    ) = y

    antigen = p["antigen"]

    # proximal receptor activation with phosphatase brake
    dbcr = p["k_bcr_on"] * antigen * (1.0 - bcr_active) - p["k_bcr_off"] * bcr_active

    # syk activation depends on bcr_active and is damped by shp1
    syk_drive = p["k_syk_act"] * bcr_active * (1.0 - syk_active) / (1.0 + 1.5 * p["shp1"])
    dsyk = syk_drive - p["k_syk_deact"] * syk_active

    # plcγ2 activation (btk is handled via intervention scaling)
    dplcg = p["k_plcg_act"] * syk_active * (1.0 - plcg_active) - p["k_plcg_deact"] * plcg_active

    # calcium is driven by plcγ2; relaxes back to baseline
    dca = p["k_ca_in"] * plcg_active - p["k_ca_out"] * max(0.0, ca - 0.05)

    # pkc activation by ca (proxy for dag/ca)
    dpkc = p["k_pkc_act"] * ca * (1.0 - pkc_active) - p["k_pkc_deact"] * pkc_active

    # pi3k activation depends on syk but inhibited by pten
    pi3k_drive = p["k_pi3k_act"] * syk_active * (1.0 - pi3k_active) / (1.0 + 1.2 * p["pten"])
    dpi3k = pi3k_drive - p["k_pi3k_deact"] * pi3k_active

    dakt = p["k_akt_act"] * pi3k_active * (1.0 - akt_active) - p["k_akt_deact"] * akt_active

    # mapk branch as an activation marker from syk and pkc
    dmapk = p["k_mapk_act"] * (0.6 * syk_active + 0.4 * pkc_active) * (1.0 - mapk_active) - p[
        "k_mapk_deact"
    ] * mapk_active

    # ikk activation by pkc and syk; inhibited by a20 feedback (tnfaip3-like)
    ikk_drive = p["k_ikk_act"] * (0.7 * pkc_active + 0.3 * syk_active) * (1.0 - ikk_active)
    ikk_drive *= 1.0 / (1.0 + p["k_a20_inhib"] * a20)
    dikk = ikk_drive - p["k_ikk_deact"] * ikk_active

    # ikb is depleted by ikk phosphorylation and resynthesized
    dikb = -p["k_ikb_phos"] * ikk_active * ikb + p["k_ikb_resyn"] * (1.0 - ikb)

    # nfkb nuclear translocation increases when ikb is low and ikk is high
    import_drive = p["k_nfkb_import"] * ikk_active * (1.0 - ikb) * (1.0 - nfkb_nuclear)
    dnuc = import_drive - p["k_nfkb_export"] * nfkb_nuclear * (0.4 + 0.6 * ikb)

    # a20-like negative feedback induced by nuclear nfkb
    da20 = p["k_a20_prod"] * nfkb_nuclear * (1.0 - a20) - p["k_a20_deg"] * a20

    return np.array([dbcr, dsyk, dplcg, dca, dpkc, dpi3k, dakt, dmapk, dikk, dikb, dnuc, da20])


def simulate(
    *,
    params: dict[str, float],
    y0: dict[str, float],
    t_end: float,
    n_steps: int,
    interventions: Interventions | None = None,
) -> pd.DataFrame:
    p = _apply_interventions(params, interventions or Interventions())

    t_eval = np.linspace(0.0, float(t_end), int(n_steps))
    y0_vec = np.array([float(y0[k]) for k in STATE_ORDER], dtype=float)

    sol = solve_ivp(
        fun=lambda t, y: odes(t, y, p),
        t_span=(0.0, float(t_end)),
        y0=y0_vec,
        t_eval=t_eval,
        method="LSODA",
        rtol=1e-6,
        atol=1e-9,
    )

    if not sol.success:
        raise RuntimeError(f"simulation failed: {sol.message}")

    df = pd.DataFrame(sol.y.T, columns=STATE_ORDER)
    df.insert(0, "t", sol.t)

    # derived markers
    df["nfkb_activity"] = df["nfkb_nuclear"]
    df["calcium"] = df["ca"]

    return df


def network() -> nx.DiGraph:
    g = nx.DiGraph()

    nodes = {
        "antigen": "input",
        "bcr": "proximal",
        "syk": "proximal",
        "plcγ2": "plcg",
        "ca2+": "second_messenger",
        "pkcβ": "pkc",
        "pi3k": "pi3k",
        "akt": "akt",
        "mapk": "mapk",
        "ikk": "nfkb",
        "iκb": "nfkb",
        "nf-κb (nuclear)": "nfkb",
        "a20": "feedback",
        "shp1": "brake",
        "pten": "brake",
    }

    for n, group in nodes.items():
        g.add_node(n, group=group)

    edges = [
        ("antigen", "bcr", {"sign": "+"}),
        ("bcr", "syk", {"sign": "+"}),
        ("shp1", "syk", {"sign": "-"}),
        ("syk", "plcγ2", {"sign": "+"}),
        ("plcγ2", "ca2+", {"sign": "+"}),
        ("ca2+", "pkcβ", {"sign": "+"}),
        ("syk", "pi3k", {"sign": "+"}),
        ("pten", "pi3k", {"sign": "-"}),
        ("pi3k", "akt", {"sign": "+"}),
        ("syk", "mapk", {"sign": "+"}),
        ("pkcβ", "mapk", {"sign": "+"}),
        ("pkcβ", "ikk", {"sign": "+"}),
        ("syk", "ikk", {"sign": "+"}),
        ("a20", "ikk", {"sign": "-"}),
        ("ikk", "iκb", {"sign": "-"}),
        ("iκb", "nf-κb (nuclear)", {"sign": "-"}),
        ("ikk", "nf-κb (nuclear)", {"sign": "+"}),
        ("nf-κb (nuclear)", "a20", {"sign": "+"}),
    ]

    g.add_edges_from([(u, v, attrs) for u, v, attrs in edges])
    return g
