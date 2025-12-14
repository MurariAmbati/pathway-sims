from __future__ import annotations

from dataclasses import dataclass

import networkx as nx


@dataclass(frozen=True)
class Node:
    id: str
    label: str
    kind: str  # receptor | gtpase | kinase | lipid_kinase | transcription


NODES: list[Node] = [
    Node("L", "ligand", "input"),
    Node("EGFR", "egfr", "receptor"),
    Node("RAS", "ras", "gtpase"),
    Node("RAF", "raf", "kinase"),
    Node("MEK", "mek", "kinase"),
    Node("ERK", "erk", "kinase"),
    Node("PI3K", "pi3k", "lipid_kinase"),
    Node("AKT", "akt", "kinase"),
    Node("PROLIF", "proliferation", "phenotype"),
    Node("SURV", "survival", "phenotype"),
]

# edges are directional influence (activation unless noted)
EDGES: list[tuple[str, str, str]] = [
    ("L", "EGFR", "+"),
    ("EGFR", "RAS", "+"),
    ("RAS", "RAF", "+"),
    ("RAF", "MEK", "+"),
    ("MEK", "ERK", "+"),
    ("EGFR", "PI3K", "+"),
    ("RAS", "PI3K", "+"),
    ("PI3K", "AKT", "+"),
    ("ERK", "PROLIF", "+"),
    ("AKT", "SURV", "+"),
    ("ERK", "RAS", "-"),  # negative feedback (via gaps / sprouty-like)
]


def build_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    for n in NODES:
        g.add_node(n.id, label=n.label, kind=n.kind)
    for src, dst, sign in EDGES:
        g.add_edge(src, dst, sign=sign)
    return g


def activity_from_state(state: dict[str, float], ligand: float) -> dict[str, float]:
    """Map model state to pathway node activities in [0,1] for visualization."""
    # receptor: use active dimer normalized to total receptor if available
    r_tot = max(state.get("R", 0.0) + state.get("RL", 0.0) + 2.0 * state.get("R2", 0.0), 1e-9)
    egfr_act = min(max(state.get("R2", 0.0) / r_tot * 2.0, 0.0), 1.0)

    out = {
        "L": min(max(ligand / (ligand + 1.0), 0.0), 1.0),
        "EGFR": egfr_act,
        "RAS": float(state.get("RAS_GTP", 0.0)),
        "RAF": float(state.get("RAF_act", 0.0)),
        "MEK": float(state.get("MEK_p", 0.0)),
        "ERK": float(state.get("ERK_p", 0.0)),
        "PI3K": float(state.get("PI3K_act", 0.0)),
        "AKT": float(state.get("AKT_p", 0.0)),
    }

    # phenotypes: simple nonlinear combinations
    erk = out.get("ERK", 0.0)
    akt = out.get("AKT", 0.0)
    out["PROLIF"] = min(max((erk**1.2), 0.0), 1.0)
    out["SURV"] = min(max((akt**1.2), 0.0), 1.0)
    return out
