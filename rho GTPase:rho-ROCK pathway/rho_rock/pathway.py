from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx


@dataclass(frozen=True)
class NodeSpec:
    key: str
    label: str
    kind: str  # protein | process | phenotype
    description: str


@dataclass(frozen=True)
class EdgeSpec:
    src: str
    dst: str
    sign: int  # +1 activation, -1 inhibition
    mechanism: str


def pathway_specs() -> Tuple[Dict[str, NodeSpec], List[EdgeSpec]]:
    """Defines a compact but mechanistically rich rhoa/rock actin module.

    This is a didactic network for visualization + coarse-grained dynamics.
    """

    nodes = {
        # gtpase module
        "rhoa": NodeSpec(
            key="rhoa",
            label="rhoA (gtp)",
            kind="protein",
            description="active rhoa gtpase; upstream of rock and mdia",
        ),
        "gef": NodeSpec(
            key="gef",
            label="rhoGEF",
            kind="protein",
            description="generic rhoa guanine nucleotide exchange factor",
        ),
        "gap": NodeSpec(
            key="gap",
            label="rhoGAP",
            kind="protein",
            description="generic rhoa gtpase-activating protein",
        ),
        # effectors
        "rock": NodeSpec(
            key="rock",
            label="ROCK",
            kind="protein",
            description="rho-associated kinase; promotes actomyosin contractility",
        ),
        "mdia": NodeSpec(
            key="mdia",
            label="mDia",
            kind="protein",
            description="formin effector of rhoa; drives actin polymerization",
        ),
        # actin remodeling branch
        "limk": NodeSpec(
            key="limk",
            label="LIMK",
            kind="protein",
            description="lim kinase; phosphorylates cofilin downstream of rock",
        ),
        "cofilin": NodeSpec(
            key="cofilin",
            label="cofilin",
            kind="protein",
            description="actin severing factor; inactivated by phosphorylation",
        ),
        "factin": NodeSpec(
            key="factin",
            label="F-actin",
            kind="process",
            description="polymerized actin; stress fibers / cortical actin proxy",
        ),
        # myosin/contractility branch
        "mlc": NodeSpec(
            key="mlc",
            label="pMLC",
            kind="protein",
            description="phosphorylated myosin light chain; myosin ii activation proxy",
        ),
        "mlcp": NodeSpec(
            key="mlcp",
            label="MLCP",
            kind="protein",
            description="myosin light chain phosphatase; antagonizes mlc phosphorylation",
        ),
        "myosin": NodeSpec(
            key="myosin",
            label="myosin II",
            kind="protein",
            description="motor generating contractile force when activated (via pmlc)",
        ),
        "contractility": NodeSpec(
            key="contractility",
            label="contractility",
            kind="phenotype",
            description="cell-scale force output proxy (actomyosin tension)",
        ),
        # phenotypes
        "migration": NodeSpec(
            key="migration",
            label="migration",
            kind="phenotype",
            description="motility proxy; non-monotonic with contractility",
        ),
        "shape": NodeSpec(
            key="shape",
            label="cell shape",
            kind="phenotype",
            description="morphology proxy; increased stress fibers -> more contractile shape",
        ),
    }

    edges = [
        EdgeSpec("gef", "rhoa", +1, "nucleotide exchange"),
        EdgeSpec("gap", "rhoa", -1, "gtp hydrolysis"),
        EdgeSpec("rhoa", "rock", +1, "effector activation"),
        EdgeSpec("rhoa", "mdia", +1, "effector activation"),
        EdgeSpec("rock", "limk", +1, "phosphorylation"),
        EdgeSpec("limk", "cofilin", -1, "inhibitory phosphorylation"),
        EdgeSpec("cofilin", "factin", -1, "filament severing"),
        EdgeSpec("mdia", "factin", +1, "formin-mediated polymerization"),
        EdgeSpec("rock", "mlcp", -1, "inhibition (e.g., mypt1 phosphorylation)"),
        EdgeSpec("mlcp", "mlc", -1, "dephosphorylation"),
        EdgeSpec("rock", "mlc", +1, "direct/indirect phosphorylation"),
        EdgeSpec("mlc", "myosin", +1, "activates motor"),
        EdgeSpec("myosin", "contractility", +1, "force generation"),
        EdgeSpec("factin", "contractility", +1, "scaffold for tension"),
        EdgeSpec("contractility", "shape", +1, "stress fibers / tension"),
        # migration is typically non-monotonic; we represent the pro-migratory effect of moderate actin + tension
        EdgeSpec("factin", "migration", +1, "protrusion support"),
        EdgeSpec("contractility", "migration", -1, "excess tension can reduce motility"),
    ]

    return nodes, edges


def build_pathway_graph() -> nx.DiGraph:
    nodes, edges = pathway_specs()
    g = nx.DiGraph()

    for key, spec in nodes.items():
        g.add_node(
            key,
            label=spec.label,
            kind=spec.kind,
            description=spec.description,
        )

    for e in edges:
        g.add_edge(
            e.src,
            e.dst,
            sign=e.sign,
            mechanism=e.mechanism,
            label=("+" if e.sign > 0 else "-") + " " + e.mechanism,
        )

    return g
