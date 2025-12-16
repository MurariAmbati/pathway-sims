from __future__ import annotations

from dataclasses import dataclass

import networkx as nx


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    kind: str  # "activates" | "inhibits" | "input"


def build_pathway_graph() -> nx.DiGraph:
    g = nx.DiGraph()

    nodes = [
        ("nutrients", {"group": "input"}),
        ("energy", {"group": "state"}),
        ("ampk", {"group": "kinase"}),
        ("mtorc1", {"group": "complex"}),
        ("ulk1", {"group": "kinase"}),
        ("autophagy", {"group": "process"}),
        ("demand", {"group": "input"}),
    ]
    g.add_nodes_from(nodes)

    edges = [
        Edge("demand", "energy", "inhibits"),
        Edge("nutrients", "energy", "activates"),
        Edge("energy", "ampk", "inhibits"),  # low energy activates ampk (shown as energy inhibits)
        Edge("nutrients", "mtorc1", "activates"),
        Edge("energy", "mtorc1", "activates"),
        Edge("ampk", "mtorc1", "inhibits"),
        Edge("ampk", "ulk1", "activates"),
        Edge("mtorc1", "ulk1", "inhibits"),
        Edge("ulk1", "autophagy", "activates"),
        Edge("ampk", "energy", "activates"),
        Edge("mtorc1", "energy", "inhibits"),
    ]

    for e in edges:
        g.add_edge(e.src, e.dst, kind=e.kind)

    return g
