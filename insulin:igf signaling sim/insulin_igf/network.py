from __future__ import annotations

from typing import Dict, Iterable, Tuple

import networkx as nx
import numpy as np
import plotly.graph_objects as go


NODES = [
    "Insulin",
    "IGF1",
    "IR",
    "IGF1R",
    "IRS",
    "PI3K",
    "PIP3",
    "AKT",
    "mTORC1",
    "S6K",
    "GLUT4",
    "Ras",
    "RAF",
    "MEK",
    "ERK",
    "Glucose uptake",
]


EDGES = [
    ("Insulin", "IR", "activates"),
    ("IGF1", "IGF1R", "activates"),
    ("IR", "IRS", "activates"),
    ("IGF1R", "IRS", "activates"),
    ("IRS", "PI3K", "activates"),
    ("PI3K", "PIP3", "activates"),
    ("PIP3", "AKT", "activates"),
    ("AKT", "mTORC1", "activates"),
    ("mTORC1", "S6K", "activates"),
    ("AKT", "GLUT4", "activates"),
    ("GLUT4", "Glucose uptake", "activates"),
    ("IR", "Ras", "activates"),
    ("IGF1R", "Ras", "activates"),
    ("IRS", "Ras", "activates"),
    ("Ras", "RAF", "activates"),
    ("RAF", "MEK", "activates"),
    ("MEK", "ERK", "activates"),
    ("S6K", "IRS", "inhibits"),
]


def _node_value(node: str, state: Dict[str, float]) -> float:
    # state contains *_act aliases and Glucose readouts
    mapping = {
        "IR": "IR_act",
        "IGF1R": "IGF1R_act",
        "IRS": "IRS_act",
        "PI3K": "PI3K_act",
        "PIP3": "PIP3_act",
        "AKT": "AKT_act",
        "mTORC1": "mTORC1_act",
        "S6K": "S6K_act",
        "GLUT4": "GLUT4_act",
        "Ras": "Ras_act",
        "RAF": "RAF_act",
        "MEK": "MEK_act",
        "ERK": "ERK_act",
    }

    if node in ("Insulin", "IGF1"):
        return 0.2
    if node == "Glucose uptake":
        return float(np.tanh(max(0.0, state.get("GlucoseUptake_mM_min", 0.0)) * 25.0))

    key = mapping.get(node)
    if key is None:
        return 0.0
    return float(state.get(key, 0.0))


def build_pathway_figure(state: Dict[str, float]) -> go.Figure:
    g = nx.DiGraph()
    for n in NODES:
        g.add_node(n)
    for a, b, kind in EDGES:
        g.add_edge(a, b, kind=kind)

    # fixed-ish layout for stability
    pos = {
        "Insulin": (-1.6, 0.6),
        "IGF1": (-1.6, -0.6),
        "IR": (-1.1, 0.6),
        "IGF1R": (-1.1, -0.6),
        "IRS": (-0.6, 0.0),
        "PI3K": (-0.1, 0.0),
        "PIP3": (0.35, 0.0),
        "AKT": (0.8, 0.0),
        "mTORC1": (1.25, 0.2),
        "S6K": (1.65, 0.2),
        "GLUT4": (1.25, -0.35),
        "Glucose uptake": (1.75, -0.35),
        "Ras": (0.0, -0.9),
        "RAF": (0.55, -0.9),
        "MEK": (1.1, -0.9),
        "ERK": (1.65, -0.9),
    }

    # edges
    edge_x = []
    edge_y = []
    edge_colors = []
    for a, b, d in g.edges(data=True):
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_colors.append("#888")

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.4, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    # nodes
    xs = []
    ys = []
    labels = []
    values = []

    for n in g.nodes():
        x, y = pos[n]
        xs.append(x)
        ys.append(y)
        labels.append(n)
        values.append(_node_value(n, state))

    node_trace = go.Scatter(
        x=xs,
        y=ys,
        mode="markers+text",
        text=labels,
        textposition="top center",
        hovertemplate="%{text}<br>value=%{marker.color:.2f}<extra></extra>",
        marker=dict(
            size=22,
            color=values,
            cmin=0.0,
            cmax=1.0,
            colorscale="Viridis",
            line=dict(width=1, color="rgba(30,30,30,0.5)"),
        ),
    )

    # annotate inhibitory edge
    annotations = []
    for a, b, d in g.edges(data=True):
        if d.get("kind") != "inhibits":
            continue
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        annotations.append(
            dict(
                x=(x0 + x1) / 2,
                y=(y0 + y1) / 2,
                text="⊣",
                showarrow=False,
                font=dict(size=18, color="crimson"),
            )
        )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=520,
        annotations=annotations,
    )

    return fig
