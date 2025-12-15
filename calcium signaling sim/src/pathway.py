from __future__ import annotations

from typing import Optional

import plotly.graph_objects as go


NODES = {
    "Stimulus": (0.0, 0.0),
    "Ca Influx": (1.7, 0.8),
    "PLC→IP3": (1.7, -0.8),
    "IP3": (3.2, -0.8),
    "IP3R (ER release)": (4.9, -0.2),
    "Ca²⁺ (cyt)": (3.2, 0.8),
    "SERCA": (4.9, 1.0),
    "Ca²⁺ (ER)": (6.6, 0.4),
    "PMCA/NCX": (4.9, 2.0),
    "CaM": (3.2, 2.0),
    "PKC": (4.9, 3.0),
}

EDGES = [
    ("Stimulus", "Ca Influx"),
    ("Stimulus", "PLC→IP3"),
    ("PLC→IP3", "IP3"),
    ("IP3", "IP3R (ER release)"),
    ("Ca²⁺ (ER)", "IP3R (ER release)"),
    ("IP3R (ER release)", "Ca²⁺ (cyt)"),
    ("Ca²⁺ (cyt)", "SERCA"),
    ("SERCA", "Ca²⁺ (ER)"),
    ("Ca²⁺ (cyt)", "PMCA/NCX"),
    ("Ca²⁺ (cyt)", "CaM"),
    ("CaM", "PKC"),
    ("Stimulus", "PKC"),
]


COMPONENT_ALIASES = {
    "Stimulus": {"Stimulus"},
    "Ca influx": {"Ca Influx"},
    "PLC/IP3": {"PLC→IP3", "IP3"},
    "IP3R": {"IP3R (ER release)"},
    "SERCA": {"SERCA"},
    "PMCA/NCX": {"PMCA/NCX"},
    "CaM": {"CaM"},
    "PKC": {"PKC"},
    "Ca cyt": {"Ca²⁺ (cyt)"},
    "Ca ER": {"Ca²⁺ (ER)"},
}


def pathway_figure(selected: Optional[str] = None) -> go.Figure:
    selected_nodes = set()
    if selected:
        selected_nodes = COMPONENT_ALIASES.get(selected, {selected})

    ex, ey = [], []
    for a, b in EDGES:
        x0, y0 = NODES[a]
        x1, y1 = NODES[b]
        ex += [x0, x1, None]
        ey += [y0, y1, None]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ex,
            y=ey,
            mode="lines",
            line=dict(width=2, color="rgba(120,120,120,0.7)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    xs, ys, labels, colors, sizes = [], [], [], [], []
    for name, (x, y) in NODES.items():
        xs.append(x)
        ys.append(y)
        labels.append(name)
        if name in selected_nodes:
            colors.append("rgba(30, 136, 229, 1.0)")
            sizes.append(26)
        else:
            colors.append("rgba(55, 71, 79, 0.95)")
            sizes.append(18)

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker=dict(size=sizes, color=colors, line=dict(width=1, color="white")),
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=420,
    )

    return fig
