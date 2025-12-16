from __future__ import annotations

from typing import Dict, Iterable

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def timeseries_figure(sim: Dict[str, np.ndarray], keys: Iterable[str]) -> go.Figure:
    df = pd.DataFrame({"t": sim["t"]})
    for k in keys:
        df[k] = sim[k]

    fig = go.Figure()
    for k in keys:
        fig.add_trace(go.Scatter(x=df["t"], y=df[k], mode="lines", name=k))

    fig.update_layout(
        height=440,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="time",
        yaxis_title="activity (0..1)",
        legend_title_text="state",
    )
    fig.update_yaxes(range=[0, 1])
    return fig


def phase_plane_figure(sim: Dict[str, np.ndarray], x_key: str, y_key: str) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Scatter(
                x=sim[x_key],
                y=sim[y_key],
                mode="lines",
                name=f"{y_key} vs {x_key}",
            )
        ]
    )
    fig.update_layout(
        height=440,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title=f"{x_key} (0..1)",
        yaxis_title=f"{y_key} (0..1)",
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return fig


def pathway_graph_figure(g: nx.DiGraph) -> go.Figure:
    # deterministic layout
    pos = nx.spring_layout(g, seed=7, k=0.9, iterations=150)

    # edges: draw activation/inhibition separately (solid vs dashed), then add arrow annotations.
    act_x: list[float | None] = []
    act_y: list[float | None] = []
    inh_x: list[float | None] = []
    inh_y: list[float | None] = []
    annotations = []

    node_shrink = 0.055  # shorten edges so arrows don’t sit on top of nodes

    for u, v, data in g.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        dx = x1 - x0
        dy = y1 - y0
        dist = float(np.hypot(dx, dy))
        if dist <= 1e-9:
            continue

        ux = dx / dist
        uy = dy / dist
        sx0 = x0 + node_shrink * ux
        sy0 = y0 + node_shrink * uy
        sx1 = x1 - node_shrink * ux
        sy1 = y1 - node_shrink * uy

        sign = int(data.get("sign", 1))
        if sign > 0:
            act_x += [sx0, sx1, None]
            act_y += [sy0, sy1, None]
            arrow_color = "rgba(0,0,0,0.55)"
        else:
            inh_x += [sx0, sx1, None]
            inh_y += [sy0, sy1, None]
            arrow_color = "rgba(0,0,0,0.35)"

        annotations.append(
            dict(
                x=sx1,
                y=sy1,
                ax=sx0,
                ay=sy0,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1.0,
                arrowwidth=1.4,
                arrowcolor=arrow_color,
                opacity=0.85,
            )
        )

    edge_act_trace = go.Scatter(
        x=act_x,
        y=act_y,
        line=dict(width=2.0, color="rgba(0,0,0,0.40)", dash="solid"),
        hoverinfo="none",
        mode="lines",
        name="activation",
    )

    edge_inh_trace = go.Scatter(
        x=inh_x,
        y=inh_y,
        line=dict(width=2.0, color="rgba(0,0,0,0.25)", dash="dash"),
        hoverinfo="none",
        mode="lines",
        name="inhibition",
    )

    # nodes
    node_x = []
    node_y = []
    node_text = []
    node_hover = []
    node_symbol = []

    kind_to_symbol = {
        "protein": "circle",
        "process": "square",
        "phenotype": "diamond",
    }

    for n, data in g.nodes(data=True):
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        node_text.append(data.get("label", n))
        desc = data.get("description", "")
        kind = data.get("kind", "")
        node_hover.append(f"<b>{data.get('label', n)}</b><br>{kind}<br>{desc}")
        node_symbol.append(kind_to_symbol.get(kind, "circle"))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hovertext=node_hover,
        hoverinfo="text",
        marker=dict(
            size=14,
            symbol=node_symbol,
            color="rgba(0,0,0,0.65)",
            line=dict(width=1.0, color="rgba(255,255,255,0.9)"),
        ),
        name="nodes",
    )

    fig = go.Figure(data=[edge_act_trace, edge_inh_trace, node_trace])
    fig.update_layout(
        height=600,
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
        annotations=annotations,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig
