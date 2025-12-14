from __future__ import annotations

import math
from typing import Iterable

import networkx as nx
import pandas as pd
import plotly.graph_objects as go


def plot_timeseries(df: pd.DataFrame, cols: list[str], title: str) -> go.Figure:
    fig = go.Figure()
    for c in cols:
        fig.add_trace(go.Scatter(x=df["t"], y=df[c], mode="lines", name=c))
    fig.update_layout(
        title=title,
        xaxis_title="time",
        yaxis_title="value",
        legend_title="state",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def plot_dose_response(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x], y=df[y], mode="lines+markers", name=y))
    fig.update_layout(
        title=title,
        xaxis_title=x,
        yaxis_title=y,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def plot_sensitivity(df: pd.DataFrame, title: str = "local sensitivity") -> go.Figure:
    fig = go.Figure(
        data=[
            go.Bar(
                x=df["sensitivity"],
                y=df["parameter"],
                orientation="h",
                name="sensitivity",
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="d log(metric) / d log(param)",
        yaxis_title="parameter",
        margin=dict(l=80, r=20, t=50, b=40),
        height=max(420, 18 * len(df) + 120),
    )
    return fig


def plot_pathway_graph(
    g: nx.DiGraph,
    activity: dict[str, float] | None = None,
    title: str = "egfr pathway",
) -> go.Figure:
    activity = activity or {}

    # layout
    pos = nx.spring_layout(g, seed=7)

    # edges
    edge_x: list[float] = []
    edge_y: list[float] = []
    edge_text: list[str] = []
    edge_color: list[str] = []

    for u, v, data in g.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        sign = data.get("sign", "+")
        edge_text.append(f"{u}→{v} ({sign})")
        edge_color.append("#666666" if sign == "+" else "#999999")

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=2, color="#888"),
        hoverinfo="none",
        name="edges",
    )

    # nodes
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []

    for n, data in g.nodes(data=True):
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)

        a = float(activity.get(n, 0.0))
        a = 0.0 if math.isnan(a) else max(0.0, min(1.0, a))

        label = data.get("label", n)
        node_text.append(f"{label}<br>activity={a:.2f}")
        node_size.append(18 + 26 * a)
        node_color.append(a)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[g.nodes[n].get("label", n) for n in g.nodes()],
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale="Viridis",
            reversescale=False,
            color=node_color,
            size=node_size,
            line_width=1,
            colorbar=dict(title="activity"),
        ),
        name="nodes",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=title,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=50, b=20),
        height=520,
    )
    return fig


def df_download_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")
