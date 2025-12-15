from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_timeseries(df: pd.DataFrame, *, series: list[str], title: str) -> go.Figure:
    fig = go.Figure()
    for s in series:
        fig.add_trace(go.Scatter(x=df["t"], y=df[s], mode="lines", name=s))

    fig.update_layout(
        title=title,
        xaxis_title="time",
        yaxis_title="activity / level",
        legend_title="state",
        height=420,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def plot_heatmap_sweep(
    sweep: pd.DataFrame,
    *,
    x: str,
    y: str,
    value: str,
    title: str,
) -> go.Figure:
    pivot = sweep.pivot(index=y, columns=x, values=value)

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorbar=dict(title=value),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=x,
        yaxis_title=y,
        height=500,
        margin=dict(l=60, r=20, t=50, b=50),
    )
    return fig


def _group_color(group: str) -> str:
    # use plotly's default qualitative palette indices (no hard-coded colors in theme sense)
    # (these are only hints; plotly will pick defaults if absent)
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    groups = {
        "input": 0,
        "proximal": 1,
        "plcg": 2,
        "second_messenger": 3,
        "pkc": 4,
        "pi3k": 5,
        "akt": 6,
        "mapk": 7,
        "nfkb": 8,
        "feedback": 9,
        "brake": 7,
    }
    return palette[groups.get(group, 0) % len(palette)]


def plot_network(g: nx.DiGraph) -> go.Figure:
    pos = nx.spring_layout(g, seed=7, k=0.9)

    edge_x: list[float] = []
    edge_y: list[float] = []
    edge_color: list[str] = []

    for u, v, attrs in g.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_color.append("rgba(120,120,120,0.55)")

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1.2, color="rgba(120,120,120,0.55)"),
        hoverinfo="none",
    )

    node_x: list[float] = []
    node_y: list[float] = []
    node_text: list[str] = []
    node_color: list[str] = []
    node_size: list[float] = []

    for n, data in g.nodes(data=True):
        x, y = pos[n]
        node_x.append(float(x))
        node_y.append(float(y))
        node_text.append(n)
        node_color.append(_group_color(data.get("group", "")))
        node_size.append(16.0 if data.get("group") not in {"input", "brake"} else 13.0)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hoverinfo="text",
        marker=dict(size=node_size, color=node_color, line=dict(width=1, color="rgba(50,50,50,0.7)")),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="bcr signaling influence graph",
        showlegend=False,
        height=560,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig


def summarize_outputs(df: pd.DataFrame) -> dict[str, float]:
    # simple summary metrics for sweeps
    out = {
        "nfkb_peak": float(df["nfkb_nuclear"].max()),
        "nfkb_auc": float(np.trapz(df["nfkb_nuclear"].values, df["t"].values)),
        "ikk_peak": float(df["ikk_active"].max()),
        "ca_peak": float(df["ca"].max()),
        "akt_peak": float(df["akt_active"].max()),
        "mapk_peak": float(df["mapk_active"].max()),
    }
    return out


def local_sensitivity(
    *,
    base_df: pd.DataFrame,
    perturbed_df: pd.DataFrame,
    param_name: str,
    base_value: float,
    perturbed_value: float,
    output_metric: str,
    metric_func,
) -> dict[str, float]:
    b = float(metric_func(base_df)[output_metric])
    p = float(metric_func(perturbed_df)[output_metric])

    # normalized sensitivity: d ln(output) / d ln(param)
    eps = 1e-12
    s = (math.log(p + eps) - math.log(b + eps)) / (math.log(perturbed_value + eps) - math.log(base_value + eps))
    return {"param": param_name, "sensitivity": float(s), "base": b, "perturbed": p}
