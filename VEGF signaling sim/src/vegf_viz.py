from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx


def to_dataframe(t: np.ndarray, series: Dict[str, np.ndarray]) -> pd.DataFrame:
    df = pd.DataFrame({"t": t})
    for k, v in series.items():
        df[k] = v
    return df


def timecourse_figure(df: pd.DataFrame, keys: Iterable[str], title: str) -> go.Figure:
    fig = go.Figure()
    for k in keys:
        fig.add_trace(go.Scatter(x=df["t"], y=df[k], mode="lines", name=k))
    fig.update_layout(
        title=title,
        xaxis_title="time",
        yaxis_title="normalized activity / level",
        legend_title="state",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def outcomes_summary(df: pd.DataFrame) -> pd.DataFrame:
    end = df.iloc[-1]
    return pd.DataFrame(
        {
            "metric": [
                "final pvegfr2",
                "final perk",
                "final pakt",
                "final psrc",
                "final no",
                "final permeability",
                "final angiogenesis",
                "auc permeability",
                "auc angiogenesis",
            ],
            "value": [
                float(end["pvegfr2"]),
                float(end["perk"]),
                float(end["pakt"]),
                float(end["psrc"]),
                float(end["no"]),
                float(end["permeability"]),
                float(end["angiogenesis"]),
                float(df["permeability"].mean() * (df["t"].iloc[-1] - df["t"].iloc[0])),
                float(df["angiogenesis"].mean() * (df["t"].iloc[-1] - df["t"].iloc[0])),
            ],
        }
    )


def network_figure(
    node_values: Dict[str, float],
    *,
    title: str = "signaling network (values at selected time)",
) -> go.Figure:
    g = nx.DiGraph()

    edges: List[Tuple[str, str, float]] = [
        ("vegf", "vegfr2", 1.0),
        ("vegf", "vegfr1", 0.8),
        ("nrp1", "vegfr2", 0.6),
        ("vegfr2", "perk", 1.0),
        ("vegfr2", "pakt", 1.0),
        ("vegfr2", "psrc", 1.0),
        ("pakt", "no", 1.0),
        ("psrc", "permeability", 1.0),
        ("no", "permeability", 0.6),
        ("perk", "angiogenesis", 0.8),
        ("pakt", "angiogenesis", 0.6),
    ]

    for u, v, w in edges:
        g.add_edge(u, v, weight=w)

    pos = nx.spring_layout(g, seed=7)

    def v(name: str) -> float:
        return float(node_values.get(name, 0.0))

    node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
    for n in g.nodes:
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{n}: {v(n):.3f}")
        node_size.append(18 + 30 * min(1.0, max(0.0, v(n))))
        node_color.append(v(n))

    edge_x, edge_y = [], []
    for u, vtx in g.edges:
        x0, y0 = pos[u]
        x1, y1 = pos[vtx]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1, color="rgba(120,120,120,0.6)"),
            hoverinfo="none",
            mode="lines",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=[n for n in g.nodes],
            textposition="bottom center",
            hovertext=node_text,
            hoverinfo="text",
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale="Viridis",
                cmin=0,
                cmax=1,
                line=dict(width=1, color="rgba(80,80,80,0.8)"),
            ),
            showlegend=False,
        )
    )

    fig.update_layout(
        title=title,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def sweep_heatmap(
    sweep_df: pd.DataFrame,
    x_key: str,
    y_key: str,
    z_key: str,
    title: str,
) -> go.Figure:
    pivot = sweep_df.pivot(index=y_key, columns=x_key, values=z_key)
    fig = px.imshow(
        pivot.values,
        x=pivot.columns.to_numpy(),
        y=pivot.index.to_numpy(),
        aspect="auto",
        labels=dict(x=x_key, y=y_key, color=z_key),
        title=title,
        origin="lower",
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return fig
