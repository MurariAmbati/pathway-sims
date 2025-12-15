from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def time_series_figure(df: pd.DataFrame, columns: list[str], title: str) -> go.Figure:
    fig = go.Figure()
    for col in columns:
        fig.add_trace(go.Scatter(x=df["t_s"], y=df[col], mode="lines", name=col))

    fig.update_layout(
        title=title,
        xaxis_title="time (s)",
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def phase_plane_figure(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Scatter(
                x=df[x],
                y=df[y],
                mode="lines",
                line=dict(width=2),
                name=f"{y} vs {x}",
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title=x,
        yaxis_title=y,
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def spectrum_figure(t_s: np.ndarray, x: np.ndarray, title: str) -> go.Figure:
    t_s = np.asarray(t_s, dtype=float)
    x = np.asarray(x, dtype=float)
    dt = float(np.median(np.diff(t_s)))
    y = x - np.mean(x)

    freqs = np.fft.rfftfreq(y.size, d=dt)
    spec = np.abs(np.fft.rfft(y))

    fig = go.Figure(data=[go.Scatter(x=freqs, y=spec, mode="lines")])
    fig.update_layout(
        title=title,
        xaxis_title="frequency (Hz)",
        yaxis_title="|FFT|",
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def heatmap_figure(z: np.ndarray, x: np.ndarray, y: np.ndarray, title: str, x_label: str, y_label: str) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=z,
                x=x,
                y=y,
                colorbar=dict(title="metric"),
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=520,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig
