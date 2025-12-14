from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_timecourse(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Receptor activity",
            "SMAD phosphorylation",
            "Nuclear complex & Smad7",
            "Downstream programs",
        ),
    )

    t = df["t"]

    fig.add_trace(go.Scatter(x=t, y=df["R_star"], name="R*"), row=1, col=1)

    fig.add_trace(go.Scatter(x=t, y=df["pS23_c"], name="pS23_c"), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=df["pS23_n"], name="pS23_n"), row=1, col=2)

    fig.add_trace(go.Scatter(x=t, y=df["C_n"], name="C_n"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=df["Smad7"], name="Smad7"), row=2, col=1)

    fig.add_trace(go.Scatter(x=t, y=df["G_prog"], name="G_prog"), row=2, col=2)
    fig.add_trace(go.Scatter(x=t, y=df["E_prog"], name="E_prog"), row=2, col=2)
    fig.add_trace(go.Scatter(x=t, y=df["F_prog"], name="F_prog"), row=2, col=2)

    fig.update_layout(
        height=650,
        margin=dict(l=30, r=20, t=60, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title_text="time (min)")
    return fig


def plot_dose_response(dr: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Signal", "Programs"),
    )

    fig.add_trace(go.Scatter(x=dr["dose"], y=dr["C_n"], mode="lines+markers", name="C_n"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dr["dose"], y=dr["Smad7"], mode="lines+markers", name="Smad7"), row=1, col=1)

    fig.add_trace(go.Scatter(x=dr["dose"], y=dr["G_prog"], mode="lines+markers", name="G_prog"), row=1, col=2)
    fig.add_trace(go.Scatter(x=dr["dose"], y=dr["E_prog"], mode="lines+markers", name="E_prog"), row=1, col=2)
    fig.add_trace(go.Scatter(x=dr["dose"], y=dr["F_prog"], mode="lines+markers", name="F_prog"), row=1, col=2)

    fig.update_layout(height=420, margin=dict(l=30, r=20, t=60, b=30))
    fig.update_xaxes(title_text="TGF-β dose (a.u.)")
    return fig


def plot_crosstalk_heatmap(grid: pd.DataFrame, *, title: str) -> go.Figure:
    pivot = grid.pivot(index="mapk", columns="pi3k", values="value")
    fig = go.Figure(
        data=
        [
            go.Heatmap(
                z=pivot.values,
                x=pivot.columns.values,
                y=pivot.index.values,
                colorbar=dict(title="end value"),
            )
        ]
    )
    fig.update_layout(
        title=title,
        height=520,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(title_text="PI3K/AKT")
    fig.update_yaxes(title_text="MAPK")
    return fig
