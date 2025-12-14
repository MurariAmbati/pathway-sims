from __future__ import annotations

import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# allow `streamlit run streamlit_app.py` without an editable install
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from integrin_fak.model import Params, STATE_NAMES, network_spec
from integrin_fak.simulate import param_sweep_heatmap, phase_plane, simulate


st.set_page_config(page_title="integrin / fak signaling", layout="wide")

st.title("integrin / fak signaling")
st.caption("cell adhesion • migration • mechanotransduction • ecm coupling")

# --- sidebar controls ---
st.sidebar.header("inputs")

ecm = st.sidebar.slider("ecm", min_value=0.0, max_value=3.0, value=1.0, step=0.05)
force = st.sidebar.slider("force", min_value=0.0, max_value=1.5, value=0.5, step=0.05)

st.sidebar.header("kinetics")

k_on_i = st.sidebar.slider("k_on_i", 0.0, 5.0, 1.5, 0.05)
k_off_i = st.sidebar.slider("k_off_i", 0.0, 5.0, 0.6, 0.05)
k_force_i = st.sidebar.slider("k_force_i", 0.0, 5.0, 1.0, 0.05)

k_on_t = st.sidebar.slider("k_on_t", 0.0, 5.0, 1.4, 0.05)
k_off_t = st.sidebar.slider("k_off_t", 0.0, 5.0, 0.9, 0.05)
k_force_t = st.sidebar.slider("k_force_t", 0.0, 5.0, 0.6, 0.05)
k_talin_i = st.sidebar.slider("k_talin_i", 0.0, 5.0, 1.2, 0.05)

k_on_f = st.sidebar.slider("k_on_f", 0.0, 5.0, 2.0, 0.05)
k_off_f = st.sidebar.slider("k_off_f", 0.0, 5.0, 0.9, 0.05)

k_talin_f = st.sidebar.slider("k_talin_f", 0.0, 5.0, 0.9, 0.05)
k_src_f = st.sidebar.slider("k_src_f", 0.0, 5.0, 0.7, 0.05)

k_on_s = st.sidebar.slider("k_on_s", 0.0, 5.0, 1.8, 0.05)
k_off_s = st.sidebar.slider("k_off_s", 0.0, 5.0, 1.2, 0.05)

k_on_p = st.sidebar.slider("k_on_p", 0.0, 5.0, 2.0, 0.05)
k_off_p = st.sidebar.slider("k_off_p", 0.0, 5.0, 0.8, 0.05)

k_src_p = st.sidebar.slider("k_src_p", 0.0, 5.0, 0.9, 0.05)

k_on_e = st.sidebar.slider("k_on_e", 0.0, 5.0, 1.4, 0.05)
k_off_e = st.sidebar.slider("k_off_e", 0.0, 5.0, 1.1, 0.05)

k_on_r = st.sidebar.slider("k_on_r", 0.0, 5.0, 1.5, 0.05)
k_off_r = st.sidebar.slider("k_off_r", 0.0, 5.0, 1.0, 0.05)
k_force_r = st.sidebar.slider("k_force_r", 0.0, 5.0, 0.8, 0.05)

st.sidebar.header("simulation")
t_end = st.sidebar.slider("t_end", 5.0, 120.0, 30.0, 1.0)
n_points = st.sidebar.slider("n_points", 100, 2500, 600, 50)

animate = st.sidebar.checkbox("animate process", value=True)
animate_speed = st.sidebar.slider("animate speed", 0.0, 0.12, 0.03, 0.005)

params = Params(
    ecm=float(ecm),
    force=float(force),
    k_on_i=float(k_on_i),
    k_off_i=float(k_off_i),
    k_force_i=float(k_force_i),

    k_on_t=float(k_on_t),
    k_off_t=float(k_off_t),
    k_force_t=float(k_force_t),
    k_talin_i=float(k_talin_i),
    k_on_f=float(k_on_f),
    k_off_f=float(k_off_f),

    k_talin_f=float(k_talin_f),
    k_src_f=float(k_src_f),
    k_on_s=float(k_on_s),
    k_off_s=float(k_off_s),
    k_on_p=float(k_on_p),
    k_off_p=float(k_off_p),

    k_src_p=float(k_src_p),
    k_on_e=float(k_on_e),
    k_off_e=float(k_off_e),
    k_force_r=float(k_force_r),
    k_on_r=float(k_on_r),
    k_off_r=float(k_off_r),
)

run = st.sidebar.button("run", type="primary")

# store latest results
if "states" not in st.session_state:
    st.session_state["states"] = None
    st.session_state["flux"] = None
    st.session_state["params"] = None


def plot_time_series(states: pd.DataFrame) -> go.Figure:
    melt = states.melt(id_vars=["time"], value_vars=list(STATE_NAMES), var_name="state", value_name="value")
    fig = px.line(melt, x="time", y="value", color="state")
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), legend_title_text="")
    fig.update_yaxes(range=[0, 1.05])
    return fig


def plot_fluxes(flux: pd.DataFrame) -> go.Figure:
    cols = [c for c in flux.columns if c.endswith("_on") or c.endswith("_off")]
    melt = flux.melt(id_vars=["time"], value_vars=cols, var_name="flux", value_name="value")
    fig = px.line(melt, x="time", y="value", color="flux")
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), legend_title_text="")
    return fig


def plot_phase(states: pd.DataFrame, x: str, y: str) -> go.Figure:
    df = phase_plane(states, x=x, y=y)
    fig = px.scatter(df, x=x, y=y, color="time", color_continuous_scale="Viridis")
    fig.update_traces(mode="lines")
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    fig.update_xaxes(range=[0, 1.05])
    fig.update_yaxes(range=[0, 1.05])
    return fig


def plot_network_overlay(states: pd.DataFrame, flux: pd.DataFrame, step_idx: int) -> go.Figure:
    nodes, edges = network_spec()

    # fixed layout (simple left-to-right)
    pos = {
        "ecm": (0.0, 0.75),
        "force": (0.0, 0.25),
        "integrin": (0.25, 0.5),
        "talin": (0.43, 0.5),
        "fak": (0.60, 0.5),
        "src": (0.72, 0.65),
        "paxillin": (0.78, 0.35),
        "erk": (0.90, 0.70),
        "rhoa": (1.00, 0.35),
    }

    # flux thickness scaling
    fx_row = flux.iloc[int(step_idx)]
    fx_vals = []
    for _, _, edge_id in edges:
        fx_vals.append(float(fx_row.get(edge_id, 0.0)))
    max_fx = max(fx_vals) if fx_vals else 1.0
    max_fx = max(max_fx, 1e-9)

    fig = go.Figure()

    # edges
    for (a, b, edge_id), v in zip(edges, fx_vals):
        xa, ya = pos[a]
        xb, yb = pos[b]
        w = 1.0 + 8.0 * (v / max_fx)
        fig.add_trace(
            go.Scatter(
                x=[xa, xb],
                y=[ya, yb],
                mode="lines",
                line=dict(width=w, color="rgba(120,120,120,0.85)"),
                hoverinfo="text",
                text=[f"{edge_id}: {v:.3g}"] * 2,
                showlegend=False,
            )
        )

    # node values
    s = states.iloc[int(step_idx)]

    # normalize inputs to 0..1 for coloring (text shows true value)
    ecm_color = min(1.0, float(params.ecm) / 3.0) if float(params.ecm) > 0 else 0.0
    force_color = min(1.0, float(params.force) / 1.5) if float(params.force) > 0 else 0.0
    node_value = {
        "ecm": ecm_color,
        "force": force_color,
        "integrin": float(s["i"]),
        "talin": float(s["t"]),
        "fak": float(s["f"]),
        "src": float(s["s"]),
        "paxillin": float(s["p"]),
        "erk": float(s["e"]),
        "rhoa": float(s["r"]),
    }

    node_text = {
        "ecm": f"ecm<br>{float(params.ecm):.2f}",
        "force": f"force<br>{float(params.force):.2f}",
        "integrin": f"integrin<br>{float(s['i']):.2f}",
        "talin": f"talin<br>{float(s['t']):.2f}",
        "fak": f"fak<br>{float(s['f']):.2f}",
        "src": f"src<br>{float(s['s']):.2f}",
        "paxillin": f"paxillin<br>{float(s['p']):.2f}",
        "erk": f"erk<br>{float(s['e']):.2f}",
        "rhoa": f"rhoa<br>{float(s['r']):.2f}",
    }

    fig.add_trace(
        go.Scatter(
            x=[pos[n][0] for n in nodes],
            y=[pos[n][1] for n in nodes],
            mode="markers+text",
            marker=dict(size=26, color=[node_value[n] for n in nodes], cmin=0.0, cmax=1.0, colorscale="Viridis"),
            text=[node_text[n] for n in nodes],
            textposition="middle center",
            hoverinfo="text",
            showlegend=False,
        )
    )

    fig.update_layout(
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=360,
    )
    return fig


def plot_sweep(base: Params, t_end: float) -> go.Figure:
    ecm_vals = np.linspace(0.0, 3.0, 19)
    force_vals = np.linspace(0.0, 1.5, 16)
    df = param_sweep_heatmap(base, ecm_vals, force_vals, t_end=t_end, metric="f")
    piv = df.pivot(index="force", columns="ecm", values="value")
    fig = px.imshow(piv.values, x=piv.columns, y=piv.index, aspect="auto", origin="lower", color_continuous_scale="Viridis")
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    fig.update_xaxes(title="ecm")
    fig.update_yaxes(title="force")
    return fig


if run:
    with st.spinner("integrating odes..."):
        states, flux = simulate(params, t_end=float(t_end), n_points=int(n_points))
    st.session_state["states"] = states
    st.session_state["flux"] = flux
    st.session_state["params"] = asdict(params)

states = st.session_state["states"]
flux = st.session_state["flux"]

if states is None or flux is None:
    st.info("set parameters on the left, then hit run")
    st.stop()
    raise SystemExit(0)

# --- main layout ---
left, right = st.columns([1.1, 1.0], gap="large")

with left:
    st.subheader("time series")
    st.plotly_chart(plot_time_series(states), use_container_width=True)

    st.subheader("fluxes (process rates)")
    st.plotly_chart(plot_fluxes(flux), use_container_width=True)

    st.subheader("phase plane")
    c1, c2 = st.columns(2)
    with c1:
        x_state = st.selectbox("x", options=list(STATE_NAMES), index=0)
    with c2:
        y_state = st.selectbox("y", options=list(STATE_NAMES), index=1)
    st.plotly_chart(plot_phase(states, x_state, y_state), use_container_width=True)

with right:
    st.subheader("process view (live pathway)")

    step = st.slider("t index", 0, len(states) - 1, len(states) - 1)
    net_slot = st.empty()

    if animate:
        prog = st.progress(0, text="animating")
        for k in range(0, len(states), max(1, len(states) // 120)):
            net_slot.plotly_chart(plot_network_overlay(states, flux, k), use_container_width=True)
            prog.progress(int(100 * (k / (len(states) - 1))), text=f"time={states['time'].iloc[k]:.2f}")
            if animate_speed > 0:
                time.sleep(float(animate_speed))
        prog.progress(100, text="done")
    else:
        net_slot.plotly_chart(plot_network_overlay(states, flux, step), use_container_width=True)

    st.subheader("parameter sweep")
    st.plotly_chart(plot_sweep(params, t_end=float(t_end)), use_container_width=True)

st.divider()

with st.expander("data"):
    st.write("states")
    st.dataframe(states[["time"] + list(STATE_NAMES)].head(20), use_container_width=True)
    st.write("flux")
    fx_cols = ["time"] + [c for c in flux.columns if c.endswith("_on") or c.endswith("_off")]
    st.dataframe(flux[fx_cols].head(20), use_container_width=True)
