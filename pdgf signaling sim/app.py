from __future__ import annotations

import json
from dataclasses import asdict

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.pdgf_sim.model import PDGFParams, STATE_NAMES, default_initial_state, dict_to_params, params_to_dict, preset_params, stoichiometry_summary
from src.pdgf_sim.sim import normalize_for_display, param_sweep_2d, simulate, to_csv_bytes


st.set_page_config(page_title="PDGF signaling (fibroblast)", layout="wide")

st.title("PDGF signaling simulation (fibroblast)")
st.caption("Mechanistic ODE model with downstream ERK/AKT activation and a proliferation proxy.")


@st.cache_data(show_spinner=False)
def _run_sim_cached(params_dict: dict, y0_list: list, t_end_min: float, n_points: int) -> pd.DataFrame:
    p = dict_to_params(params_dict)
    y0 = np.asarray(y0_list, dtype=float)
    df = simulate(p, y0=y0, t_end_min=t_end_min, n_points=n_points)
    return normalize_for_display(df)


@st.cache_data(show_spinner=False)
def _run_sweep_cached(
    params_dict: dict,
    y0_list: list,
    t_end_min: float,
    x_name: str,
    x_vals: list,
    y_name: str,
    y_vals: list,
    metric: str,
) -> pd.DataFrame:
    p = dict_to_params(params_dict)
    y0 = np.asarray(y0_list, dtype=float)
    return param_sweep_2d(
        base_params=p,
        y0=y0,
        t_end_min=t_end_min,
        sweep_x=(x_name, x_vals),
        sweep_y=(y_name, y_vals),
        metric=metric,
    )


with st.sidebar:
    st.header("Model")
    preset = st.selectbox("Parameter preset", ["Baseline", "Fast/Strong", "Slow/Weak", "High internalization"], index=0)

    if preset == "Baseline":
        params = preset_params("baseline")
    elif preset == "Fast/Strong":
        params = preset_params("fast")
    elif preset == "Slow/Weak":
        params = preset_params("slow")
    else:
        params = preset_params("internalization")

    st.subheader("Initial conditions")
    ligand_nM = st.number_input("PDGF ligand L0 (nM)", min_value=0.0, value=2.0, step=0.5)
    receptor_nM = st.number_input("Receptor R0 (nM)", min_value=0.0, value=10.0, step=1.0)

    st.subheader("Time")
    t_end_min = st.slider("Simulation duration (min)", min_value=30, max_value=720, value=240, step=30)
    n_points = st.slider("Sampling points", min_value=200, max_value=2000, value=800, step=100)

    st.subheader("Kinetics")
    with st.expander("Edit parameters", expanded=False):
        p = params_to_dict(params)
        cols = st.columns(2)
        for i, (k, v) in enumerate(p.items()):
            target = cols[i % 2]
            p[k] = target.number_input(k, value=float(v), min_value=0.0, step=float(max(1e-3, abs(v) * 0.05)))
        params = dict_to_params(p)

    st.divider()
    st.subheader("Sweep")
    metric = st.selectbox("Metric", ["P_final", "ERK_auc", "AKT_auc", "Dp_auc"], index=0)
    sweep_x_name = st.selectbox(
        "X parameter",
        ["k_on", "k_off", "k_dim", "k_undim", "k_phos", "k_dephos", "k_int", "k_erk_act", "k_erk_deact", "k_akt_act", "k_akt_deact"],
        index=4,
    )
    sweep_y_name = st.selectbox(
        "Y parameter",
        ["k_on", "k_off", "k_dim", "k_undim", "k_phos", "k_dephos", "k_int", "k_erk_act", "k_erk_deact", "k_akt_act", "k_akt_deact"],
        index=6,
    )

    grid_n = st.slider("Grid size", min_value=5, max_value=25, value=15, step=1)
    sweep_span = st.slider("Span (log10 decades)", min_value=0.1, max_value=2.0, value=1.0, step=0.1)


y0 = default_initial_state(ligand_nM=ligand_nM, receptor_nM=receptor_nM)
params_dict = params_to_dict(params)

st.info(stoichiometry_summary())

df = _run_sim_cached(params_dict=params_dict, y0_list=y0.tolist(), t_end_min=float(t_end_min), n_points=int(n_points))

colA, colB = st.columns([2, 1])

with colA:
    st.subheader("Time courses")
    tabs = st.tabs(["Ligand/Receptor", "Active receptor", "Downstream", "Proliferation"])

    with tabs[0]:
        fig = px.line(df, x="t_min", y=["L", "R", "C", "D"], labels={"value": "nM", "t_min": "Time (min)"})
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        fig = px.line(df, x="t_min", y=["Dp"], labels={"Dp": "Dp (nM)", "t_min": "Time (min)"})
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        fig = px.line(df, x="t_min", y=["ERK", "AKT"], labels={"value": "Activation (0..1)", "t_min": "Time (min)"})
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.scatter(df, x="ERK", y="AKT", color="t_min", color_continuous_scale="Viridis", labels={"t_min": "Time (min)"})
        st.plotly_chart(fig2, use_container_width=True)

    with tabs[3]:
        fig = px.line(df, x="t_min", y=["P"], labels={"P": "Proliferation proxy", "t_min": "Time (min)"})
        st.plotly_chart(fig, use_container_width=True)

with colB:
    st.subheader("Summary")

    final = df.iloc[-1]
    st.metric("Final Dp", f"{final['Dp']:.3f} nM")
    st.metric("ERK (final)", f"{final['ERK']:.3f}")
    st.metric("AKT (final)", f"{final['AKT']:.3f}")
    st.metric("P (final)", f"{final['P']:.3f}")

    st.divider()
    st.subheader("Downloads")
    st.download_button("Download simulation CSV", data=to_csv_bytes(df), file_name="pdgf_simulation.csv", mime="text/csv")
    st.download_button("Download parameters JSON", data=json.dumps(params_dict, indent=2).encode("utf-8"), file_name="pdgf_params.json", mime="application/json")

    st.divider()
    st.subheader("Network view")
    G = nx.DiGraph()
    G.add_edges_from(
        [
            ("PDGF (L)", "PDGFR (R)"),
            ("PDGFR (R)", "Bound complex (C)"),
            ("PDGF (L)", "Bound complex (C)"),
            ("Bound complex (C)", "Dimer (D)"),
            ("Dimer (D)", "Active receptor (Dp)"),
            ("Active receptor (Dp)", "ERK"),
            ("Active receptor (Dp)", "AKT"),
            ("ERK", "Proliferation (P)"),
            ("AKT", "Proliferation (P)"),
        ]
    )
    pos = nx.spring_layout(G, seed=7)
    edges_x, edges_y = [], []
    for u, v in G.edges():
        x0, y0p = pos[u]
        x1, y1p = pos[v]
        edges_x += [x0, x1, None]
        edges_y += [y0p, y1p, None]

    edge_trace = go.Scatter(x=edges_x, y=edges_y, mode="lines", line=dict(width=1), hoverinfo="none")
    node_x, node_y, node_text = [], [], []
    for n in G.nodes():
        x, y_ = pos[n]
        node_x.append(x)
        node_y.append(y_)
        node_text.append(n)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="bottom center",
        hoverinfo="text",
        marker=dict(size=10),
    )

    fig_net = go.Figure(data=[edge_trace, node_trace])
    fig_net.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
    st.plotly_chart(fig_net, use_container_width=True)


st.subheader("2D parameter sweep")

p_base = params_to_dict(params)

# Log-spaced sweep around the base value
x0 = max(1e-9, float(p_base[sweep_x_name]))
# Avoid identical axes (still works, but less useful)
y0b = max(1e-9, float(p_base[sweep_y_name]))

x_vals = np.logspace(np.log10(x0) - sweep_span / 2.0, np.log10(x0) + sweep_span / 2.0, int(grid_n)).tolist()
y_vals = np.logspace(np.log10(y0b) - sweep_span / 2.0, np.log10(y0b) + sweep_span / 2.0, int(grid_n)).tolist()

sweep_df = _run_sweep_cached(
    params_dict=p_base,
    y0_list=y0.tolist(),
    t_end_min=float(t_end_min),
    x_name=sweep_x_name,
    x_vals=x_vals,
    y_name=sweep_y_name,
    y_vals=y_vals,
    metric=metric,
)

heat = sweep_df.pivot(index=sweep_y_name, columns=sweep_x_name, values=metric)
fig_hm = px.imshow(
    heat,
    origin="lower",
    aspect="auto",
    labels=dict(x=sweep_x_name, y=sweep_y_name, color=metric),
)
fig_hm.update_layout(height=520)
st.plotly_chart(fig_hm, use_container_width=True)

st.download_button("Download sweep CSV", data=to_csv_bytes(sweep_df), file_name="pdgf_sweep.csv", mime="text/csv")
