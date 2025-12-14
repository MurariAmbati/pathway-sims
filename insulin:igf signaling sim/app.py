from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from insulin_igf.model import (
    DEFAULT_PARAMS,
    SimulationInputs,
    simulate,
    sweep_dose_response,
)
from insulin_igf.network import build_pathway_figure


st.set_page_config(page_title="Insulin / IGF signaling", layout="wide")

st.title("Insulin / IGF signaling")
st.caption(
    "Interactive insulin/IGF pathway simulator (PI3K/AKT + MAPK) with glucose homeostasis and pathway visualization."
)

with st.sidebar:
    st.header("Inputs")
    insulin_nM = st.slider("Insulin (nM)", 0.0, 50.0, 5.0, 0.5)
    igf_nM = st.slider("IGF-1 (nM)", 0.0, 50.0, 2.0, 0.5)

    st.subheader("Stimulation pattern")
    stim_mode = st.selectbox("Mode", ["step", "pulse", "meals"], index=0)
    t0 = st.number_input("Start time t0 (min)", value=10.0, min_value=0.0)
    duration = st.number_input("Duration (min)", value=30.0, min_value=0.0)
    period = st.number_input("Meal period (min)", value=120.0, min_value=10.0)

    st.header("Simulation")
    t_end = st.slider("Total time (min)", 30, 360, 180, 10)
    dt = st.slider("Output resolution (min)", 0.1, 2.0, 0.25, 0.05)

    st.header("Knobs")
    feedback_strength = st.slider("S6K → IRS feedback", 0.0, 4.0, 1.0, 0.1)
    pten_strength = st.slider("PTEN → PIP3", 0.0, 4.0, 1.0, 0.1)
    mapk_crosstalk = st.slider("IRS → MAPK coupling", 0.0, 2.0, 0.6, 0.05)

    st.header("Readout")
    highlight_time = st.slider("Highlight time (min)", 0, int(t_end), 60, 1)


inputs = SimulationInputs(
    insulin_nM=insulin_nM,
    igf_nM=igf_nM,
    stim_mode=stim_mode,
    t0=float(t0),
    duration=float(duration),
    period=float(period),
)

params = DEFAULT_PARAMS.model_copy()
params.feedback_strength = float(feedback_strength)
params.pten_strength = float(pten_strength)
params.mapk_crosstalk = float(mapk_crosstalk)


@st.cache_data(show_spinner=False)
def _run_sim(t_end: float, dt: float, params_dict: dict, inputs_dict: dict) -> pd.DataFrame:
    return simulate(t_end=t_end, dt=dt, params_dict=params_dict, inputs_dict=inputs_dict)


df = _run_sim(
    t_end=float(t_end),
    dt=float(dt),
    params_dict=params.model_dump(),
    inputs_dict=inputs.model_dump(),
)

if df.empty:
    st.error("Simulation produced no output.")
    st.stop()

# --- Layout ---
tabs = st.tabs(["Processes", "Pathway graph", "Time series", "Dose-response"])

with tabs[0]:
    st.subheader("Processes (rates / fluxes)")
    st.write(
        "These are computed from the ODE terms at the selected time to help you *see the process* (activation vs deactivation vs feedback)."
    )

    t_sel = float(highlight_time)
    df_sel = df.iloc[(df["t_min"] - t_sel).abs().argsort()[:1]].copy()
    row = df_sel.iloc[0].to_dict()

    col_a, col_b, col_c = st.columns([1.1, 1, 1])

    with col_a:
        st.metric("t (min)", f"{row['t_min']:.1f}")
        st.metric("Glucose (mM)", f"{row['Glucose_mM']:.2f}")
        st.metric("Uptake rate (mM/min)", f"{row['GlucoseUptake_mM_min']:.4f}")
        st.metric("Hepatic prod (mM/min)", f"{row['HepaticProd_mM_min']:.4f}")

    process_cols = [
        c
        for c in df.columns
        if c.endswith("_act_rate")
        or c.endswith("_deact_rate")
        or c.endswith("_feedback")
        or c.endswith("_pten")
    ]
    proc = df_sel[process_cols].T.reset_index()
    proc.columns = ["term", "value"]

    with col_b:
        st.write("Top +terms")
        st.dataframe(proc.sort_values("value", ascending=False).head(12), use_container_width=True)

    with col_c:
        st.write("Top -terms")
        st.dataframe(proc.sort_values("value", ascending=True).head(12), use_container_width=True)

with tabs[1]:
    st.subheader("Pathway graph")
    st.write("Node color reflects activation at the selected time.")

    t_sel = float(highlight_time)
    df_sel = df.iloc[(df["t_min"] - t_sel).abs().argsort()[:1]].copy()
    state = df_sel.iloc[0].to_dict()
    fig_net = build_pathway_figure(state)
    st.plotly_chart(fig_net, use_container_width=True)

with tabs[2]:
    st.subheader("Time series")

    left, right = st.columns([1.2, 1])

    with left:
        candidates = [
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
        ]
        selected = st.multiselect("Signals", candidates, default=["AKT", "ERK", "GLUT4", "PIP3"]) 

        cols = ["t_min"] + [f"{name}_act" for name in selected]
        plot_df = df[cols].copy()
        plot_df = plot_df.melt(id_vars=["t_min"], var_name="signal", value_name="activation")
        plot_df["signal"] = plot_df["signal"].str.replace("_act", "", regex=False)

        fig = px.line(plot_df, x="t_min", y="activation", color="signal")
        fig.add_vline(x=float(highlight_time), line_width=1, line_dash="dash")
        fig.update_yaxes(range=[-0.02, 1.02])
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig_g = go.Figure()
        fig_g.add_trace(go.Scatter(x=df["t_min"], y=df["Glucose_mM"], name="Glucose (mM)"))
        fig_g.add_vline(x=float(highlight_time), line_width=1, line_dash="dash")
        fig_g.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_g, use_container_width=True)

        fig_u = go.Figure()
        fig_u.add_trace(go.Scatter(x=df["t_min"], y=df["GlucoseUptake_mM_min"], name="Uptake rate"))
        fig_u.add_trace(go.Scatter(x=df["t_min"], y=df["HepaticProd_mM_min"], name="Hepatic prod"))
        fig_u.add_vline(x=float(highlight_time), line_width=1, line_dash="dash")
        fig_u.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_u, use_container_width=True)

with tabs[3]:
    st.subheader("Dose-response")
    st.write("Sweeps steady-state response vs insulin dose (optionally with IGF background).")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        dose_min = st.number_input("Dose min (nM)", value=0.0, min_value=0.0)
        dose_max = st.number_input("Dose max (nM)", value=50.0, min_value=0.1)
    with col2:
        n_points = st.slider("Points", 8, 80, 25, 1)
        settle_min = st.slider("Settle time (min)", 20, 240, 90, 5)
    with col3:
        readout = st.selectbox("Readout", ["AKT_act", "ERK_act", "GLUT4_act", "Glucose_mM"], index=0)

    doses = np.linspace(float(dose_min), float(dose_max), int(n_points))

    sweep_df = sweep_dose_response(
        doses_nM=doses,
        igf_background_nM=float(igf_nM),
        params_dict=params.model_dump(),
        settle_time_min=float(settle_min),
    )

    fig_dr = px.line(sweep_df, x="insulin_nM", y=readout)
    fig_dr.update_layout(height=380)
    st.plotly_chart(fig_dr, use_container_width=True)

st.divider()
with st.expander("Model parameters (advanced)"):
    st.json(params.model_dump())
