from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ampk.model import Inputs, ModelParams, ModelState, simulate
from ampk.network import build_pathway_graph


st.set_page_config(page_title="ampk", layout="wide")

st.title("ampk energy-sensing pathway")
st.caption("cellular energy status links metabolism, autophagy, and mtor regulation")

with st.sidebar:
    st.subheader("inputs")
    nutrient = st.slider("nutrient availability", 0.0, 1.0, 0.6, 0.01)
    demand = st.slider("energy demand", 0.0, 1.5, 0.5, 0.01)

    st.subheader("events")
    stress_pulse_on = st.checkbox("stress pulse", value=False)
    stress_pulse_t0 = st.number_input("stress t0", min_value=0.0, value=4.0, step=0.25)
    stress_pulse_dur = st.number_input("stress duration", min_value=0.0, value=1.0, step=0.25)
    stress_pulse_amp = st.slider("stress amplitude", 0.0, 1.5, 0.5, 0.01)

    nutrient_step_on = st.checkbox("nutrient step", value=False)
    nutrient_step_t0 = st.number_input("step t0", min_value=0.0, value=6.0, step=0.25)
    nutrient_step_to = st.slider("step to", 0.0, 1.0, 0.2, 0.01)

    st.subheader("simulation")
    t_end = st.number_input("t end", min_value=1.0, value=12.0, step=1.0)
    n_points = st.number_input("points", min_value=200, max_value=20000, value=2000, step=200)

    st.subheader("initial state")
    e0 = st.slider("energy (e0)", 0.0, 1.0, 0.65, 0.01)
    a0 = st.slider("ampk (a0)", 0.0, 1.0, 0.15, 0.01)
    m0 = st.slider("mtorc1 (m0)", 0.0, 1.0, 0.55, 0.01)
    u0 = st.slider("ulk1 (u0)", 0.0, 1.0, 0.20, 0.01)
    g0 = st.slider("autophagy (g0)", 0.0, 1.0, 0.15, 0.01)

    st.subheader("parameters")
    with st.expander("energy", expanded=False):
        k_prod = st.slider("k_prod", 0.0, 2.0, 0.55, 0.01)
        k_cons = st.slider("k_cons", 0.0, 2.0, 0.65, 0.01)
        e_set = st.slider("e_set", 0.0, 1.0, 0.65, 0.01)
        alpha_ampk_supply = st.slider("alpha_ampk_supply", 0.0, 2.0, 0.8, 0.01)
        alpha_mtor_demand = st.slider("alpha_mtor_demand", 0.0, 2.0, 0.6, 0.01)

    with st.expander("kinase logic", expanded=False):
        k_ampk = st.slider("k_ampk (energy stress)", 0.01, 1.0, 0.25, 0.01)
        beta_ampk_inhib_mtor = st.slider("beta_ampk_inhib_mtor", 0.0, 6.0, 2.5, 0.05)
        beta_mtor_inhib_ulk = st.slider("beta_mtor_inhib_ulk", 0.0, 6.0, 2.8, 0.05)

    run = st.button("run", type="primary")

inputs = Inputs(
    nutrient=nutrient,
    demand=demand,
    stress_pulse_on=stress_pulse_on,
    stress_pulse_t0=stress_pulse_t0,
    stress_pulse_dur=stress_pulse_dur,
    stress_pulse_amp=stress_pulse_amp,
    nutrient_step_on=nutrient_step_on,
    nutrient_step_t0=nutrient_step_t0,
    nutrient_step_to=nutrient_step_to,
)

params = ModelParams(
    t_end=float(t_end),
    n_points=int(n_points),
    k_prod=float(k_prod),
    k_cons=float(k_cons),
    e_set=float(e_set),
    alpha_ampk_supply=float(alpha_ampk_supply),
    alpha_mtor_demand=float(alpha_mtor_demand),
    k_ampk=float(k_ampk),
    beta_ampk_inhib_mtor=float(beta_ampk_inhib_mtor),
    beta_mtor_inhib_ulk=float(beta_mtor_inhib_ulk),
)

state0 = ModelState(e=e0, ampk=a0, mtorc1=m0, ulk1=u0, autophagy=g0)

if "last_df" not in st.session_state:
    st.session_state.last_df = None
    st.session_state.last_summary = None

if run or st.session_state.last_df is None:
    with st.spinner("simulating..."):
        df, summary = simulate(params=params, inputs=inputs, state0=state0)
    st.session_state.last_df = df
    st.session_state.last_summary = summary


df: pd.DataFrame = st.session_state.last_df
summary: dict = st.session_state.last_summary

c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric("mean energy", f"{summary['mean_energy']:.3f}")
c2.metric("min energy", f"{summary['min_energy']:.3f}")
c3.metric("mean ampk", f"{summary['mean_ampk']:.3f}")
c4.metric("mean mtorc1", f"{summary['mean_mtorc1']:.3f}")
c5.metric("mean autophagy", f"{summary['mean_autophagy']:.3f}")
c6.metric("autophagy auc", f"{summary['autophagy_auc']:.3f}")

st.divider()

left, right = st.columns([2, 1])

with left:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["t"], y=df["energy"], name="energy", mode="lines"))
    fig.add_trace(go.Scatter(x=df["t"], y=df["ampk"], name="ampk", mode="lines"))
    fig.add_trace(go.Scatter(x=df["t"], y=df["mtorc1"], name="mtorc1", mode="lines"))
    fig.add_trace(go.Scatter(x=df["t"], y=df["ulk1"], name="ulk1", mode="lines"))
    fig.add_trace(go.Scatter(x=df["t"], y=df["autophagy"], name="autophagy", mode="lines"))
    fig.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h"),
        xaxis_title="time",
        yaxis_title="activity / proxy",
    )
    st.plotly_chart(fig, use_container_width=True)

    fig_in = go.Figure()
    fig_in.add_trace(go.Scatter(x=df["t"], y=df["nutrient"], name="nutrient", mode="lines"))
    fig_in.add_trace(go.Scatter(x=df["t"], y=df["demand"], name="demand", mode="lines"))
    fig_in.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h"),
        xaxis_title="time",
        yaxis_title="input",
    )
    st.plotly_chart(fig_in, use_container_width=True)

with right:
    st.subheader("export")
    st.download_button(
        "download csv",
        df.to_csv(index=False).encode("utf-8"),
        file_name="ampk_sim.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.download_button(
        "download run config",
        json.dumps(
            {
                "inputs": inputs.__dict__,
                "params": params.__dict__,
                "state0": state0.__dict__,
            },
            indent=2,
            sort_keys=True,
        ).encode("utf-8"),
        file_name="ampk_run.json",
        mime="application/json",
        use_container_width=True,
    )

    st.subheader("pathway")
    g = build_pathway_graph()

    # simple circular-ish layout for stability
    pos = {
        "nutrients": (-1.0, 0.5),
        "energy": (-0.4, 0.2),
        "ampk": (0.2, 0.6),
        "mtorc1": (0.2, -0.2),
        "ulk1": (0.8, 0.3),
        "autophagy": (1.3, 0.3),
        "demand": (-1.0, -0.3),
    }

    edge_x = []
    edge_y = []
    edge_colors = []
    for src, dst, data in g.edges(data=True):
        x0, y0p = pos[src]
        x1, y1p = pos[dst]
        edge_x += [x0, x1, None]
        edge_y += [y0p, y1p, None]
        kind = data.get("kind", "activates")
        edge_colors.append("#999")

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=2, color="#999"),
        hoverinfo="none",
    )

    node_x = []
    node_y = []
    node_text = []
    for node in g.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        marker=dict(size=18, color="#666"),
        hoverinfo="text",
    )

    net_fig = go.Figure(data=[edge_trace, node_trace])
    net_fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    st.plotly_chart(net_fig, use_container_width=True)

st.divider()

with st.expander("data preview", expanded=False):
    st.dataframe(df.head(30), use_container_width=True)

with st.expander("notes", expanded=False):
    st.write(
        "this is a compact, phenomenological control model: ampk rises with low energy, inhibits mtorc1, "
        "activates ulk1, and increases autophagy. energy is a proxy state driven by nutrients and demand."
    )
