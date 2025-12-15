from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from bcr_signaling.defaults import DEFAULTS, initial_conditions
from bcr_signaling.model import Interventions, network, simulate
from bcr_signaling.viz import (
    local_sensitivity,
    plot_heatmap_sweep,
    plot_network,
    plot_timeseries,
    summarize_outputs,
)


st.set_page_config(page_title="bcr signaling", layout="wide")

st.title("b cell receptor (bcr) signaling")
st.caption("ode simulation with nf-κb overlap (tcr-like proximal → canonical ikk/nf-κb)")


with st.sidebar:
    st.header("stimulus")
    antigen = st.slider("antigen", min_value=0.0, max_value=3.0, value=float(DEFAULTS.antigen), step=0.05)

    st.header("timing")
    t_end = st.slider("t_end", min_value=10.0, max_value=240.0, value=float(DEFAULTS.t_end), step=5.0)
    n_steps = st.slider("n_steps", min_value=200, max_value=3000, value=int(DEFAULTS.n_steps), step=50)

    st.header("interventions")
    syk_inhib = st.slider("syk inhibitor", 0.0, 1.0, 0.0, 0.05)
    btk_inhib = st.slider("btk inhibitor", 0.0, 1.0, 0.0, 0.05)
    pi3k_inhib = st.slider("pi3k inhibitor", 0.0, 1.0, 0.0, 0.05)
    ikk_inhib = st.slider("ikk inhibitor", 0.0, 1.0, 0.0, 0.05)

    shp1_kd = st.slider("shp1 knockdown", 0.0, 1.0, 0.0, 0.05)
    pten_kd = st.slider("pten knockdown", 0.0, 1.0, 0.0, 0.05)

    st.header("advanced parameters")
    with st.expander("rate constants", expanded=False):
        k_bcr_on = st.slider("k_bcr_on", 0.1, 8.0, float(DEFAULTS.k_bcr_on), 0.1)
        k_syk_act = st.slider("k_syk_act", 0.1, 8.0, float(DEFAULTS.k_syk_act), 0.1)
        k_plcg_act = st.slider("k_plcg_act", 0.1, 8.0, float(DEFAULTS.k_plcg_act), 0.1)
        k_ikk_act = st.slider("k_ikk_act", 0.1, 8.0, float(DEFAULTS.k_ikk_act), 0.1)
        k_a20_inhib = st.slider("k_a20_inhib", 0.0, 8.0, float(DEFAULTS.k_a20_inhib), 0.1)


params = {
    **DEFAULTS.__dict__,
    "antigen": antigen,
    "t_end": t_end,
    "n_steps": n_steps,
    "k_bcr_on": k_bcr_on,
    "k_syk_act": k_syk_act,
    "k_plcg_act": k_plcg_act,
    "k_ikk_act": k_ikk_act,
    "k_a20_inhib": k_a20_inhib,
}

y0 = initial_conditions()

interventions = Interventions(
    syk_inhib=syk_inhib,
    btk_inhib=btk_inhib,
    pi3k_inhib=pi3k_inhib,
    ikk_inhib=ikk_inhib,
    shp1_knockdown=shp1_kd,
    pten_knockdown=pten_kd,
)


tabs = st.tabs(["simulate", "network", "sweeps", "sensitivity", "about"])

with tabs[0]:
    st.subheader("timecourses")

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("**core readouts**")
        series = st.multiselect(
            "plot series",
            options=[
                "bcr_active",
                "syk_active",
                "plcg_active",
                "ca",
                "pkc_active",
                "pi3k_active",
                "akt_active",
                "mapk_active",
                "ikk_active",
                "ikb",
                "nfkb_nuclear",
                "a20",
            ],
            default=["syk_active", "pkc_active", "ikk_active", "nfkb_nuclear", "a20"],
        )

    with col_b:
        st.markdown("**computed summaries**")
        run = st.button("run simulation", type="primary")

    if run or "df" not in st.session_state:
        df = simulate(params=params, y0=y0, t_end=t_end, n_steps=n_steps, interventions=interventions)
        st.session_state["df"] = df
    else:
        df = st.session_state["df"]

    left, right = st.columns([1.4, 1])

    with left:
        fig = plot_timeseries(df, series=series, title="bcr signaling dynamics")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        metrics = summarize_outputs(df)
        st.dataframe(pd.DataFrame([metrics]).T.rename(columns={0: "value"}), use_container_width=True)

        st.download_button(
            "download csv",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="bcr_signaling_timeseries.csv",
            mime="text/csv",
        )

with tabs[1]:
    st.subheader("influence network")
    g = network()
    st.plotly_chart(plot_network(g), use_container_width=True)

with tabs[2]:
    st.subheader("parameter sweeps")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        x_param = st.selectbox("x parameter", ["antigen", "k_syk_act", "k_ikk_act", "k_a20_inhib"], index=0)
    with col2:
        y_param = st.selectbox("y parameter", ["k_plcg_act", "k_bcr_on", "k_syk_act", "k_ikk_act"], index=0)
    with col3:
        metric = st.selectbox("metric", ["nfkb_peak", "nfkb_auc", "ikk_peak", "ca_peak", "akt_peak", "mapk_peak"], index=0)

    st.caption("runs a 2d grid sweep and reports the selected summary metric")

    n_x = st.slider("x points", 4, 25, 12, 1)
    n_y = st.slider("y points", 4, 25, 12, 1)

    x_min, x_max = st.slider("x range", 0.1, 6.0, (0.5, 2.5), 0.1)
    y_min, y_max = st.slider("y range", 0.1, 6.0, (0.5, 2.5), 0.1)

    do_sweep = st.button("run sweep")

    if do_sweep:
        xs = [float(v) for v in np.linspace(x_min, x_max, n_x)]
        ys = [float(v) for v in np.linspace(y_min, y_max, n_y)]

        rows = []
        prog = st.progress(0)
        total = len(xs) * len(ys)
        for i, (xv, yv) in enumerate(itertools.product(xs, ys), start=1):
            p2 = dict(params)
            p2[x_param] = xv
            p2[y_param] = yv

            df2 = simulate(params=p2, y0=y0, t_end=t_end, n_steps=n_steps, interventions=interventions)
            out = summarize_outputs(df2)
            rows.append({x_param: xv, y_param: yv, **out})
            prog.progress(i / total)

        sweep = pd.DataFrame(rows)
        st.session_state["sweep"] = sweep

    sweep = st.session_state.get("sweep")
    if sweep is not None:
        fig = plot_heatmap_sweep(sweep, x=x_param, y=y_param, value=metric, title=f"sweep: {metric}")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(sweep.head(30), use_container_width=True)

with tabs[3]:
    st.subheader("local sensitivity")
    st.caption("estimates d ln(output) / d ln(param) via a small perturbation")

    base_df = st.session_state.get("df")
    if base_df is None:
        st.info("run a simulation first in the simulate tab")
    else:
        out_metric = st.selectbox("output metric", ["nfkb_peak", "nfkb_auc", "ikk_peak", "ca_peak"], index=0)
        param = st.selectbox(
            "parameter",
            ["antigen", "k_bcr_on", "k_syk_act", "k_plcg_act", "k_ikk_act", "k_a20_inhib"],
            index=0,
        )
        frac = st.slider("fractional increase", 0.01, 0.5, 0.1, 0.01)

        p0 = float(params[param])
        p1 = p0 * (1.0 + frac)

        p_pert = dict(params)
        p_pert[param] = p1

        df_pert = simulate(params=p_pert, y0=y0, t_end=t_end, n_steps=n_steps, interventions=interventions)
        s = local_sensitivity(
            base_df=base_df,
            perturbed_df=df_pert,
            param_name=param,
            base_value=p0,
            perturbed_value=p1,
            output_metric=out_metric,
            metric_func=summarize_outputs,
        )
        st.dataframe(pd.DataFrame([s]), use_container_width=True)

with tabs[4]:
    st.subheader("about")
    st.markdown(
        """
this app is a compact mechanistic simulator for bcr-driven signaling with explicit overlap to canonical ikk/nf-κb dynamics.

notes:
- the model is simplified and uses dimensionless activities (0–1-ish) for interpretability.
- it is intended for exploration and teaching, not quantitative prediction without re-fitting.
- interventions are implemented as multiplicative scalings of selected rate constants.
"""
    )
