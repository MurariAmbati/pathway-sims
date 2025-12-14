from __future__ import annotations

import json

import numpy as np
import streamlit as st

from rtk_egfr.analysis import dose_response, local_sensitivity
from rtk_egfr.params import PARAM_BOUNDS, Params, default_initial_state, default_params
from rtk_egfr.pathway import activity_from_state, build_graph
from rtk_egfr.sim import simulate, summary_metrics
from rtk_egfr.viz import (
    df_download_bytes,
    plot_dose_response,
    plot_pathway_graph,
    plot_sensitivity,
    plot_timeseries,
)


st.set_page_config(page_title="rtk_egfr", layout="wide")

st.title("rtk / egfr signaling")
st.caption("growth factor signaling via receptor tyrosine kinases")


def _sidebar_params() -> Params:
    base = default_params()
    st.sidebar.header("inputs")

    # core
    ligand = st.sidebar.slider("ligand dose", *PARAM_BOUNDS["ligand"], value=float(base.ligand))
    feedback = st.sidebar.slider(
        "erk→ras negative feedback",
        *PARAM_BOUNDS["feedback_strength"],
        value=float(base.feedback_strength),
    )

    # receptor kinetics
    with st.sidebar.expander("receptor kinetics", expanded=False):
        k_on = st.slider("k_on", *PARAM_BOUNDS["k_on"], value=float(base.k_on))
        k_off = st.slider("k_off", *PARAM_BOUNDS["k_off"], value=float(base.k_off))
        k_dim = st.slider("k_dim", *PARAM_BOUNDS["k_dim"], value=float(base.k_dim))
        k_undim = st.slider("k_undim", *PARAM_BOUNDS["k_undim"], value=float(base.k_undim))
        k_r2_deact = st.slider("k_r2_deact", *PARAM_BOUNDS["k_r2_deact"], value=float(base.k_r2_deact))

    with st.sidebar.expander("ras/erk cascade", expanded=False):
        k_ras_act = st.slider("k_ras_act", *PARAM_BOUNDS["k_ras_act"], value=float(base.k_ras_act))
        k_ras_gap = st.slider("k_ras_gap", *PARAM_BOUNDS["k_ras_gap"], value=float(base.k_ras_gap))
        ras_km = st.slider("ras_km", *PARAM_BOUNDS["ras_km"], value=float(base.ras_km))

        k_raf_act = st.slider("k_raf_act", *PARAM_BOUNDS["k_raf_act"], value=float(base.k_raf_act))
        k_raf_deact = st.slider("k_raf_deact", *PARAM_BOUNDS["k_raf_deact"], value=float(base.k_raf_deact))

        k_mek_phos = st.slider("k_mek_phos", *PARAM_BOUNDS["k_mek_phos"], value=float(base.k_mek_phos))
        k_mek_dephos = st.slider("k_mek_dephos", *PARAM_BOUNDS["k_mek_dephos"], value=float(base.k_mek_dephos))

        k_erk_phos = st.slider("k_erk_phos", *PARAM_BOUNDS["k_erk_phos"], value=float(base.k_erk_phos))
        k_erk_dephos = st.slider("k_erk_dephos", *PARAM_BOUNDS["k_erk_dephos"], value=float(base.k_erk_dephos))

    with st.sidebar.expander("pi3k/akt arm", expanded=False):
        k_pi3k_act = st.slider("k_pi3k_act", *PARAM_BOUNDS["k_pi3k_act"], value=float(base.k_pi3k_act))
        k_pi3k_deact = st.slider("k_pi3k_deact", *PARAM_BOUNDS["k_pi3k_deact"], value=float(base.k_pi3k_deact))
        pi3k_ras_crosstalk = st.slider(
            "pi3k_ras_crosstalk",
            *PARAM_BOUNDS["pi3k_ras_crosstalk"],
            value=float(base.pi3k_ras_crosstalk),
        )
        k_akt_phos = st.slider("k_akt_phos", *PARAM_BOUNDS["k_akt_phos"], value=float(base.k_akt_phos))
        k_akt_dephos = st.slider("k_akt_dephos", *PARAM_BOUNDS["k_akt_dephos"], value=float(base.k_akt_dephos))

    st.sidebar.header("simulation")
    t_end = st.sidebar.number_input("t_end", min_value=10.0, max_value=600.0, value=60.0, step=10.0)
    dt = st.sidebar.number_input("dt", min_value=0.02, max_value=5.0, value=0.2, step=0.02)

    st.session_state["t_end"] = float(t_end)
    st.session_state["dt"] = float(dt)

    return Params(
        ligand=float(ligand),
        feedback_strength=float(feedback),
        k_on=float(k_on),
        k_off=float(k_off),
        k_dim=float(k_dim),
        k_undim=float(k_undim),
        k_r2_deact=float(k_r2_deact),
        k_ras_act=float(k_ras_act),
        k_ras_gap=float(k_ras_gap),
        ras_km=float(ras_km),
        k_raf_act=float(k_raf_act),
        k_raf_deact=float(k_raf_deact),
        k_mek_phos=float(k_mek_phos),
        k_mek_dephos=float(k_mek_dephos),
        k_erk_phos=float(k_erk_phos),
        k_erk_dephos=float(k_erk_dephos),
        k_pi3k_act=float(k_pi3k_act),
        k_pi3k_deact=float(k_pi3k_deact),
        pi3k_ras_crosstalk=float(pi3k_ras_crosstalk),
        k_akt_phos=float(k_akt_phos),
        k_akt_dephos=float(k_akt_dephos),
    )


params = _sidebar_params()

t_end = float(st.session_state.get("t_end", 60.0))
dt = float(st.session_state.get("dt", 0.2))

# run base sim once per rerun
init = default_initial_state(params)
df = simulate(params=params, initial_state=init, t_end=t_end, dt=dt)
metrics = summary_metrics(df)

# tabs
pathway_tab, sim_tab, dose_tab, sens_tab, export_tab = st.tabs(
    ["pathway", "simulate", "dose-response", "sensitivity", "export"]
)

with pathway_tab:
    st.subheader("pathway")
    g = build_graph()
    state_last = df.iloc[-1].to_dict()
    act = activity_from_state(state_last, ligand=float(params.ligand))

    left, right = st.columns([1.2, 1])
    with left:
        st.plotly_chart(plot_pathway_graph(g, activity=act), use_container_width=True)
    with right:
        st.write("state (steady)")
        st.json({k: round(float(v), 4) for k, v in act.items()})
        st.write("metrics")
        st.json({k: round(float(v), 4) for k, v in metrics.items()})

with sim_tab:
    st.subheader("simulate")

    left, right = st.columns([1.4, 1])
    with left:
        st.plotly_chart(
            plot_timeseries(df, ["R2", "RAS_GTP", "ERK_p", "AKT_p"], "key signals"),
            use_container_width=True,
        )
        st.plotly_chart(
            plot_timeseries(df, ["prolif_index", "survival_index"], "phenotype indices"),
            use_container_width=True,
        )

    with right:
        st.write("summary")
        st.json({k: round(float(v), 4) for k, v in metrics.items()})
        st.write("raw timecourse")
        st.dataframe(df, use_container_width=True, height=420)

with dose_tab:
    st.subheader("dose-response")
    st.caption("sweep ligand dose; everything else fixed")

    metric = st.selectbox(
        "metric",
        options=[
            "ERK_p_peak",
            "ERK_p_steady",
            "AKT_p_peak",
            "AKT_p_steady",
            "prolif_index_peak",
            "prolif_index_steady",
            "survival_index_peak",
            "survival_index_steady",
        ],
        index=0,
    )

    d_min, d_max = st.slider("dose range", 0.0, 10.0, (0.0, 6.0))
    n = st.slider("points", 8, 60, 20)

    doses = np.linspace(d_min, d_max, int(n))
    dr = dose_response(doses=doses, params=params, t_end=t_end, dt=dt, metric=metric)

    st.plotly_chart(plot_dose_response(dr, "dose", metric, metric), use_container_width=True)
    st.dataframe(dr, use_container_width=True, height=360)

with sens_tab:
    st.subheader("sensitivity")
    st.caption("one-at-a-time local sensitivities around current parameters")

    metric = st.selectbox(
        "metric for sensitivity",
        options=[
            "ERK_p_peak",
            "ERK_p_steady",
            "AKT_p_peak",
            "AKT_p_steady",
            "prolif_index_peak",
            "prolif_index_steady",
            "survival_index_peak",
            "survival_index_steady",
        ],
        index=0,
        key="sens_metric",
    )
    rel_step = st.slider("relative step", 0.01, 0.5, 0.1)

    sens = local_sensitivity(params=params, t_end=t_end, dt=dt, metric=metric, rel_step=float(rel_step))
    st.plotly_chart(plot_sensitivity(sens), use_container_width=True)
    st.dataframe(sens, use_container_width=True, height=420)

with export_tab:
    st.subheader("export")

    st.download_button(
        "download timecourse csv",
        data=df_download_bytes(df),
        file_name="rtk_egfr_timecourse.csv",
        mime="text/csv",
    )

    payload = {
        "params": json.loads(json.dumps(params.__dict__)),
        "metrics": metrics,
    }
    st.download_button(
        "download params+metrics json",
        data=json.dumps(payload, indent=2).encode("utf-8"),
        file_name="rtk_egfr_run.json",
        mime="application/json",
    )
