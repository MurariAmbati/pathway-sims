from __future__ import annotations

import dataclasses
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from src.vegf_model import VegfParams, initial_state, simulate
from src.vegf_viz import (
    network_figure,
    outcomes_summary,
    sweep_heatmap,
    timecourse_figure,
    to_dataframe,
)


st.set_page_config(page_title="vegf signaling", layout="wide")

st.title("vegf signaling: angiogenesis + vascular permeability")

with st.sidebar:
    st.header("simulation")
    vegf0 = st.slider("vegf-a input (normalized)", 0.0, 3.0, 1.0, 0.05)
    vegfr20 = st.slider("vegfr2 level (normalized)", 0.1, 3.0, 1.0, 0.05)
    vegfr10 = st.slider("vegfr1 level (decoy; normalized)", 0.0, 3.0, 0.6, 0.05)
    nrp10 = st.slider("nrp1 level (co-receptor; normalized)", 0.0, 3.0, 0.8, 0.05)
    t_end = st.slider("t end", 10, 240, 60, 5)
    n_steps = st.slider("time steps", 200, 2000, 600, 50)

    st.divider()
    st.header("parameters")

    base = VegfParams()
    editable: Dict[str, float] = {}

    cols = st.columns(2)
    fields = [f.name for f in dataclasses.fields(VegfParams)]
    for i, name in enumerate(fields):
        default = float(getattr(base, name))
        c = cols[i % 2]
        editable[name] = c.number_input(name, value=default, step=0.05, format="%.4f")

    st.divider()
    st.header("analysis")
    show_sweep = st.checkbox("run 2d sweep", value=False)


@st.cache_data(show_spinner=False)
def run_once(
    vegf0: float,
    vegfr20: float,
    vegfr10: float,
    nrp10: float,
    t_end: int,
    n_steps: int,
    params_dict: Dict[str, float],
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    params = VegfParams(**params_dict)
    x0 = initial_state(vegf=vegf0, vegfr2=vegfr20, vegfr1=vegfr10, nrp1=nrp10)
    t, series = simulate(t_end=float(t_end), n_steps=int(n_steps), x0=x0, params=params)
    df = to_dataframe(t, series)
    return df, series


df, series = run_once(vegf0, vegfr20, vegfr10, nrp10, t_end, n_steps, editable)

c1, c2 = st.columns([1.2, 0.8])

with c1:
    st.subheader("time courses")
    tab1, tab2, tab3 = st.tabs(["receptor", "downstream", "outputs"])

    with tab1:
        st.plotly_chart(
            timecourse_figure(
                df,
                keys=[
                    "vegf_free",
                    "vegf_bound",
                    "vegf_bound_r1",
                    "vegfr2_free",
                    "vegfr1_free",
                    "nrp1_free",
                    "pvegfr2",
                    "vegfr2_internal",
                ],
                title="ligand/receptor dynamics",
            ),
            use_container_width=True,
        )

    with tab2:
        st.plotly_chart(
            timecourse_figure(
                df,
                keys=["perk", "pakt", "psrc", "no"],
                title="downstream modules",
            ),
            use_container_width=True,
        )

    with tab3:
        st.plotly_chart(
            timecourse_figure(
                df,
                keys=["permeability", "angiogenesis"],
                title="phenotype proxies",
            ),
            use_container_width=True,
        )

with c2:
    st.subheader("readouts")
    st.dataframe(outcomes_summary(df), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("network view")

    # map internal variable names to a compact signaling graph
    node_vals = {
        "vegf": float(df["vegf_free"].iloc[-1]),
        "vegfr2": float(df["pvegfr2"].iloc[-1]),
        "vegfr1": float(df["vegfr1_free"].iloc[-1]),
        "nrp1": float(df["nrp1_free"].iloc[-1]),
        "perk": float(df["perk"].iloc[-1]),
        "pakt": float(df["pakt"].iloc[-1]),
        "psrc": float(df["psrc"].iloc[-1]),
        "no": float(df["no"].iloc[-1]),
        "permeability": float(df["permeability"].iloc[-1]),
        "angiogenesis": float(df["angiogenesis"].iloc[-1]),
    }

    st.plotly_chart(network_figure(node_vals), use_container_width=True)

    st.divider()
    st.subheader("export")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("download csv", data=csv, file_name="vegf_sim.csv", mime="text/csv")


if show_sweep:
    st.subheader("2d sweep")
    st.caption("sweeps vegf input vs vegfr2 level; reports final outputs")

    sx = st.slider("vegf sweep max", 0.5, 4.0, 2.5, 0.1)
    sy = st.slider("vegfr2 sweep max", 0.5, 4.0, 2.5, 0.1)
    n = st.slider("grid size", 10, 60, 25, 5)

    x_vals = np.linspace(0.0, float(sx), int(n))
    y_vals = np.linspace(0.2, float(sy), int(n))

    @st.cache_data(show_spinner=True)
    def run_sweep(
        x_vals: np.ndarray,
        y_vals: np.ndarray,
        vegfr10: float,
        nrp10: float,
        t_end: int,
        n_steps: int,
        params_dict: Dict[str, float],
    ) -> pd.DataFrame:
        rows = []
        for xv in x_vals:
            for yv in y_vals:
                params = VegfParams(**params_dict)
                x0 = initial_state(vegf=float(xv), vegfr2=float(yv), vegfr1=float(vegfr10), nrp1=float(nrp10))
                t, series = simulate(t_end=float(t_end), n_steps=int(n_steps), x0=x0, params=params)
                df_local = to_dataframe(t, series)
                end = df_local.iloc[-1]
                rows.append(
                    {
                        "vegf0": float(xv),
                        "vegfr20": float(yv),
                        "final_permeability": float(end["permeability"]),
                        "final_angiogenesis": float(end["angiogenesis"]),
                        "final_pvegfr2": float(end["pvegfr2"]),
                        "final_perk": float(end["perk"]),
                        "final_pakt": float(end["pakt"]),
                        "final_psrc": float(end["psrc"]),
                    }
                )
        return pd.DataFrame(rows)

    sweep_df = run_sweep(x_vals, y_vals, vegfr10, nrp10, t_end, max(300, n_steps // 2), editable)

    t1, t2 = st.tabs(["permeability", "angiogenesis"])
    with t1:
        st.plotly_chart(
            sweep_heatmap(
                sweep_df,
                x_key="vegf0",
                y_key="vegfr20",
                z_key="final_permeability",
                title="final permeability across (vegf, vegfr2)",
            ),
            use_container_width=True,
        )

    with t2:
        st.plotly_chart(
            sweep_heatmap(
                sweep_df,
                x_key="vegf0",
                y_key="vegfr20",
                z_key="final_angiogenesis",
                title="final angiogenesis across (vegf, vegfr2)",
            ),
            use_container_width=True,
        )
