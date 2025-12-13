import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Allow running via `streamlit run app.py` even if the package isn't installed
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pi3k_akt_mtor.params import Inputs, ModelParams
from pi3k_akt_mtor.sim import simulate
from pi3k_akt_mtor.variants import Variant


st.set_page_config(page_title="PI3K/AKT/mTOR ODE", layout="wide")

st.title("PI3K / AKT / mTOR pathway — ODE simulator")

with st.expander("Pathway diagram", expanded=False):
        st.graphviz_chart(
                """
digraph G {
    rankdir=LR;
    node [shape=box, style=rounded];
    R [label="RTK/R"];
    IRS [label="IRS1 (func=1-Is)"];
    PI3K [label="PI3K/P"];
    PIP3 [label="PIP3"];
    AKT [label="AKT/Ap"];
    FOXO [label="FOXO/F"];
    TSC [label="TSC (act=1-Ti)"];
    M1 [label="mTORC1/M1"];
    S6K [label="S6K/S"];
    M2 [label="mTORC2/M2"];
    ERK [label="ERK/X"];
    AMPK [label="AMPK/K"];
    AUT [label="Autophagy/Au"];

    R -> IRS [label="+"];
    IRS -> PI3K [label="+"];
    R -> PI3K [label="+"];
    PI3K -> PIP3 [label="+"];
    PIP3 -> AKT [label="+"];
    M2 -> AKT [label="+"];
    AKT -> TSC [label="inhib"];
    ERK -> TSC [label="inhib"];
    TSC -> M1 [label="inhib gate"];
    M1 -> S6K [label="+"];
    S6K -> IRS [label="inhib (feedback)"];
    S6K -> M2 [label="inhib (feedback)"];
    AKT -> FOXO [label="inhib"];
    FOXO -> IRS [label="restore"];
    AMPK -> AUT [label="+"];
    M1 -> AUT [label="inhib"];
}
"""
        )

with st.sidebar:
    st.subheader("Variant")
    variant = st.selectbox(
        "Model variant",
        options=[v.value for v in Variant],
        index=0,
    )

    compare_variants = st.checkbox("Compare variants (overlay)", value=False)

    st.subheader("Inputs")
    ligand = st.slider("Ligand (RTK drive)", 0.0, 1.0, 1.0, 0.01)
    erk_input = st.slider("ERK input (crosstalk)", 0.0, 1.0, 0.0, 0.01)
    ampk_input = st.slider("AMPK input (crosstalk)", 0.0, 1.0, 0.0, 0.01)

    st.subheader("Drug multipliers")
    pi3k_activity = st.slider("PI3K activity", 0.0, 1.0, 1.0, 0.01)
    akt_activity = st.slider("AKT activity", 0.0, 1.0, 1.0, 0.01)
    mtorc1_activity = st.slider("mTORC1 activity", 0.0, 1.0, 1.0, 0.01)

    st.subheader("Simulation")
    t_end = st.number_input("t_end", min_value=1.0, max_value=10000.0, value=120.0, step=10.0)
    n_points = st.number_input("n_points", min_value=50, max_value=5000, value=600, step=50)

    st.subheader("Visualization")
    plot_series = st.multiselect(
        "Time-course series",
        options=[
            "R",
            "IRS1_func",
            "P",
            "PIP3",
            "Ap",
            "F",
            "TSC_act",
            "M1",
            "S",
            "M2",
            "X",
            "K",
            "Au",
            "Growth",
            "Survival",
            "Metabolism",
        ],
        default=["Ap", "M1", "S", "Growth", "Survival"],
    )

    with st.expander("Dose-response sweep", expanded=False):
        do_sweep = st.checkbox("Enable sweep", value=False)
        sweep_target = st.selectbox(
            "Target",
            options=["pi3k_activity", "akt_activity", "mtorc1_activity"],
            index=0,
        )
        sweep_min = st.slider("Min", 0.0, 1.0, 0.0, 0.01)
        sweep_max = st.slider("Max", 0.0, 1.0, 1.0, 0.01)
        sweep_n = st.number_input("Points", min_value=5, max_value=60, value=21, step=1)
        sweep_readout = st.selectbox(
            "Readout (final)",
            options=["Ap", "M1", "S", "Au", "Growth", "Survival", "Metabolism"],
            index=4,
        )

    st.subheader("Advanced")
    show_params = st.checkbox("Show/edit parameters (JSON)", value=False)

params = ModelParams()

params_json = json.dumps(params.as_dict(), indent=2, sort_keys=True)
if show_params:
    edited = st.text_area("ModelParams as JSON", value=params_json, height=350)
    try:
        params_dict = json.loads(edited)
        params = ModelParams(**{k: float(v) for k, v in params_dict.items()})
    except Exception as e:
        st.warning(f"Invalid params JSON: {e}")

inputs = Inputs(
    ligand=ligand,
    erk_input=erk_input,
    ampk_input=ampk_input,
    pi3k_activity=pi3k_activity,
    akt_activity=akt_activity,
    mtorc1_activity=mtorc1_activity,
)

try:
    if compare_variants:
        dfs = []
        meta = {"variants": []}
        for v in Variant:
            dfi, metai = simulate(
                variant=v,
                params=params,
                inputs=inputs,
                t_end=float(t_end),
                n_points=int(n_points),
                include_derived=True,
            )
            dfi["variant"] = v.value
            dfs.append(dfi)
            meta["variants"].append(metai)
        df = pd.concat(dfs, ignore_index=True)
    else:
        df, meta = simulate(
            variant=Variant(variant),
            params=params,
            inputs=inputs,
            t_end=float(t_end),
            n_points=int(n_points),
            include_derived=True,
        )
except Exception as e:
    st.error(str(e))
    st.stop()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Time courses")
    if not plot_series:
        st.info("Select at least one series.")
    else:
        id_vars = ["t"] + (["variant"] if "variant" in df.columns else [])
        present = [c for c in plot_series if c in df.columns]
        missing = sorted(set(plot_series) - set(present))
        if missing:
            st.warning(f"Not available: {missing}")

        long_df = df[id_vars + present].melt(id_vars=id_vars, var_name="series", value_name="value")
        if "variant" in long_df.columns:
            fig = px.line(long_df, x="t", y="value", color="series", line_dash="variant")
        else:
            fig = px.line(long_df, x="t", y="value", color="series")
        fig.update_layout(legend_title_text="Series")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Phase plot")
    phase_candidates = [c for c in ["Ap", "M1", "S", "Au", "Growth", "Survival", "Metabolism"] if c in df.columns]
    x_axis = st.selectbox("x", options=phase_candidates, index=0)
    y_axis = st.selectbox("y", options=phase_candidates, index=min(1, len(phase_candidates) - 1))
    if "variant" in df.columns:
        fig2 = px.line(df, x=x_axis, y=y_axis, color="variant")
    else:
        fig2 = px.line(df, x=x_axis, y=y_axis)
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.subheader("Final state")
    show_cols = [c for c in ["Ap", "M1", "S", "Au", "Growth", "Survival", "Metabolism"] if c in df.columns]
    if "variant" in df.columns:
        latest = df.sort_values("t").groupby("variant").tail(1).set_index("variant")
        st.dataframe(latest[show_cols], use_container_width=True)
    else:
        final = df.sort_values("t").tail(1).iloc[0]
        st.dataframe(final[show_cols].to_frame("value"), use_container_width=True)

    if do_sweep:
        st.subheader("Dose-response")
        lo, hi = float(min(sweep_min, sweep_max)), float(max(sweep_min, sweep_max))
        xs = np.linspace(lo, hi, int(sweep_n))

        rows = []
        for x in xs:
            sweep_inputs = replace(inputs, **{sweep_target: float(x)})
            dfi, _ = simulate(
                variant=Variant(variant),
                params=params,
                inputs=sweep_inputs,
                t_end=float(t_end),
                n_points=int(n_points),
                include_derived=True,
            )
            rows.append({"dose": float(x), "readout": float(dfi[sweep_readout].iloc[-1])})

        sweep_df = pd.DataFrame(rows)
        fig3 = px.line(sweep_df, x="dose", y="readout", markers=True)
        fig3.update_layout(xaxis_title=sweep_target, yaxis_title=f"Final {sweep_readout}")
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Export")
    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="pi3k_akt_mtor_timeseries.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download metadata JSON",
        data=json.dumps(meta, indent=2, sort_keys=True).encode("utf-8"),
        file_name="pi3k_akt_mtor_meta.json",
        mime="application/json",
    )

st.caption("All nodes are normalized activities in [0, 1]. Variants enable feedback and/or crosstalk.")
