from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.metrics import metrics_table
from src.model import ModelParams, StimulusParams, simulate
from src.pathway import pathway_figure
from src.viz import heatmap_figure, phase_plane_figure, spectrum_figure, time_series_figure


st.set_page_config(page_title="Calcium (Ca²⁺) signaling", layout="wide")

st.title("Calcium (Ca²⁺) signaling")
st.caption(
    "Fast second‑messenger signaling (synaptic activity, contraction, secretion), often via Ca²⁺/calmodulin and PKC. "
    "Interactive simulator with pathway diagram, visualizations, and quantitative metrics."
)

with st.sidebar:
    st.header("Simulation")
    t_end_s = st.slider("Duration (s)", min_value=10.0, max_value=200.0, value=60.0, step=5.0)
    dt_s = st.slider("Time step (s)", min_value=0.001, max_value=0.2, value=0.01, step=0.001)
    solver = st.selectbox("ODE solver", ["LSODA", "RK45"], index=0)

    st.divider()
    st.header("Stimulus")
    stim_type = st.selectbox("Type", ["Step", "Pulse train", "Sine"], index=1)
    stim_start = st.number_input("Start (s)", min_value=0.0, value=5.0, step=0.5)
    stim_end = st.number_input("End (s)", min_value=0.0, value=40.0, step=0.5)
    stim_amp = st.slider("Amplitude (a.u.)", min_value=0.0, max_value=3.0, value=1.0, step=0.05)

    period_s = st.slider("Period (s) (pulse train)", min_value=0.2, max_value=20.0, value=2.0, step=0.1)
    duty = st.slider("Duty (0..1) (pulse train)", min_value=0.01, max_value=0.99, value=0.2, step=0.01)

    sine_hz = st.slider("Frequency (Hz) (sine)", min_value=0.01, max_value=5.0, value=0.5, step=0.01)
    sine_base = st.slider("Baseline (a.u.) (sine)", min_value=0.0, max_value=2.0, value=0.0, step=0.05)

    st.divider()
    st.header("Pathway focus")
    focus = st.selectbox(
        "Highlight component",
        [
            "None",
            "Stimulus",
            "Ca influx",
            "PLC/IP3",
            "IP3R",
            "SERCA",
            "PMCA/NCX",
            "CaM",
            "PKC",
            "Ca cyt",
            "Ca ER",
        ],
        index=0,
    )

    st.divider()
    st.header("Model parameters")
    with st.expander("Initial conditions"):
        ca_cyt0 = st.number_input("Ca²⁺ cytosol (uM)", min_value=0.0, value=0.10, step=0.01, format="%.3f")
        ca_er0 = st.number_input("Ca²⁺ ER (uM)", min_value=0.0, value=200.0, step=5.0)
        ip30 = st.number_input("IP3 (uM)", min_value=0.0, value=0.05, step=0.01, format="%.3f")

    with st.expander("Fluxes"):
        v_in_max = st.slider("Max Ca influx v_in (uM/s)", 0.0, 5.0, 0.8, 0.05)
        v_pmca_max = st.slider("PMCA/NCX v_pmca (uM/s)", 0.0, 5.0, 1.0, 0.05)
        k_pmca = st.slider("PMCA K (uM)", 0.01, 2.0, 0.25, 0.01)

        v_serca_max = st.slider("SERCA v_serca (uM/s)", 0.0, 5.0, 1.2, 0.05)
        k_serca = st.slider("SERCA K (uM)", 0.01, 2.0, 0.2, 0.01)

        v_ip3r_max = st.slider("IP3R v_ip3r (uM/s)", 0.0, 30.0, 8.0, 0.25)
        k_leak = st.slider("ER leak k_leak (1/s)", 0.0, 0.2, 0.02, 0.002)
        vol_cyt_to_er = st.slider("V_cyt/V_er", 1.0, 20.0, 5.0, 0.5)

    with st.expander("IP3"):
        v_plc_max = st.slider("PLC v_plc (uM/s)", 0.0, 2.0, 0.25, 0.01)
        k_ip3_deg = st.slider("IP3 degradation k_deg (1/s)", 0.0, 2.0, 0.15, 0.01)
        k_ip3 = st.slider("IP3R K_ip3 (uM)", 0.01, 3.0, 0.2, 0.01)

    with st.expander("CaM / PKC"):
        k_cam_on = st.slider("CaM on (1/s·uM^n)", 0.0, 20.0, 4.0, 0.1)
        k_cam_off = st.slider("CaM off (1/s)", 0.0, 10.0, 1.5, 0.1)
        n_cam = st.slider("CaM Hill n", 1.0, 4.0, 2.0, 0.1)

        k_pkc_on = st.slider("PKC on (1/s)", 0.0, 20.0, 2.5, 0.1)
        k_pkc_off = st.slider("PKC off (1/s)", 0.0, 10.0, 0.7, 0.1)
        k_pkc_ca = st.slider("PKC K_Ca (uM)", 0.01, 3.0, 0.3, 0.01)
        k_pkc_dag = st.slider("PKC K_DAG (a.u.)", 0.01, 3.0, 0.25, 0.01)


stim_params = StimulusParams(
    stim_type=stim_type, start_s=float(stim_start), end_s=float(stim_end), amplitude=float(stim_amp),
    period_s=float(period_s), duty=float(duty),
    frequency_hz=float(sine_hz), baseline=float(sine_base),
)

model_params = ModelParams(
    ca_cyt0_uM=float(ca_cyt0),
    ca_er0_uM=float(ca_er0),
    ip3_0_uM=float(ip30),
    v_in_max=float(v_in_max),
    v_pmca_max=float(v_pmca_max),
    k_pmca_uM=float(k_pmca),
    v_serca_max=float(v_serca_max),
    k_serca_uM=float(k_serca),
    v_ip3r_max=float(v_ip3r_max),
    k_leak_per_s=float(k_leak),
    vol_cyt_to_er=float(vol_cyt_to_er),
    v_plc_max=float(v_plc_max),
    k_ip3_deg_per_s=float(k_ip3_deg),
    k_ip3_uM=float(k_ip3),
    k_cam_on_per_s=float(k_cam_on),
    k_cam_off_per_s=float(k_cam_off),
    n_cam=float(n_cam),
    k_pkc_on_per_s=float(k_pkc_on),
    k_pkc_off_per_s=float(k_pkc_off),
    k_pkc_ca_uM=float(k_pkc_ca),
    k_pkc_dag=float(k_pkc_dag),
)


@st.cache_data(show_spinner=False)
def _run(df_key: tuple) -> pd.DataFrame:
    mp, sp, t_end_s, dt_s, solver = df_key
    return simulate(model=mp, stim=sp, t_end_s=t_end_s, dt_s=dt_s, solver=solver)


run_key = (model_params, stim_params, float(t_end_s), float(dt_s), solver)

with st.spinner("Simulating…"):
    df = _run(run_key)


tabs = st.tabs(["Pathway", "Time series", "Phase & spectrum", "Metrics", "Parameter sweep"])

with tabs[0]:
    selected = None if focus == "None" else focus
    st.subheader("Pathway")
    st.plotly_chart(pathway_figure(selected), use_container_width=True)

    st.markdown("**Component inspector**")
    if focus == "None":
        st.info("Pick a component in the sidebar to highlight it and see the main equations/interpretation.")
    elif focus == "Ca influx":
        st.write("Membrane Ca²⁺ entry driven by stimulus.")
        st.latex(r"J_{in} = v_{in} \cdot s(t)")
    elif focus == "PLC/IP3":
        st.write("Stimulus drives PLC which produces IP3; IP3 is degraded." )
        st.latex(r"\frac{dIP3}{dt} = v_{plc}\,s(t) - k_{deg}\,IP3")
    elif focus == "IP3R":
        st.write("IP3R releases Ca²⁺ from ER to cytosol; depends on IP3, cytosolic Ca²⁺ activation, and ER loading." )
        st.latex(r"J_{IP3R} = v_{ip3r}\,g(IP3)\,h(Ca)\,q(Ca_{ER})")
    elif focus == "SERCA":
        st.write("SERCA pumps Ca²⁺ back into ER (Hill-like uptake).")
        st.latex(r"J_{SERCA} = v_{serca}\,\frac{Ca^2}{K_{serca}^2 + Ca^2}")
    elif focus == "PMCA/NCX":
        st.write("Plasma membrane extrusion via PMCA/NCX (Michaelis–Menten).")
        st.latex(r"J_{PM} = v_{pmca}\,\frac{Ca}{K_{pmca} + Ca}")
    elif focus == "CaM":
        st.write("Ca²⁺/calmodulin activation (fractional binding model).")
        st.latex(r"\frac{dCaM^*}{dt} = k_{on} Ca^n (1-CaM^*) - k_{off} CaM^*")
    elif focus == "PKC":
        st.write("PKC activation as a Ca²⁺ + DAG-proxy (stimulus) driven fraction.")
        st.latex(r"\frac{dPKC^*}{dt} = k_{on}\,f(Ca)\,f(DAG)\,(1-PKC^*) - k_{off}PKC^*")
    elif focus in ("Ca cyt", "Ca ER"):
        st.write("Two-compartment Ca²⁺ exchange between cytosol and ER plus membrane influx/extrusion.")

with tabs[1]:
    st.subheader("Time series")
    st.plotly_chart(time_series_figure(df, ["stim"], "Stimulus"), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(time_series_figure(df, ["ca_cyt_uM", "ca_er_uM"], "Ca²⁺ compartments (uM)"), use_container_width=True)
    with c2:
        st.plotly_chart(time_series_figure(df, ["ip3_uM", "cam_active", "pkc_active"], "IP3 + downstream activations"), use_container_width=True)

    st.plotly_chart(time_series_figure(df, ["ip3r_flux_uM_s", "serca_flux_uM_s", "pmca_flux_uM_s"], "Fluxes (uM/s)"), use_container_width=True)

with tabs[2]:
    st.subheader("Phase & spectrum")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(phase_plane_figure(df, "ip3_uM", "ca_cyt_uM", "Phase plane: Ca²⁺ vs IP3"), use_container_width=True)
        st.plotly_chart(phase_plane_figure(df, "ca_cyt_uM", "cam_active", "Phase plane: CaM* vs Ca²⁺"), use_container_width=True)
    with c2:
        t = df["t_s"].to_numpy()
        x = df["ca_cyt_uM"].to_numpy()
        if t.size >= 16:
            st.plotly_chart(spectrum_figure(t, x, "Spectrum: cytosolic Ca²⁺"), use_container_width=True)
        else:
            st.info("Increase duration or decrease dt for FFT.")

with tabs[3]:
    st.subheader("Metrics")
    met = metrics_table(df, stim_start_s=float(stim_start), stim_end_s=float(stim_end))
    st.dataframe(met, use_container_width=True)

    # Quick headline metrics for Ca cyt
    ca_row = met[met["signal"] == "ca_cyt_uM"].iloc[0]
    a, b, c, d = st.columns(4)
    a.metric("Ca²⁺ baseline (uM)", f"{ca_row['baseline']:.4g}")
    b.metric("Ca²⁺ peak (uM)", f"{ca_row['peak']:.4g}")
    c.metric("Δ peak (uM)", f"{ca_row['delta_peak']:.4g}")
    d.metric("AUC (uM·s)", f"{ca_row['auc']:.4g}")

    st.download_button(
        "Download metrics (CSV)",
        data=met.to_csv(index=False).encode("utf-8"),
        file_name="calcium_metrics.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download simulation (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="calcium_timeseries.csv",
        mime="text/csv",
    )

with tabs[4]:
    st.subheader("Parameter sweep (heatmap)")
    st.caption("Sweeps two parameters and plots a chosen metric of cytosolic Ca²⁺.")

    param_options = {
        "v_in_max": ("Max Ca influx v_in (uM/s)", 0.0, 5.0),
        "v_plc_max": ("PLC v_plc (uM/s)", 0.0, 2.0),
        "v_serca_max": ("SERCA v_serca (uM/s)", 0.0, 5.0),
        "v_pmca_max": ("PMCA/NCX v_pmca (uM/s)", 0.0, 5.0),
        "v_ip3r_max": ("IP3R v_ip3r (uM/s)", 0.0, 30.0),
        "k_ip3_deg_per_s": ("IP3 degradation k_deg (1/s)", 0.0, 2.0),
    }

    c1, c2, c3 = st.columns(3)
    with c1:
        x_param = st.selectbox("X parameter", list(param_options.keys()), index=0)
    with c2:
        y_param = st.selectbox("Y parameter", list(param_options.keys()), index=1)
    with c3:
        metric_name = st.selectbox("Metric", ["peak", "auc", "dominant_fft_hz", "n_peaks"], index=0)

    grid_n = st.slider("Grid size", 5, 20, 10, 1)

    x_label, x_min, x_max = param_options[x_param]
    y_label, y_min, y_max = param_options[y_param]

    x_vals = np.linspace(x_min, x_max, grid_n)
    y_vals = np.linspace(y_min, y_max, grid_n)

    run_btn = st.button("Run sweep", type="primary")

    if run_btn:
        from src.metrics import compute_signal_metrics

        z = np.zeros((grid_n, grid_n), dtype=float)
        t_end = float(t_end_s)
        dt = float(dt_s)

        progress = st.progress(0)
        total = grid_n * grid_n
        k = 0

        for yi, yv in enumerate(y_vals):
            for xi, xv in enumerate(x_vals):
                mp = model_params
                mp = mp.__class__(**{**mp.__dict__, x_param: float(xv), y_param: float(yv)})
                try:
                    sim_df = simulate(model=mp, stim=stim_params, t_end_s=t_end, dt_s=dt, solver=solver)
                    m = compute_signal_metrics(
                        sim_df["t_s"].to_numpy(),
                        sim_df["ca_cyt_uM"].to_numpy(),
                        stim_start_s=float(stim_start),
                        stim_end_s=float(stim_end),
                    )
                    z[yi, xi] = float(getattr(m, metric_name)) if getattr(m, metric_name) is not None else np.nan
                except Exception:
                    z[yi, xi] = np.nan

                k += 1
                progress.progress(min(1.0, k / total))

        st.plotly_chart(
            heatmap_figure(
                z=z,
                x=x_vals,
                y=y_vals,
                title=f"Sweep: {metric_name} of cytosolic Ca²⁺",
                x_label=x_label,
                y_label=y_label,
            ),
            use_container_width=True,
        )
