import streamlit as st

from tgfb_smad.defaults import default_initial_conditions, default_parameters
from tgfb_smad.simulate import crosstalk_grid, dose_response, simulate_timecourse
from tgfb_smad.viz import plot_crosstalk_heatmap, plot_dose_response, plot_timecourse


st.set_page_config(page_title="TGF-β / SMAD (tgfb_smad)", layout="wide")

st.title("TGF-β / SMAD pathway — ODE simulator")
st.caption("Canonical TGF-β receptor → SMAD2/3/4 with Smad7 feedback, plus MAPK/PI3K crosstalk knobs.")


def _freeze(d: dict) -> tuple:
    return tuple(sorted((str(k), float(v)) for k, v in d.items()))


@st.cache_data(show_spinner=False)
def _cached_timecourse(ligand: float, mapk: float, pi3k: float, params_f: tuple, y0_f: tuple, t_end: float, n_points: int):
    params = {k: v for k, v in params_f}
    y0 = {k: v for k, v in y0_f}
    return simulate_timecourse(
        ligand=ligand,
        mapk=mapk,
        pi3k=pi3k,
        params=params,
        y0=y0,
        t_end=t_end,
        n_points=n_points,
    )


@st.cache_data(show_spinner=False)
def _cached_dose_response(d_min: float, d_max: float, n_doses: int, logspace: bool, mapk: float, pi3k: float, params_f: tuple, y0_f: tuple, t_end: float, n_points: int):
    params = {k: v for k, v in params_f}
    y0 = {k: v for k, v in y0_f}
    return dose_response(
        dose_min=d_min,
        dose_max=d_max,
        n_doses=n_doses,
        logspace=logspace,
        mapk=mapk,
        pi3k=pi3k,
        params=params,
        y0=y0,
        t_end=t_end,
        n_points=n_points,
    )


@st.cache_data(show_spinner=False)
def _cached_crosstalk_grid(ligand: float, grid_n: int, readout: str, params_f: tuple, y0_f: tuple, t_end: float, n_points: int):
    import numpy as np

    params = {k: v for k, v in params_f}
    y0 = {k: v for k, v in y0_f}
    mapk_vals = np.linspace(0.0, 1.0, grid_n)
    pi3k_vals = np.linspace(0.0, 1.0, grid_n)
    return crosstalk_grid(
        ligand=ligand,
        mapk_values=mapk_vals,
        pi3k_values=pi3k_vals,
        params=params,
        y0=y0,
        t_end=t_end,
        n_points=n_points,
        readout=readout,
    )

with st.sidebar:
    st.header("Inputs")

    ligand = st.slider("TGF-β dose (a.u.)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    mapk = st.slider("MAPK activity (0–1)", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    pi3k = st.slider("PI3K/AKT activity (0–1)", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    st.divider()
    st.header("Simulation")
    t_end = st.slider("End time", min_value=10.0, max_value=600.0, value=240.0, step=10.0)
    n_points = st.slider("Time points", min_value=200, max_value=3000, value=1200, step=100)

    st.divider()
    st.header("Model knobs")
    p = default_parameters()

    p["k_act"] = st.slider("Receptor activation k_act", 0.001, 1.0, float(p["k_act"]), 0.001)
    p["k_smad7_txn"] = st.slider("Smad7 transcription strength", 0.0, 2.0, float(p["k_smad7_txn"]), 0.01)
    p["k_linker_phos_mapk"] = st.slider("MAPK linker-phos strength", 0.0, 5.0, float(p["k_linker_phos_mapk"]), 0.05)
    p["k_pi3k_synergy"] = st.slider("PI3K synergy on EMT/fibrosis", 0.0, 5.0, float(p["k_pi3k_synergy"]), 0.05)

    advanced = st.checkbox("Show all parameters", value=False)
    if advanced:
        st.caption("Editing many parameters can make stiff systems; defaults are tuned for stability.")
        for k in sorted(p.keys()):
            if k in {"k_act", "k_smad7_txn", "k_linker_phos_mapk", "k_pi3k_synergy"}:
                continue
            if isinstance(p[k], (int, float)):
                p[k] = st.number_input(k, value=float(p[k]))


ic = default_initial_conditions(p)

params_f = _freeze(p)
y0_f = _freeze(ic)

colA, colB = st.columns([2, 1])
with colB:
    st.subheader("Readouts (end time)")

    res = _cached_timecourse(ligand, mapk, pi3k, params_f, y0_f, t_end, n_points)

    end = res.df.iloc[-1]
    st.metric("Nuclear SMAD complex", f"{end['C_n']:.3f}")
    st.metric("Smad7", f"{end['Smad7']:.3f}")
    st.metric("Growth inhibition program", f"{end['G_prog']:.3f}")
    st.metric("EMT program", f"{end['E_prog']:.3f}")
    st.metric("Fibrosis program", f"{end['F_prog']:.3f}")

    st.download_button(
        "Download timecourse CSV",
        data=res.df.to_csv(index=False).encode("utf-8"),
        file_name="tgfb_smad_timecourse.csv",
        mime="text/csv",
    )

with colA:
    tabs = st.tabs(["Time-course", "Dose–response", "Crosstalk (MAPK×PI3K)"])

    with tabs[0]:
        st.subheader("Time-course")
        fig = plot_timecourse(res.df)
        st.plotly_chart(fig, use_container_width=True)


    with tabs[1]:
        st.subheader("Dose–response")
        dose_colA, dose_colB = st.columns([1, 2])
        with dose_colA:
            d_min, d_max = st.slider("Dose range", 0.0, 20.0, (0.0, 10.0), step=0.5)
            n_doses = st.slider("# doses", 10, 120, 40, step=5)
            logspace = st.checkbox("Log-spaced doses", value=True)

            do_sweep = st.button("Run dose sweep")

        with dose_colB:
            if do_sweep:
                dr = _cached_dose_response(
                    d_min,
                    d_max,
                    n_doses,
                    logspace,
                    mapk,
                    pi3k,
                    params_f,
                    y0_f,
                    t_end,
                    max(400, int(n_points * 0.4)),
                )
                fig2 = plot_dose_response(dr)
                st.plotly_chart(fig2, use_container_width=True)
                st.download_button(
                    "Download dose–response CSV",
                    data=dr.to_csv(index=False).encode("utf-8"),
                    file_name="tgfb_smad_dose_response.csv",
                    mime="text/csv",
                )
            else:
                st.info("Click **Run dose sweep** to compute dose–response curves.")

    with tabs[2]:
        st.subheader("Crosstalk map")
        st.caption("Heatmap shows the selected readout at end time as MAPK and PI3K vary from 0→1.")
        cA, cB = st.columns([1, 2])
        with cA:
            grid_n = st.slider("Grid size", 5, 35, 17, step=2)
            readout = st.selectbox("Readout", ["C_n", "Smad7", "G_prog", "E_prog", "F_prog"], index=3)
            run_map = st.button("Run crosstalk map")
        with cB:
            if run_map:
                grid = _cached_crosstalk_grid(
                    ligand,
                    grid_n,
                    readout,
                    params_f,
                    y0_f,
                    t_end,
                    max(300, int(n_points * 0.25)),
                )
                fig3 = plot_crosstalk_heatmap(grid, title=f"{readout} at t={t_end:g} (ligand={ligand:g})")
                st.plotly_chart(fig3, use_container_width=True)
                st.download_button(
                    "Download crosstalk grid CSV",
                    data=grid.to_csv(index=False).encode("utf-8"),
                    file_name="tgfb_smad_crosstalk_grid.csv",
                    mime="text/csv",
                )
            else:
                st.info("Click **Run crosstalk map** to compute the heatmap.")
