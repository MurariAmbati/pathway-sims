from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.nfkb.analysis import summarize
from src.nfkb.model import NFkBParams
from src.nfkb.simulate import simulate_deterministic, simulate_stochastic


st.set_page_config(page_title="nfkb", layout="wide")

st.title("nf‑κb pathway · nfkb")
st.caption("inflammation · immune signaling · stress responses")


def ikk_function(kind: str, base: float, pulse_amp: float, pulse_start: float, pulse_end: float):
    if kind == "constant":
        return lambda t: base
    if kind == "pulse":
        def f(t: float) -> float:
            return base + (pulse_amp if (pulse_start <= t <= pulse_end) else 0.0)
        return f
    raise ValueError("unknown ikk kind")


with st.sidebar:
    st.header("model")

    ntot = st.number_input("ntot", min_value=0.1, value=1.0, step=0.1)

    k_imp = st.number_input("k_imp", min_value=0.0, value=2.5, step=0.1)
    k_exp = st.number_input("k_exp", min_value=0.0, value=1.2, step=0.1)

    k_tx = st.number_input("k_tx", min_value=0.0, value=2.0, step=0.1)
    k_mdeg = st.number_input("k_mdeg", min_value=0.0, value=0.6, step=0.05)

    k_tl = st.number_input("k_tl", min_value=0.0, value=1.6, step=0.1)
    k_pdeg = st.number_input("k_pdeg", min_value=0.0, value=0.2, step=0.05)
    k_ikk = st.number_input("k_ikk", min_value=0.0, value=1.0, step=0.1)

    hill_h = st.number_input("hill_h", min_value=1.0, value=3.0, step=1.0)
    hill_k = st.number_input("hill_k", min_value=0.01, value=0.25, step=0.05)

    st.divider()
    st.header("input (ikk)")
    ikk_kind = st.selectbox("ikk kind", ["constant", "pulse"], index=0)
    ikk_base = st.number_input("ikk base", min_value=0.0, value=0.8, step=0.05)
    pulse_amp = st.number_input("pulse amp", min_value=0.0, value=0.8, step=0.05)
    pulse_start = st.number_input("pulse start", min_value=0.0, value=5.0, step=0.5)
    pulse_end = st.number_input("pulse end", min_value=0.0, value=25.0, step=0.5)

    st.divider()
    st.header("simulation")
    mode = st.selectbox("mode", ["deterministic (ode)", "stochastic (sde-like)"], index=0)
    t_end = st.number_input("t_end", min_value=5.0, value=100.0, step=5.0)
    dt = st.number_input("dt", min_value=0.001, value=0.05, step=0.01)

    if mode.startswith("deterministic"):
        method = st.selectbox("ode method", ["RK45", "Radau", "BDF"], index=0)
        sigma = None
        seed = None
    else:
        sigma = st.number_input("sigma", min_value=0.0, value=0.02, step=0.01)
        seed = st.number_input("seed", min_value=0, value=0, step=1)
        method = None

    st.divider()
    st.header("initial state")
    x0_nn = st.number_input("nn0", min_value=0.0, value=0.1, step=0.05)
    x0_im = st.number_input("im0", min_value=0.0, value=0.1, step=0.05)
    x0_i = st.number_input("i0", min_value=0.0, value=1.0, step=0.1)

    run = st.button("run", type="primary")


p = NFkBParams(
    ntot=float(ntot),
    k_imp=float(k_imp),
    k_exp=float(k_exp),
    k_tx=float(k_tx),
    k_mdeg=float(k_mdeg),
    k_tl=float(k_tl),
    k_pdeg=float(k_pdeg),
    k_ikk=float(k_ikk),
    hill_h=float(hill_h),
    hill_k=float(hill_k),
)

ikk_fn = ikk_function(
    ikk_kind,
    base=float(ikk_base),
    pulse_amp=float(pulse_amp),
    pulse_start=float(pulse_start),
    pulse_end=float(pulse_end),
)

x0 = np.array([float(x0_nn), float(x0_im), float(x0_i)], dtype=float)


@st.cache_data(show_spinner=False)
def run_sim_cached(p_dict: dict, x0_list: list[float], ikk_kind_: str, ikk_params: tuple, mode_: str, t_end_: float, dt_: float, method_: str | None, sigma_: float | None, seed_: int | None) -> pd.DataFrame:
    p2 = NFkBParams(**p_dict)
    ikk_fn2 = ikk_function(ikk_kind_, *ikk_params)
    x02 = np.array(x0_list, dtype=float)

    if mode_.startswith("deterministic"):
        return simulate_deterministic(p2, x02, ikk_fn2, t_end=t_end_, dt=dt_, method=method_ or "RK45")
    return simulate_stochastic(p2, x02, ikk_fn2, t_end=t_end_, dt=dt_, sigma=float(sigma_ or 0.0), seed=seed_)


if run or "df" not in st.session_state:
    try:
        df = run_sim_cached(
            p_dict=p.__dict__,
            x0_list=x0.tolist(),
            ikk_kind_=ikk_kind,
            ikk_params=(float(ikk_base), float(pulse_amp), float(pulse_start), float(pulse_end)),
            mode_=mode,
            t_end_=float(t_end),
            dt_=float(dt),
            method_=method,
            sigma_=sigma,
            seed_=int(seed) if seed is not None else None,
        )
        st.session_state["df"] = df
    except Exception as e:
        st.error(str(e))


df = st.session_state.get("df")
if df is None or df.empty:
    st.stop()

metrics = summarize(df)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("nn mean", f"{metrics['nn_mean']:.3f}" if metrics["nn_mean"] is not None else "-")
c2.metric("nn ptp", f"{metrics['nn_ptp']:.3f}" if metrics["nn_ptp"] is not None else "-")
c3.metric("i mean", f"{metrics['i_mean']:.3f}" if metrics["i_mean"] is not None else "-")
c4.metric("i ptp", f"{metrics['i_ptp']:.3f}" if metrics["i_ptp"] is not None else "-")
c5.metric("period (fft)", f"{metrics['period_fft']:.2f}" if metrics["period_fft"] is not None else "-")


left, right = st.columns([2, 1])

with left:
    st.subheader("time series")
    df_long = df.melt(id_vars=["t"], value_vars=["nn", "im", "i", "ikk"], var_name="var", value_name="value")
    fig = px.line(df_long, x="t", y="value", color="var", template="plotly_white")
    fig.update_layout(legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("phase portrait")
    fig2 = px.line(df, x="i", y="nn", template="plotly_white")
    fig2.update_layout(xaxis_title="i (iκb protein)", yaxis_title="nn (nuclear nf‑κb)")
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("spectrum (nn)")

# fft plot using evenly-sampled series

t = df["t"].to_numpy()
y = df["nn"].to_numpy()

burn = int(0.2 * len(df))
t2 = t[burn:]
y2 = y[burn:] - float(np.mean(y[burn:]))

dt_eff = float(np.median(np.diff(t2)))
if len(t2) > 8 and dt_eff > 0:
    win = np.hanning(len(y2))
    yf = np.fft.rfft(y2 * win)
    freqs = np.fft.rfftfreq(len(y2), d=dt_eff)
    power = (np.abs(yf) ** 2)

    spec = pd.DataFrame({"freq": freqs[1:], "power": power[1:]})
    fig3 = px.line(spec, x="freq", y="power", template="plotly_white")
    fig3.update_layout(xaxis_title="frequency", yaxis_title="power")
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("not enough samples for spectrum")

with st.expander("data"):
    st.dataframe(df, use_container_width=True, height=260)
