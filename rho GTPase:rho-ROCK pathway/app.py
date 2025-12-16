from __future__ import annotations

import streamlit as st

from rho_rock.pathway import build_pathway_graph
from rho_rock.sim import Params, simulate
from rho_rock.viz import phase_plane_figure, pathway_graph_figure, timeseries_figure


st.set_page_config(page_title="rho_rock", layout="wide")

st.title("rho gtpase / rho–rock pathway")
st.caption("actin cytoskeleton dynamics · cell shape · migration · contractility")


with st.sidebar:
    st.subheader("controls")
    view = st.radio("view", ["pathway", "dynamics"], horizontal=False)

    st.divider()
    st.subheader("upstream")
    gef = st.slider("rhoGEF drive", 0.0, 2.0, 1.0, 0.05)
    gap = st.slider("rhoGAP activity", 0.0, 2.0, 1.0, 0.05)

    st.divider()
    st.subheader("perturbations")
    st.caption("multipliers applied during simulation")
    pert_rock = st.slider("rock", 0.0, 2.0, 1.0, 0.05)
    pert_mdia = st.slider("mdia", 0.0, 2.0, 1.0, 0.05)
    pert_cofilin = st.slider("cofilin", 0.0, 2.0, 1.0, 0.05)

    st.divider()
    st.subheader("time")
    t_end = st.slider("t_end", 10, 240, 60, 5)


if view == "pathway":
    g = build_pathway_graph()
    left, right = st.columns([1.6, 1.0], gap="large")

    with left:
        st.subheader("graph")
        st.plotly_chart(pathway_graph_figure(g), use_container_width=True)

    with right:
        st.subheader("notes")
        st.write(
            "this is a compact mechanistic map centered on rhoa → rock / mdia. "
            "it’s meant for interactive exploration rather than exhaustive biology."
        )
        st.markdown("**node legend**")
        st.markdown("- circle: protein")
        st.markdown("- square: process")
        st.markdown("- diamond: phenotype")

        st.markdown("**edge sign**")
        st.markdown("- activation: +")
        st.markdown("- inhibition: -")

else:
    p = Params(gef=gef, gap=gap)
    perturb = {"rock": pert_rock, "mdia": pert_mdia, "cofilin": pert_cofilin}
    sim = simulate(p, t_end=float(t_end), n_points=int(max(200, t_end * 10)), perturb=perturb)

    st.subheader("state trajectories")
    st.plotly_chart(
        timeseries_figure(sim, ["rhoa", "rock", "mdia", "limk", "cofilin", "factin", "mlc"]),
        use_container_width=True,
    )

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("phenotype readouts")
        st.plotly_chart(
            timeseries_figure(sim, ["contractility", "migration"]),
            use_container_width=True,
        )

    with c2:
        st.subheader("phase plane")
        st.plotly_chart(
            phase_plane_figure(sim, "factin", "contractility"),
            use_container_width=True,
        )
