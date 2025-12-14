from __future__ import annotations

import tempfile

import numpy as np

from notch.boolean import BooleanParams, simulate_boolean
from notch.graph import load_edge_list
from notch.metrics import field_summary, neighbor_anticorrelation
from notch.neighbors import grid_adjacency, neighbor_mean
from notch.ode import ODEParams, simulate_ode
from notch.plot import plot_ode_snapshot
from notch.types import Grid


def _uploaded_edges_to_adjacency(uploaded, *, n: int) -> list[list[int]] | None:
    if uploaded is None:
        return None

    data = uploaded.getvalue()
    text = data.decode("utf-8", errors="replace")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=True) as f:
        f.write(text)
        f.flush()
        return load_edge_list(f.name, n=n, undirected=True)


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="notch", layout="wide")

    st.title("notch")
    st.caption("cell–cell communication for fate decisions and patterning")

    if "running" not in st.session_state:
        st.session_state.running = False

    with st.sidebar:
        model = st.radio("model", ["ode", "boolean"], index=0)

        rows = st.number_input("rows", min_value=1, max_value=200, value=20, step=1)
        cols = st.number_input("cols", min_value=1, max_value=200, value=20, step=1)
        topology = st.selectbox("topology", ["von_neumann", "moore"], index=0)
        boundary = st.selectbox("boundary", ["periodic", "reflect"], index=0)

        st.divider()
        st.write("adjacency")
        edges = st.file_uploader("edge list (optional)", type=["txt", "csv"], accept_multiple_files=False)

        st.divider()
        seed = st.number_input("seed", min_value=0, max_value=10_000_000, value=0, step=1)

        run_clicked = st.button("run", type="primary")
        if run_clicked:
            st.session_state.running = True

    grid = Grid(int(rows), int(cols), topology=str(topology), boundary=str(boundary))
    adj = _uploaded_edges_to_adjacency(edges, n=grid.n)

    if not st.session_state.running:
        st.info("set parameters in the sidebar, then click run")
        return

    if model == "ode":
        c1, c2 = st.columns([1, 1])
        with c1:
            t_end = st.slider("t", min_value=1.0, max_value=200.0, value=40.0, step=1.0)
            dt = st.slider("dt", min_value=0.01, max_value=2.0, value=0.2, step=0.01)

            alpha = st.slider("alpha", min_value=0.0, max_value=20.0, value=5.0, step=0.1)
            beta = st.slider("beta", min_value=0.0, max_value=10.0, value=0.0, step=0.1)

            pn = st.slider("pn", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
            pd = st.slider("pd", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

        with c2:
            gn = st.slider("gn", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
            gd = st.slider("gd", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
            gi = st.slider("gi", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

            k_trans = st.slider("k_trans", min_value=0.01, max_value=5.0, value=0.5, step=0.01)
            n_trans = st.slider("n_trans", min_value=1.0, max_value=8.0, value=2.0, step=0.5)

            k_rep = st.slider("k_rep", min_value=0.01, max_value=5.0, value=0.5, step=0.01)
            n_rep = st.slider("n_rep", min_value=1.0, max_value=8.0, value=2.0, step=0.5)

            eps = st.slider("eps", min_value=0.0, max_value=0.05, value=0.001, step=0.001)

        p = ODEParams(
            pn=float(pn),
            pd=float(pd),
            gn=float(gn),
            gd=float(gd),
            gi=float(gi),
            alpha=float(alpha),
            beta=float(beta),
            k_trans=float(k_trans),
            n_trans=float(n_trans),
            k_rep=float(k_rep),
            n_rep=float(n_rep),
            eps=float(eps),
        )

        out = simulate_ode(
            grid=grid,
            adjacency=adj,
            t_span=(0.0, float(t_end)),
            dt=float(dt),
            params=p,
            seed=int(seed),
        )

        st.subheader("snapshot")
        fig = plot_ode_snapshot(grid, out, idx=-1)
        st.pyplot(fig, clear_figure=True)

        st.subheader("metrics")
        adjacency_eff = adj if adj is not None else grid_adjacency(grid)
        d_last = out["d"][-1]
        icd_last = out["icd"][-1]
        d_bar = neighbor_mean(d_last, adjacency_eff)

        ds = field_summary(d_last)
        is_ = field_summary(icd_last)
        metrics = {
            "d_neighbor_corr": neighbor_anticorrelation(d_last, d_bar),
            "d_mean": ds["mean"],
            "d_std": ds["std"],
            "icd_mean": is_["mean"],
            "icd_std": is_["std"],
        }
        st.json(metrics)

    else:
        steps = st.slider("steps", min_value=1, max_value=300, value=80, step=1)
        trans_threshold = st.slider("trans_threshold", min_value=0.0, max_value=1.0, value=0.35, step=0.01)
        bias = st.slider("bias", min_value=-2.0, max_value=2.0, value=0.0, step=0.05)
        noise = st.slider("noise", min_value=0.0, max_value=3.0, value=1.0, step=0.05)

        p = BooleanParams(
            trans_threshold=float(trans_threshold),
            bias=float(bias),
            noise=float(noise),
        )

        out = simulate_boolean(grid=grid, adjacency=adj, steps=int(steps), params=p, seed=int(seed))

        d_last = out["d"][-1].astype(float)
        icd_last = out["icd"][-1].astype(float)

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(d_last.reshape(grid.rows, grid.cols), cmap="viridis", interpolation="nearest")
        axes[0].set_title("delta (boolean)")
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        axes[1].imshow(icd_last.reshape(grid.rows, grid.cols), cmap="viridis", interpolation="nearest")
        axes[1].set_title("icd (boolean)")
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)


if __name__ == "__main__":
    main()
