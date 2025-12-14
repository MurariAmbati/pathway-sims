from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from .boolean import BooleanParams, simulate_boolean
from .graph import load_edge_list
from .metrics import field_summary, neighbor_anticorrelation
from .ode import ODEParams, simulate_ode
from .neighbors import neighbor_mean
from .plot import plot_ode_snapshot
from .types import Grid


def _add_grid_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--rows", type=int, default=20)
    p.add_argument("--cols", type=int, default=20)
    p.add_argument("--topology", choices=["von_neumann", "moore"], default="von_neumann")
    p.add_argument("--boundary", choices=["periodic", "reflect"], default="periodic")


def _parse_floats_csv(s: str) -> list[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("expected at least one value")
    return [float(p) for p in parts]


def _maybe_load_adjacency(args: argparse.Namespace, grid: Grid) -> list[list[int]] | None:
    if args.edges is None:
        return None
    adj = load_edge_list(args.edges, n=grid.n, undirected=True)
    return adj


def cmd_ode(args: argparse.Namespace) -> int:
    grid = Grid(args.rows, args.cols, topology=args.topology, boundary=args.boundary)
    adjacency = _maybe_load_adjacency(args, grid)
    params = ODEParams(
        pn=args.pn,
        pd=args.pd,
        gn=args.gn,
        gd=args.gd,
        gi=args.gi,
        alpha=args.alpha,
        beta=args.beta,
        k_trans=args.k_trans,
        n_trans=args.n_trans,
        k_rep=args.k_rep,
        n_rep=args.n_rep,
        eps=args.eps,
    )

    out = simulate_ode(
        grid=grid,
        adjacency=adjacency,
        t_span=(0.0, args.t),
        dt=args.dt,
        params=params,
        seed=args.seed,
    )

    if args.save_npz:
        np.savez_compressed(args.save_npz, **out)

    if args.plot:
        import matplotlib.pyplot as plt

        fig = plot_ode_snapshot(grid, out, idx=-1)
        if args.savefig:
            fig.savefig(args.savefig, dpi=160)
        else:
            plt.show()

    return 0


def cmd_boolean(args: argparse.Namespace) -> int:
    grid = Grid(args.rows, args.cols, topology=args.topology, boundary=args.boundary)
    adjacency = _maybe_load_adjacency(args, grid)
    params = BooleanParams(
        trans_threshold=args.trans_threshold,
        bias=args.bias,
        noise=args.noise,
    )

    out = simulate_boolean(grid=grid, adjacency=adjacency, steps=args.steps, params=params, seed=args.seed)

    if args.save_npz:
        np.savez_compressed(args.save_npz, **out)

    if args.plot:
        import matplotlib.pyplot as plt

        last_d = out["d"][-1].astype(float)
        plt.figure(figsize=(5, 4))
        plt.imshow(last_d.reshape(grid.rows, grid.cols), cmap="viridis", interpolation="nearest")
        plt.title("delta (boolean)")
        plt.xticks([])
        plt.yticks([])
        plt.colorbar(fraction=0.046, pad=0.04)
        if args.savefig:
            plt.savefig(args.savefig, dpi=160)
        else:
            plt.show()

    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    grid = Grid(args.rows, args.cols, topology=args.topology, boundary=args.boundary)
    adjacency = _maybe_load_adjacency(args, grid)

    alphas = _parse_floats_csv(args.alpha)

    out_f = open(args.out, "w", newline="") if args.out is not None else sys.stdout
    close_out = args.out is not None

    try:
        w = csv.DictWriter(
            out_f,
            fieldnames=[
                "alpha",
                "d_mean",
                "d_std",
                "d_min",
                "d_max",
                "d_neighbor_corr",
                "icd_mean",
                "icd_std",
            ],
        )
        w.writeheader()

        for a in alphas:
            p = ODEParams(
                pn=args.pn,
                pd=args.pd,
                gn=args.gn,
                gd=args.gd,
                gi=args.gi,
                alpha=a,
                beta=args.beta,
                k_trans=args.k_trans,
                n_trans=args.n_trans,
                k_rep=args.k_rep,
                n_rep=args.n_rep,
                eps=args.eps,
            )
            sim = simulate_ode(grid=grid, adjacency=adjacency, t_span=(0.0, args.t), dt=args.dt, params=p, seed=args.seed)

            d_last = sim["d"][-1]
            icd_last = sim["icd"][-1]

            # need neighbor mean for the metric; use the same adjacency as the sim.
            if adjacency is None:
                # simulate_ode built one internally; rebuild for metric consistency
                adjacency_eff = _maybe_load_adjacency(args, grid)
            else:
                adjacency_eff = adjacency
            if adjacency_eff is None:
                from .neighbors import grid_adjacency

                adjacency_eff = grid_adjacency(grid)

            d_bar = neighbor_mean(d_last, adjacency_eff)

            ds = field_summary(d_last)
            is_ = field_summary(icd_last)

            w.writerow(
                {
                    "alpha": a,
                    "d_mean": ds["mean"],
                    "d_std": ds["std"],
                    "d_min": ds["min"],
                    "d_max": ds["max"],
                    "d_neighbor_corr": neighbor_anticorrelation(d_last, d_bar),
                    "icd_mean": is_["mean"],
                    "icd_std": is_["std"],
                }
            )
    finally:
        if close_out:
            out_f.close()

    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="notch", description="notch pathway simulation")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_ode = sub.add_parser("ode", help="multicellular ode model")
    _add_grid_args(ap_ode)
    ap_ode.add_argument("--edges", type=str, default=None, help="edge list file (0-based) to override grid adjacency")
    ap_ode.add_argument("--t", type=float, default=40.0)
    ap_ode.add_argument("--dt", type=float, default=0.2)
    ap_ode.add_argument("--seed", type=int, default=0)

    ap_ode.add_argument("--pn", type=float, default=1.0)
    ap_ode.add_argument("--pd", type=float, default=1.0)
    ap_ode.add_argument("--gn", type=float, default=1.0)
    ap_ode.add_argument("--gd", type=float, default=1.0)
    ap_ode.add_argument("--gi", type=float, default=1.0)

    ap_ode.add_argument("--alpha", type=float, default=5.0)
    ap_ode.add_argument("--beta", type=float, default=0.0)

    ap_ode.add_argument("--k-trans", type=float, default=0.5, dest="k_trans")
    ap_ode.add_argument("--n-trans", type=float, default=2.0, dest="n_trans")
    ap_ode.add_argument("--k-rep", type=float, default=0.5, dest="k_rep")
    ap_ode.add_argument("--n-rep", type=float, default=2.0, dest="n_rep")

    ap_ode.add_argument("--eps", type=float, default=1e-3)

    ap_ode.add_argument("--plot", action="store_true")
    ap_ode.add_argument("--savefig", type=str, default="")
    ap_ode.add_argument("--save-npz", type=str, default="", dest="save_npz")
    ap_ode.set_defaults(func=cmd_ode)

    ap_bool = sub.add_parser("boolean", help="logical/boolean model")
    _add_grid_args(ap_bool)
    ap_bool.add_argument("--edges", type=str, default=None, help="edge list file (0-based) to override grid adjacency")
    ap_bool.add_argument("--steps", type=int, default=60)
    ap_bool.add_argument("--seed", type=int, default=0)
    ap_bool.add_argument("--trans-threshold", type=float, default=0.35, dest="trans_threshold")
    ap_bool.add_argument("--bias", type=float, default=0.0)
    ap_bool.add_argument("--noise", type=float, default=1.0)
    ap_bool.add_argument("--plot", action="store_true")
    ap_bool.add_argument("--savefig", type=str, default="")
    ap_bool.add_argument("--save-npz", type=str, default="", dest="save_npz")
    ap_bool.set_defaults(func=cmd_boolean)

    ap_sweep = sub.add_parser("sweep", help="ode sweep over alpha (csv output)")
    _add_grid_args(ap_sweep)
    ap_sweep.add_argument("--edges", type=str, default=None, help="edge list file (0-based) to override grid adjacency")
    ap_sweep.add_argument("--alpha", type=str, required=True, help="comma-separated alpha values, e.g. 1,2,4,8")
    ap_sweep.add_argument("--t", type=float, default=40.0)
    ap_sweep.add_argument("--dt", type=float, default=0.2)
    ap_sweep.add_argument("--seed", type=int, default=0)

    ap_sweep.add_argument("--pn", type=float, default=1.0)
    ap_sweep.add_argument("--pd", type=float, default=1.0)
    ap_sweep.add_argument("--gn", type=float, default=1.0)
    ap_sweep.add_argument("--gd", type=float, default=1.0)
    ap_sweep.add_argument("--gi", type=float, default=1.0)
    ap_sweep.add_argument("--beta", type=float, default=0.0)
    ap_sweep.add_argument("--k-trans", type=float, default=0.5, dest="k_trans")
    ap_sweep.add_argument("--n-trans", type=float, default=2.0, dest="n_trans")
    ap_sweep.add_argument("--k-rep", type=float, default=0.5, dest="k_rep")
    ap_sweep.add_argument("--n-rep", type=float, default=2.0, dest="n_rep")
    ap_sweep.add_argument("--eps", type=float, default=1e-3)
    ap_sweep.add_argument("--out", type=str, default=None, help="write csv to file (default: stdout)")
    ap_sweep.set_defaults(func=cmd_sweep)

    args = ap.parse_args(argv)

    if hasattr(args, "savefig") and args.savefig == "":
        args.savefig = None
    if hasattr(args, "save_npz") and args.save_npz == "":
        args.save_npz = None

    return int(args.func(args))
