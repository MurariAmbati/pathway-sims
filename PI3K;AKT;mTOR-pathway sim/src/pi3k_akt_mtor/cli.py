from __future__ import annotations

import argparse
import json
from pathlib import Path

from .params import Inputs, ModelParams
from .sim import simulate
from .variants import Variant


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PI3K/AKT/mTOR ODE simulator")

    p.add_argument(
        "--variant",
        type=str,
        default=Variant.baseline.value,
        choices=[v.value for v in Variant],
        help="Model variant toggles",
    )

    p.add_argument("--t-end", type=float, default=120.0)
    p.add_argument("--n-points", type=int, default=600)

    p.add_argument(
        "--no-derived",
        action="store_true",
        help="Disable derived output columns (Growth/Survival/etc)",
    )

    # Inputs
    p.add_argument("--ligand", type=float, default=1.0)
    p.add_argument("--erk-input", type=float, default=0.0)
    p.add_argument("--ampk-input", type=float, default=0.0)

    # Drugs
    p.add_argument("--pi3k-activity", type=float, default=1.0)
    p.add_argument("--akt-activity", type=float, default=1.0)
    p.add_argument("--mtorc1-activity", type=float, default=1.0)

    # Output
    p.add_argument("--export", type=str, default="", help="Export CSV timeseries")
    p.add_argument("--export-meta", type=str, default="", help="Export JSON metadata")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    variant = Variant(args.variant)

    inputs = Inputs(
        ligand=args.ligand,
        erk_input=args.erk_input,
        ampk_input=args.ampk_input,
        pi3k_activity=args.pi3k_activity,
        akt_activity=args.akt_activity,
        mtorc1_activity=args.mtorc1_activity,
    )

    df, meta = simulate(
        variant=variant,
        params=ModelParams(),
        inputs=inputs,
        t_end=args.t_end,
        n_points=args.n_points,
        include_derived=(not args.no_derived),
    )

    if args.export:
        Path(args.export).write_text(df.to_csv(index=False))

    if args.export_meta:
        Path(args.export_meta).write_text(json.dumps(meta, indent=2, sort_keys=True))

    # Default behavior: print final state as JSON-ish summary
    final = df.iloc[-1].to_dict()
    print(json.dumps({"final": final, "meta": meta}, indent=2, sort_keys=True))

    return 0
