from __future__ import annotations

import json

from rtk_egfr.params import default_initial_state, default_params
from rtk_egfr.sim import simulate, summary_metrics


def main() -> None:
    params = default_params()
    df = simulate(params=params, initial_state=default_initial_state(params), t_end=60.0, dt=0.2)
    print(json.dumps(summary_metrics(df), indent=2))


if __name__ == "__main__":
    main()
