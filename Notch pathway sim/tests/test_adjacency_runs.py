import numpy as np

from notch.graph import load_edge_list
from notch.ode import simulate_ode
from notch.types import Grid


def test_ode_with_explicit_adjacency(tmp_path):
    # 2x3 grid but with custom edges (ring + diagonal)
    p = tmp_path / "edges.txt"
    p.write_text("0 1\n1 2\n2 3\n3 4\n4 5\n5 0\n0 3\n")
    adj = load_edge_list(p, n=6)

    grid = Grid(2, 3)
    out = simulate_ode(grid=grid, adjacency=adj, t_span=(0.0, 1.0), dt=0.5, seed=2)
    assert out["d"].shape[1] == 6
    assert np.isfinite(out["icd"]).all()
