import numpy as np

from notch.ode import simulate_ode
from notch.types import Grid


def test_ode_shapes():
    grid = Grid(4, 5)
    out = simulate_ode(grid=grid, t_span=(0.0, 1.0), dt=0.5, seed=1)
    assert out["n"].shape[1] == grid.n
    assert out["d"].shape == out["n"].shape
    assert out["icd"].shape == out["n"].shape
    assert out["t"].shape[0] == out["n"].shape[0]
    assert np.isfinite(out["icd"]).all()
