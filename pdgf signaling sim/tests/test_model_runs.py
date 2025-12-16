import numpy as np

from src.pdgf_sim.model import PDGFParams, STATE_NAMES, default_initial_state
from src.pdgf_sim.sim import normalize_for_display, simulate


def test_simulation_runs_and_shapes():
    y0 = default_initial_state(ligand_nM=2.0, receptor_nM=10.0)
    df = simulate(PDGFParams(), y0=y0, t_end_min=60.0, n_points=200)
    assert df.shape[1] == 1 + len(STATE_NAMES)
    assert df.shape[0] == 200


def test_states_reasonable_ranges_after_clipping():
    y0 = default_initial_state(ligand_nM=5.0, receptor_nM=20.0)
    df = normalize_for_display(simulate(PDGFParams(), y0=y0, t_end_min=120.0, n_points=250))
    assert (df["L"] >= 0).all()
    assert (df["R"] >= 0).all()
    assert (df["ERK"].between(0.0, 1.0)).all()
    assert (df["AKT"].between(0.0, 1.0)).all()
    assert (df["P"] >= 0).all()
