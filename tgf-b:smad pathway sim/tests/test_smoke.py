import numpy as np

from tgfb_smad.simulate import dose_response, simulate_timecourse


def test_timecourse_runs():
    df = simulate_timecourse(1.0, rtk_level=0.2, t_end=8.0, n_points=120)
    assert len(df) == 120
    assert (df["N"] >= 0).all()


def test_dose_response_runs():
    doses = np.linspace(0.0, 2.0, 8)
    dr = dose_response(doses, metric="EMT_end", rtk_level=0.3, t_end=6.0)
    assert len(dr) == 8
    assert (dr["dose"].to_numpy() >= 0).all()
