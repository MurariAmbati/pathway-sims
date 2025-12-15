from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


StimType = Literal["Step", "Pulse train", "Sine"]


@dataclass(frozen=True)
class StimulusParams:
    stim_type: StimType = "Pulse train"
    start_s: float = 5.0
    end_s: float = 40.0
    amplitude: float = 1.0  # dimensionless (0..)

    # Pulse train
    period_s: float = 2.0
    duty: float = 0.2  # 0..1

    # Sine
    frequency_hz: float = 0.5
    baseline: float = 0.0


@dataclass(frozen=True)
class ModelParams:
    # Initial conditions (uM or fraction)
    ca_cyt0_uM: float = 0.10
    ca_er0_uM: float = 200.0
    ip3_0_uM: float = 0.05
    cam0: float = 0.0
    pkc0: float = 0.0

    # Volume ratio conversion (V_cyt / V_er)
    vol_cyt_to_er: float = 5.0

    # Membrane influx (uM/s)
    v_in_max: float = 0.8

    # PMCA/NCX extrusion (uM/s)
    v_pmca_max: float = 1.0
    k_pmca_uM: float = 0.25

    # SERCA uptake to ER (uM/s)
    v_serca_max: float = 1.2
    k_serca_uM: float = 0.2

    # ER leak (uM/s)
    k_leak_per_s: float = 0.02

    # IP3 production/degradation (uM/s, 1/s)
    v_plc_max: float = 0.25
    k_ip3_deg_per_s: float = 0.15

    # IP3R release (uM/s)
    v_ip3r_max: float = 8.0
    k_ip3_uM: float = 0.2
    h_ip3: float = 2.0
    k_ca_act_uM: float = 0.25
    h_ca_act: float = 2.0
    k_er_uM: float = 60.0

    # CaM activation (fraction)
    k_cam_on_per_s: float = 4.0
    k_cam_off_per_s: float = 1.5
    n_cam: float = 2.0

    # PKC activation (fraction) (DAG is approximated by stimulus)
    k_pkc_on_per_s: float = 2.5
    k_pkc_off_per_s: float = 0.7
    k_pkc_ca_uM: float = 0.3
    k_pkc_dag: float = 0.25


def stimulus(t_s: float, sp: StimulusParams) -> float:
    if t_s < sp.start_s or t_s > sp.end_s:
        return 0.0

    if sp.stim_type == "Step":
        return max(0.0, sp.amplitude)

    if sp.stim_type == "Pulse train":
        period = max(1e-6, sp.period_s)
        duty = min(max(sp.duty, 0.0), 1.0)
        phase = (t_s - sp.start_s) % period
        on = phase < duty * period
        return max(0.0, sp.amplitude) if on else 0.0

    if sp.stim_type == "Sine":
        w = 2.0 * np.pi * max(0.0, sp.frequency_hz)
        value = sp.baseline + sp.amplitude * 0.5 * (1.0 + np.sin(w * (t_s - sp.start_s)))
        return float(max(0.0, value))

    raise ValueError(f"Unknown stim_type: {sp.stim_type}")


def _ip3r_flux(ca_cyt_uM: float, ca_er_uM: float, ip3_uM: float, p: ModelParams) -> float:
    ip3_gate = (ip3_uM / (p.k_ip3_uM + ip3_uM + 1e-12)) ** p.h_ip3
    ca_gate = (ca_cyt_uM**p.h_ca_act / (p.k_ca_act_uM**p.h_ca_act + ca_cyt_uM**p.h_ca_act + 1e-12))
    er_drive = ca_er_uM / (p.k_er_uM + ca_er_uM + 1e-12)
    return p.v_ip3r_max * ip3_gate * ca_gate * er_drive


def simulate(
    *,
    model: ModelParams,
    stim: StimulusParams,
    t_end_s: float,
    dt_s: float,
    solver: Literal["LSODA", "RK45"] = "LSODA",
) -> pd.DataFrame:
    t_end_s = float(max(1e-6, t_end_s))
    dt_s = float(max(1e-4, dt_s))
    t_eval = np.arange(0.0, t_end_s + 0.5 * dt_s, dt_s)

    y0 = np.array(
        [
            model.ca_cyt0_uM,
            model.ca_er0_uM,
            model.ip3_0_uM,
            model.cam0,
            model.pkc0,
        ],
        dtype=float,
    )

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        ca, ca_er, ip3, cam, pkc = y
        ca = max(0.0, float(ca))
        ca_er = max(0.0, float(ca_er))
        ip3 = max(0.0, float(ip3))
        cam = float(np.clip(cam, 0.0, 1.0))
        pkc = float(np.clip(pkc, 0.0, 1.0))

        s = stimulus(t, stim)

        j_in = model.v_in_max * s
        j_pmca = model.v_pmca_max * (ca / (model.k_pmca_uM + ca + 1e-12))
        j_serca = model.v_serca_max * (ca**2 / (model.k_serca_uM**2 + ca**2 + 1e-12))

        j_ip3r = _ip3r_flux(ca, ca_er, ip3, model)
        j_leak = model.k_leak_per_s * (ca_er - ca)

        dca = j_in + j_ip3r + j_leak - j_serca - j_pmca
        dca_er = model.vol_cyt_to_er * (j_serca - j_ip3r - j_leak)

        dip3 = model.v_plc_max * s - model.k_ip3_deg_per_s * ip3

        dcam = model.k_cam_on_per_s * (ca**model.n_cam) * (1.0 - cam) - model.k_cam_off_per_s * cam

        dag = s
        pkc_drive = (ca / (model.k_pkc_ca_uM + ca + 1e-12)) * (dag / (model.k_pkc_dag + dag + 1e-12))
        dpkc = model.k_pkc_on_per_s * pkc_drive * (1.0 - pkc) - model.k_pkc_off_per_s * pkc

        return np.array([dca, dca_er, dip3, dcam, dpkc], dtype=float)

    sol = solve_ivp(
        rhs,
        t_span=(0.0, t_end_s),
        y0=y0,
        t_eval=t_eval,
        method=solver,
        rtol=1e-6,
        atol=1e-9,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    out = pd.DataFrame(
        {
            "t_s": sol.t,
            "stim": [stimulus(t, stim) for t in sol.t],
            "ca_cyt_uM": sol.y[0],
            "ca_er_uM": sol.y[1],
            "ip3_uM": sol.y[2],
            "cam_active": np.clip(sol.y[3], 0.0, 1.0),
            "pkc_active": np.clip(sol.y[4], 0.0, 1.0),
        }
    )

    # Useful derived quantities
    out["ip3r_flux_uM_s"] = [
        _ip3r_flux(float(ca), float(er), float(ip3), model)
        for ca, er, ip3 in zip(out["ca_cyt_uM"], out["ca_er_uM"], out["ip3_uM"], strict=False)
    ]
    out["pmca_flux_uM_s"] = model.v_pmca_max * (out["ca_cyt_uM"] / (model.k_pmca_uM + out["ca_cyt_uM"] + 1e-12))
    out["serca_flux_uM_s"] = model.v_serca_max * (
        out["ca_cyt_uM"] ** 2 / (model.k_serca_uM**2 + out["ca_cyt_uM"] ** 2 + 1e-12)
    )

    return out
