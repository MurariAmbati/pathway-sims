from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.integrate import solve_ivp

StimType = Literal["step", "pulse_train", "sine"]


@dataclass(frozen=True)
class Stimulus:
    kind: StimType = "step"
    start_s: float = 5.0
    end_s: float = 25.0
    amplitude: float = 1.0

    # pulse_train
    period_s: float = 2.0
    duty: float = 0.2

    # sine
    freq_hz: float = 0.5


@dataclass(frozen=True)
class Params:
    # Baselines / initial conditions (uM)
    ca_cyt_0: float = 0.1
    ca_er_0: float = 200.0
    ip3_0: float = 0.05

    cam_0: float = 0.0  # fraction
    pkc_0: float = 0.0  # fraction

    # Plasma membrane influx/efflux
    v_influx: float = 0.8  # uM/s at stim=1
    k_pmca: float = 0.4  # 1/s
    k_pmca_half: float = 0.3  # uM

    # SERCA uptake into ER
    v_serca: float = 4.0  # uM/s
    k_serca_half: float = 0.3  # uM

    # ER leak
    k_leak: float = 0.02  # 1/s

    # IP3 dynamics
    v_plc: float = 0.15  # uM/s at stim=1
    k_ip3_deg: float = 0.12  # 1/s

    # IP3R-mediated release (reduced De Young–Keizer style gating)
    v_ip3r: float = 20.0  # uM/s
    k_ip3_half: float = 0.15  # uM
    k_act_half: float = 0.2  # uM
    k_inh_half: float = 0.8  # uM
    hill_ip3: float = 2.0
    hill_act: float = 2.0
    hill_inh: float = 4.0

    # CaM activation (phenomenological)
    cam_k_on: float = 6.0  # 1/(uM^n s)
    cam_k_off: float = 2.0  # 1/s
    cam_hill: float = 2.0

    # PKC activation (phenomenological; uses stim as DAG proxy)
    pkc_k_on: float = 3.5  # 1/(uM^n s)
    pkc_k_off: float = 1.5  # 1/s
    pkc_hill: float = 2.0


def stimulus_value(t: float, stim: Stimulus) -> float:
    if stim.kind == "step":
        return stim.amplitude if (stim.start_s <= t <= stim.end_s) else 0.0

    if stim.kind == "pulse_train":
        if t < stim.start_s or t > stim.end_s:
            return 0.0
        if stim.period_s <= 0:
            return 0.0
        phase = (t - stim.start_s) % stim.period_s
        on = phase <= (stim.duty * stim.period_s)
        return stim.amplitude if on else 0.0

    if stim.kind == "sine":
        if t < stim.start_s or t > stim.end_s:
            return 0.0
        return float(stim.amplitude * (0.5 + 0.5 * np.sin(2 * np.pi * stim.freq_hz * (t - stim.start_s))))

    raise ValueError(f"Unknown stimulus kind: {stim.kind}")


def _hill(x: float, k: float, n: float) -> float:
    x = max(x, 0.0)
    k = max(k, 1e-12)
    return (x**n) / (k**n + x**n)


def _ip3r_open_prob(ca_cyt: float, ip3: float, p: Params) -> float:
    ip3_term = _hill(ip3, p.k_ip3_half, p.hill_ip3)
    act_term = _hill(ca_cyt, p.k_act_half, p.hill_act)
    inh_term = 1.0 / (1.0 + (max(ca_cyt, 0.0) / max(p.k_inh_half, 1e-12)) ** p.hill_inh)
    return float(ip3_term * act_term * inh_term)


def simulate(
    *,
    params: Params,
    stim: Stimulus,
    t_end_s: float = 60.0,
    dt_s: float = 0.02,
    method: str = "LSODA",
) -> dict[str, np.ndarray]:
    """Simulate a compact Ca²⁺ signaling network.

    State vector: [Ca_cyt, Ca_er, IP3, CaM_active, PKC_active]

    Units:
      - Ca, IP3 in uM
      - time in seconds
      - CaM/PKC are fractions in [0,1]
    """

    if t_end_s <= 0:
        raise ValueError("t_end_s must be > 0")
    if dt_s <= 0:
        raise ValueError("dt_s must be > 0")

    y0 = np.array(
        [
            params.ca_cyt_0,
            params.ca_er_0,
            params.ip3_0,
            params.cam_0,
            params.pkc_0,
        ],
        dtype=float,
    )

    t_eval = np.arange(0.0, t_end_s + 1e-12, dt_s)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        ca_cyt, ca_er, ip3, cam_a, pkc_a = y
        s = stimulus_value(t, stim)

        # fluxes
        j_in = params.v_influx * s
        j_pmca = params.k_pmca * _hill(ca_cyt, params.k_pmca_half, 2.0) * ca_cyt
        j_serca = params.v_serca * _hill(ca_cyt, params.k_serca_half, 2.0)
        j_leak = params.k_leak * max(ca_er - ca_cyt, 0.0)
        p_open = _ip3r_open_prob(ca_cyt, ip3, params)
        j_ip3r = params.v_ip3r * p_open * max(ca_er - ca_cyt, 0.0) / max(params.ca_er_0, 1e-9)

        d_ca_cyt = j_in + j_ip3r + j_leak - j_serca - j_pmca
        d_ca_er = -j_ip3r - j_leak + j_serca

        d_ip3 = params.v_plc * s - params.k_ip3_deg * ip3

        cam_on = params.cam_k_on * (max(ca_cyt, 0.0) ** params.cam_hill)
        d_cam = cam_on * (1.0 - cam_a) - params.cam_k_off * cam_a

        pkc_on = params.pkc_k_on * (max(ca_cyt, 0.0) ** params.pkc_hill) * (s / max(stim.amplitude, 1e-12))
        d_pkc = pkc_on * (1.0 - pkc_a) - params.pkc_k_off * pkc_a

        return np.array([d_ca_cyt, d_ca_er, d_ip3, d_cam, d_pkc], dtype=float)

    sol = solve_ivp(rhs, (0.0, t_end_s), y0, t_eval=t_eval, method=method, rtol=1e-6, atol=1e-9)
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    t = sol.t
    ca_cyt = sol.y[0]
    ca_er = sol.y[1]
    ip3 = sol.y[2]
    cam_a = np.clip(sol.y[3], 0.0, 1.0)
    pkc_a = np.clip(sol.y[4], 0.0, 1.0)

    stim_vec = np.array([stimulus_value(float(tt), stim) for tt in t], dtype=float)

    return {
        "t_s": t,
        "stim": stim_vec,
        "ca_cyt_uM": ca_cyt,
        "ca_er_uM": ca_er,
        "ip3_uM": ip3,
        "cam_active": cam_a,
        "pkc_active": pkc_a,
    }
