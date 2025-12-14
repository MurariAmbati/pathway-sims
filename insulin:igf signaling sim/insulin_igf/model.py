from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


# --- Parameter containers -------------------------------------------------


@dataclass
class SimulationInputs:
    insulin_nM: float
    igf_nM: float
    stim_mode: str = "step"  # step | pulse | meals
    t0: float = 10.0
    duration: float = 30.0
    period: float = 120.0

    def model_dump(self) -> dict:
        return {
            "insulin_nM": float(self.insulin_nM),
            "igf_nM": float(self.igf_nM),
            "stim_mode": str(self.stim_mode),
            "t0": float(self.t0),
            "duration": float(self.duration),
            "period": float(self.period),
        }


@dataclass
class Params:
    # receptor activation
    K_insulin_IR: float = 2.5
    K_igf_IGF1R: float = 1.5
    hill_rec: float = 1.4

    k_IR_on: float = 1.0
    k_IR_off: float = 0.8

    k_IGF1R_on: float = 1.0
    k_IGF1R_off: float = 0.8

    # IRS and PI3K branch
    k_IRS_on: float = 1.2
    k_IRS_off: float = 0.9

    k_PI3K_on: float = 1.2
    k_PI3K_off: float = 1.0

    k_PIP3_on: float = 1.6
    k_PIP3_off: float = 0.6

    k_AKT_on: float = 1.3
    k_AKT_off: float = 0.9

    k_mTOR_on: float = 0.9
    k_mTOR_off: float = 0.6

    k_S6K_on: float = 1.0
    k_S6K_off: float = 0.8

    k_GLUT4_on: float = 1.0
    k_GLUT4_off: float = 0.7

    # MAPK branch
    k_Ras_on: float = 1.0
    k_Ras_off: float = 0.9

    k_RAF_on: float = 1.0
    k_RAF_off: float = 0.9

    k_MEK_on: float = 1.0
    k_MEK_off: float = 0.9

    k_ERK_on: float = 1.0
    k_ERK_off: float = 0.9

    # coupling / feedback
    mapk_crosstalk: float = 0.6  # IRS contribution to Ras

    feedback_strength: float = 1.0  # S6K inhibits IRS
    K_S6K_inhib: float = 0.35
    hill_fb: float = 2.0

    pten_strength: float = 1.0  # PTEN dephosphorylates PIP3
    K_PIP3_pten: float = 0.5

    # Glucose homeostasis (toy 1-compartment)
    glucose0_mM: float = 5.2

    hepatic_prod0_mM_min: float = 0.012
    hepatic_supp_AKT: float = 0.7  # fraction suppressible by AKT

    uptake0_mM_min: float = 0.010
    uptake_insulin_gain: float = 0.020  # max additional uptake
    glucose_Km_mM: float = 5.0

    def model_dump(self) -> dict:
        return {k: float(getattr(self, k)) for k in self.__dataclass_fields__.keys()}

    def model_copy(self) -> "Params":
        return Params(**self.model_dump())


DEFAULT_PARAMS = Params()


# --- Helpers --------------------------------------------------------------


def _hill(x: float, K: float, n: float) -> float:
    x = max(0.0, float(x))
    K = max(1e-12, float(K))
    n = max(1e-12, float(n))
    return (x**n) / (K**n + x**n)


def _stimulus_fn(inputs: SimulationInputs) -> Callable[[float], Tuple[float, float]]:
    insulin = float(inputs.insulin_nM)
    igf = float(inputs.igf_nM)
    mode = str(inputs.stim_mode)
    t0 = float(inputs.t0)
    dur = max(0.0, float(inputs.duration))
    period = max(1e-6, float(inputs.period))

    if mode == "step":
        def f(t: float) -> Tuple[float, float]:
            return (insulin if t >= t0 else 0.0, igf if t >= t0 else 0.0)

        return f

    if mode == "pulse":
        def f(t: float) -> Tuple[float, float]:
            on = (t >= t0) and (t <= (t0 + dur))
            return (insulin if on else 0.0, igf if on else 0.0)

        return f

    if mode == "meals":
        def f(t: float) -> Tuple[float, float]:
            if t < t0:
                return (0.0, 0.0)
            # repeating pulses
            phase = (t - t0) % period
            on = phase <= dur
            return (insulin if on else 0.0, igf if on else 0.0)

        return f

    raise ValueError(f"Unknown stim_mode: {mode}")


def _clamp01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


# --- ODE system -----------------------------------------------------------


STATE_NAMES = [
    "IR",
    "IGF1R",
    "IRS",
    "PI3K",
    "PIP3",
    "AKT",
    "mTORC1",
    "S6K",
    "GLUT4",
    "Ras",
    "RAF",
    "MEK",
    "ERK",
    "Glucose",
]


def _rhs(
    t: float,
    y: np.ndarray,
    params: Params,
    stim: Callable[[float], Tuple[float, float]],
    rates_out: Dict[str, float] | None = None,
) -> np.ndarray:
    # unpack
    (
        IR,
        IGF1R,
        IRS,
        PI3K,
        PIP3,
        AKT,
        mTORC1,
        S6K,
        GLUT4,
        Ras,
        RAF,
        MEK,
        ERK,
        Glucose,
    ) = y

    insulin_t, igf_t = stim(t)

    # receptors toward Hill targets
    IR_target = _hill(insulin_t, params.K_insulin_IR, params.hill_rec)
    IGF1R_target = _hill(igf_t, params.K_igf_IGF1R, params.hill_rec)

    dIR_act = params.k_IR_on * (IR_target) * (1.0 - IR)
    dIR_deact = params.k_IR_off * IR
    dIR = dIR_act - dIR_deact

    dIGF1R_act = params.k_IGF1R_on * (IGF1R_target) * (1.0 - IGF1R)
    dIGF1R_deact = params.k_IGF1R_off * IGF1R
    dIGF1R = dIGF1R_act - dIGF1R_deact

    # feedback: S6K inhibits IRS activation
    fb = 1.0 / (1.0 + params.feedback_strength * _hill(S6K, params.K_S6K_inhib, params.hill_fb))

    IRS_drive = _clamp01(0.65 * IR + 0.55 * IGF1R)
    dIRS_act = params.k_IRS_on * IRS_drive * fb * (1.0 - IRS)
    dIRS_deact = params.k_IRS_off * IRS
    dIRS = dIRS_act - dIRS_deact

    dPI3K_act = params.k_PI3K_on * IRS * (1.0 - PI3K)
    dPI3K_deact = params.k_PI3K_off * PI3K
    dPI3K = dPI3K_act - dPI3K_deact

    # PIP3 with PTEN term
    dPIP3_act = params.k_PIP3_on * PI3K * (1.0 - PIP3)
    dPIP3_deact = params.k_PIP3_off * PIP3
    dPIP3_pten = params.pten_strength * _hill(PIP3, params.K_PIP3_pten, 1.4) * 0.8 * PIP3
    dPIP3 = dPIP3_act - dPIP3_deact - dPIP3_pten

    dAKT_act = params.k_AKT_on * PIP3 * (1.0 - AKT)
    dAKT_deact = params.k_AKT_off * AKT
    dAKT = dAKT_act - dAKT_deact

    dmTOR_act = params.k_mTOR_on * AKT * (1.0 - mTORC1)
    dmTOR_deact = params.k_mTOR_off * mTORC1
    dmTOR = dmTOR_act - dmTOR_deact

    dS6K_act = params.k_S6K_on * mTORC1 * (1.0 - S6K)
    dS6K_deact = params.k_S6K_off * S6K
    dS6K = dS6K_act - dS6K_deact

    dGLUT4_act = params.k_GLUT4_on * AKT * (1.0 - GLUT4)
    dGLUT4_deact = params.k_GLUT4_off * GLUT4
    dGLUT4 = dGLUT4_act - dGLUT4_deact

    # MAPK branch driven by receptors + IRS crosstalk
    Ras_drive = _clamp01(0.45 * IR + 0.65 * IGF1R + params.mapk_crosstalk * IRS)
    dRas_act = params.k_Ras_on * Ras_drive * (1.0 - Ras)
    dRas_deact = params.k_Ras_off * Ras
    dRas = dRas_act - dRas_deact

    dRAF_act = params.k_RAF_on * Ras * (1.0 - RAF)
    dRAF_deact = params.k_RAF_off * RAF
    dRAF = dRAF_act - dRAF_deact

    dMEK_act = params.k_MEK_on * RAF * (1.0 - MEK)
    dMEK_deact = params.k_MEK_off * MEK
    dMEK = dMEK_act - dMEK_deact

    dERK_act = params.k_ERK_on * MEK * (1.0 - ERK)
    dERK_deact = params.k_ERK_off * ERK
    dERK = dERK_act - dERK_deact

    # Glucose homeostasis (very simplified)
    hepatic_prod = params.hepatic_prod0_mM_min * (1.0 - params.hepatic_supp_AKT * AKT)
    hepatic_prod = max(0.0, hepatic_prod)

    sat_g = Glucose / (params.glucose_Km_mM + max(1e-9, Glucose))
    uptake = params.uptake0_mM_min + params.uptake_insulin_gain * GLUT4 * sat_g

    dG = hepatic_prod - uptake

    if rates_out is not None:
        rates_out.update(
            {
                "IR_act_rate": dIR_act,
                "IR_deact_rate": dIR_deact,
                "IGF1R_act_rate": dIGF1R_act,
                "IGF1R_deact_rate": dIGF1R_deact,
                "IRS_act_rate": dIRS_act,
                "IRS_deact_rate": dIRS_deact,
                "IRS_feedback": fb,
                "PI3K_act_rate": dPI3K_act,
                "PI3K_deact_rate": dPI3K_deact,
                "PIP3_act_rate": dPIP3_act,
                "PIP3_deact_rate": dPIP3_deact,
                "PIP3_pten": dPIP3_pten,
                "AKT_act_rate": dAKT_act,
                "AKT_deact_rate": dAKT_deact,
                "mTORC1_act_rate": dmTOR_act,
                "mTORC1_deact_rate": dmTOR_deact,
                "S6K_act_rate": dS6K_act,
                "S6K_deact_rate": dS6K_deact,
                "GLUT4_act_rate": dGLUT4_act,
                "GLUT4_deact_rate": dGLUT4_deact,
                "Ras_act_rate": dRas_act,
                "Ras_deact_rate": dRas_deact,
                "RAF_act_rate": dRAF_act,
                "RAF_deact_rate": dRAF_deact,
                "MEK_act_rate": dMEK_act,
                "MEK_deact_rate": dMEK_deact,
                "ERK_act_rate": dERK_act,
                "ERK_deact_rate": dERK_deact,
                "HepaticProd_mM_min": hepatic_prod,
                "GlucoseUptake_mM_min": uptake,
            }
        )

    return np.array(
        [
            dIR,
            dIGF1R,
            dIRS,
            dPI3K,
            dPIP3,
            dAKT,
            dmTOR,
            dS6K,
            dGLUT4,
            dRas,
            dRAF,
            dMEK,
            dERK,
            dG,
        ],
        dtype=float,
    )


def _initial_state(params: Params) -> np.ndarray:
    # basal state, low signaling
    return np.array(
        [
            0.05,  # IR
            0.05,  # IGF1R
            0.05,  # IRS
            0.05,  # PI3K
            0.05,  # PIP3
            0.05,  # AKT
            0.05,  # mTORC1
            0.05,  # S6K
            0.05,  # GLUT4
            0.05,  # Ras
            0.05,  # RAF
            0.05,  # MEK
            0.05,  # ERK
            float(params.glucose0_mM),
        ],
        dtype=float,
    )


# --- Public API -----------------------------------------------------------


def simulate(t_end: float, dt: float, params_dict: dict, inputs_dict: dict) -> pd.DataFrame:
    params = Params(**params_dict)
    inputs = SimulationInputs(**inputs_dict)
    stim = _stimulus_fn(inputs)

    t_end = float(t_end)
    dt = max(1e-3, float(dt))

    t_eval = np.arange(0.0, t_end + 1e-9, dt)
    y0 = _initial_state(params)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        return _rhs(t, y, params, stim, rates_out=None)

    sol = solve_ivp(
        rhs,
        t_span=(0.0, t_end),
        y0=y0,
        t_eval=t_eval,
        method="LSODA",
        rtol=1e-6,
        atol=1e-9,
    )

    if not sol.success or sol.y.size == 0:
        return pd.DataFrame()

    df = pd.DataFrame(sol.y.T, columns=STATE_NAMES)
    df.insert(0, "t_min", sol.t)

    # clamp the activation states to keep visualization stable
    for name in STATE_NAMES:
        if name == "Glucose":
            continue
        df[name] = df[name].clip(lower=0.0, upper=1.0)

    # compute rates/terms at each output time (process explorer)
    rates_rows = []
    for i, t in enumerate(df["t_min"].to_numpy()):
        y = df.loc[i, STATE_NAMES].to_numpy(dtype=float)
        terms: Dict[str, float] = {}
        _rhs(float(t), y, params, stim, rates_out=terms)
        rates_rows.append(terms)

    terms_df = pd.DataFrame(rates_rows)
    df = pd.concat([df, terms_df], axis=1)

    # derived aliases
    df = df.rename(columns={"Glucose": "Glucose_mM"})

    # convenience series for plotting
    for n in [
        "IR",
        "IGF1R",
        "IRS",
        "PI3K",
        "PIP3",
        "AKT",
        "mTORC1",
        "S6K",
        "GLUT4",
        "Ras",
        "RAF",
        "MEK",
        "ERK",
    ]:
        df[f"{n}_act"] = df[n]

    return df


def sweep_dose_response(
    doses_nM: np.ndarray,
    igf_background_nM: float,
    params_dict: dict,
    settle_time_min: float = 90.0,
) -> pd.DataFrame:
    params = Params(**params_dict)

    rows = []
    for d in np.asarray(doses_nM, dtype=float):
        inputs = SimulationInputs(
            insulin_nM=float(d),
            igf_nM=float(igf_background_nM),
            stim_mode="step",
            t0=0.0,
            duration=0.0,
            period=120.0,
        )
        df = simulate(
            t_end=float(settle_time_min),
            dt=0.5,
            params_dict=params.model_dump(),
            inputs_dict=inputs.model_dump(),
        )
        if df.empty:
            continue
        last = df.iloc[-1]
        rows.append(
            {
                "insulin_nM": float(d),
                "IGF_background_nM": float(igf_background_nM),
                "AKT_act": float(last["AKT_act"]),
                "ERK_act": float(last["ERK_act"]),
                "GLUT4_act": float(last["GLUT4_act"]),
                "Glucose_mM": float(last["Glucose_mM"]),
            }
        )

    return pd.DataFrame(rows)
