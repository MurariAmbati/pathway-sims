from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


@dataclass(frozen=True)
class SignalMetrics:
    baseline: float
    peak: float
    peak_t_s: float
    delta_peak: float
    auc: float
    min_value: float
    min_t_s: float
    rise_10_90_s: Optional[float]
    decay_tau_s: Optional[float]
    n_peaks: int
    mean_peak_period_s: Optional[float]
    mean_peak_frequency_hz: Optional[float]
    dominant_fft_hz: Optional[float]


def _exp_decay(t, a, tau, c):
    return a * np.exp(-t / tau) + c


def compute_signal_metrics(
    t_s: np.ndarray,
    x: np.ndarray,
    *,
    stim_start_s: float,
    stim_end_s: float,
    baseline_window_s: float = 3.0,
    peak_prominence: Optional[float] = None,
) -> SignalMetrics:
    t_s = np.asarray(t_s, dtype=float)
    x = np.asarray(x, dtype=float)

    if t_s.size != x.size or t_s.size < 5:
        raise ValueError("t_s and x must have same length and be >= 5")

    pre_mask = t_s < max(0.0, stim_start_s)
    if pre_mask.sum() >= 3:
        t0 = max(0.0, stim_start_s - baseline_window_s)
        base_mask = (t_s >= t0) & (t_s < stim_start_s)
        baseline = float(np.mean(x[base_mask])) if base_mask.sum() >= 3 else float(np.mean(x[pre_mask]))
    else:
        baseline = float(np.mean(x[: min(10, x.size)]))

    peak_idx = int(np.argmax(x))
    peak = float(x[peak_idx])
    peak_t_s = float(t_s[peak_idx])

    min_idx = int(np.argmin(x))
    min_value = float(x[min_idx])
    min_t_s = float(t_s[min_idx])

    delta_peak = peak - baseline

    stim_mask = (t_s >= stim_start_s) & (t_s <= stim_end_s)
    if stim_mask.sum() >= 2:
        auc = float(np.trapz(np.maximum(0.0, x[stim_mask] - baseline), t_s[stim_mask]))
    else:
        auc = float(np.trapz(np.maximum(0.0, x - baseline), t_s))

    rise_10_90_s = None
    if delta_peak > 0:
        lo = baseline + 0.1 * delta_peak
        hi = baseline + 0.9 * delta_peak
        post_mask = t_s >= stim_start_s
        tt, xx = t_s[post_mask], x[post_mask]
        try:
            t_lo = float(tt[np.where(xx >= lo)[0][0]])
            t_hi = float(tt[np.where(xx >= hi)[0][0]])
            rise_10_90_s = max(0.0, t_hi - t_lo)
        except Exception:
            rise_10_90_s = None

    decay_tau_s = None
    if peak_idx < x.size - 6:
        fit_t = t_s[peak_idx:]
        fit_x = x[peak_idx:]
        if np.percentile(fit_x, 90) > np.percentile(fit_x, 10):
            try:
                t_rel = fit_t - fit_t[0]
                a0 = float(fit_x[0] - fit_x[-1])
                tau0 = max(0.5, 0.2 * (fit_t[-1] - fit_t[0]))
                c0 = float(fit_x[-1])
                popt, _ = curve_fit(
                    _exp_decay,
                    t_rel,
                    fit_x,
                    p0=(a0, tau0, c0),
                    bounds=((-np.inf, 1e-6, -np.inf), (np.inf, np.inf, np.inf)),
                    maxfev=4000,
                )
                tau = float(popt[1])
                decay_tau_s = tau if np.isfinite(tau) else None
            except Exception:
                decay_tau_s = None

    if peak_prominence is None:
        peak_prominence = max(1e-12, 0.2 * float(np.std(x)))
    peaks, _ = find_peaks(x, prominence=peak_prominence)
    n_peaks = int(peaks.size)

    mean_peak_period_s = None
    mean_peak_frequency_hz = None
    if n_peaks >= 2:
        peak_times = t_s[peaks]
        periods = np.diff(peak_times)
        if periods.size and np.all(periods > 0):
            mean_peak_period_s = float(np.mean(periods))
            mean_peak_frequency_hz = float(1.0 / mean_peak_period_s) if mean_peak_period_s > 0 else None

    dominant_fft_hz = None
    dt = float(np.median(np.diff(t_s)))
    if dt > 0 and t_s.size >= 16:
        y = x - float(np.mean(x))
        freqs = np.fft.rfftfreq(y.size, d=dt)
        spec = np.abs(np.fft.rfft(y))
        if freqs.size >= 3:
            idx = int(np.argmax(spec[1:]) + 1)
            f = float(freqs[idx])
            dominant_fft_hz = f if f > 0 else None

    return SignalMetrics(
        baseline=baseline,
        peak=peak,
        peak_t_s=peak_t_s,
        delta_peak=delta_peak,
        auc=auc,
        min_value=min_value,
        min_t_s=min_t_s,
        rise_10_90_s=rise_10_90_s,
        decay_tau_s=decay_tau_s,
        n_peaks=n_peaks,
        mean_peak_period_s=mean_peak_period_s,
        mean_peak_frequency_hz=mean_peak_frequency_hz,
        dominant_fft_hz=dominant_fft_hz,
    )


def metrics_table(df: pd.DataFrame, *, stim_start_s: float, stim_end_s: float) -> pd.DataFrame:
    t = df["t_s"].to_numpy()
    rows: list[dict] = []
    for col in [
        "stim",
        "ca_cyt_uM",
        "ca_er_uM",
        "ip3_uM",
        "cam_active",
        "pkc_active",
        "ip3r_flux_uM_s",
        "pmca_flux_uM_s",
        "serca_flux_uM_s",
    ]:
        m = compute_signal_metrics(t, df[col].to_numpy(), stim_start_s=stim_start_s, stim_end_s=stim_end_s)
        rows.append(
            {
                "signal": col,
                "baseline": m.baseline,
                "peak": m.peak,
                "peak_t_s": m.peak_t_s,
                "delta_peak": m.delta_peak,
                "auc": m.auc,
                "min": m.min_value,
                "min_t_s": m.min_t_s,
                "rise_10_90_s": m.rise_10_90_s,
                "decay_tau_s": m.decay_tau_s,
                "n_peaks": m.n_peaks,
                "mean_peak_period_s": m.mean_peak_period_s,
                "mean_peak_freq_hz": m.mean_peak_frequency_hz,
                "dominant_fft_hz": m.dominant_fft_hz,
            }
        )
    return pd.DataFrame(rows)
