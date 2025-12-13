from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_period_fft(t: np.ndarray, y: np.ndarray) -> float | None:
    # returns dominant period in same time units, or none.
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size < 8:
        return None

    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return None

    y0 = y - np.mean(y)
    win = np.hanning(y0.size)
    yf = np.fft.rfft(y0 * win)
    freqs = np.fft.rfftfreq(y0.size, d=dt)

    power = (np.abs(yf) ** 2)
    if power.size < 3:
        return None

    # ignore dc component
    idx = int(np.argmax(power[1:]) + 1)
    f = float(freqs[idx])
    if f <= 0:
        return None
    return 1.0 / f


def summarize(df: pd.DataFrame) -> dict[str, float | None]:
    t = df["t"].to_numpy()
    nn = df["nn"].to_numpy()
    i = df["i"].to_numpy()

    burn = int(0.2 * len(df))
    nn2 = nn[burn:]
    i2 = i[burn:]

    out: dict[str, float | None] = {
        "nn_mean": float(np.mean(nn2)) if nn2.size else None,
        "nn_ptp": float(np.ptp(nn2)) if nn2.size else None,
        "i_mean": float(np.mean(i2)) if i2.size else None,
        "i_ptp": float(np.ptp(i2)) if i2.size else None,
        "period_fft": estimate_period_fft(t[burn:], nn2) if nn2.size > 8 else None,
    }

    return out
