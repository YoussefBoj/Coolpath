from __future__ import annotations

import numpy as np
from scipy import stats


def exponential_history(values, dt_s: float, tau_s: float) -> np.ndarray:
    """Causal exponential moving average for regularly sampled data."""
    x = np.asarray(values, float)
    out = np.empty_like(x)
    alpha = np.exp(-dt_s / tau_s)
    state = x[0] if len(x) and np.isfinite(x[0]) else np.nanmean(x)
    for i, value in enumerate(x):
        if np.isfinite(value):
            state = alpha * state + (1.0 - alpha) * value
        out[i] = state
    return out


def exponential_history_variable_dt(values, t_s, tau_s: float) -> np.ndarray:
    """Causal EMA using the real time gap between consecutive panorama frames."""
    x = np.asarray(values, float)
    t = np.asarray(t_s, float)
    out = np.empty_like(x)
    if len(x) == 0:
        return out
    state = x[0] if np.isfinite(x[0]) else np.nanmean(x)
    out[0] = state
    for i in range(1, len(x)):
        alpha = np.exp(-max(t[i] - t[i - 1], 0.0) / tau_s)
        if np.isfinite(x[i]):
            state = alpha * state + (1.0 - alpha) * x[i]
        out[i] = state
    return out


def apply_lag(series, t_s, lag_s: float) -> np.ndarray:
    """Value of a time series at t-lag, using linear interpolation."""
    t = np.asarray(t_s, float)
    x = np.asarray(series, float)
    return np.interp(t - lag_s, t, x)


def detrend(values, t):
    values = np.asarray(values, float)
    t = np.asarray(t, float)
    ok = np.isfinite(values) & np.isfinite(t)
    out = np.full_like(values, np.nan)
    if ok.sum() < 3:
        return out
    coef = np.polyfit(t[ok], values[ok], 1)
    out[ok] = values[ok] - np.polyval(coef, t[ok])
    return out


def calibrate_globe_inertia(
    kdown,
    tg,
    dt_s,
    taus=(30, 60, 120, 180, 240, 300, 420),
    lags=(0, 30, 60, 90, 120, 180),
):
    """Grid-search tau/lag by maximizing detrended r(Kdown_history, Tg)."""
    kdown = np.asarray(kdown, float)
    tg = np.asarray(tg, float)
    t = np.arange(len(kdown)) * dt_s
    rows = []
    best = None
    for tau in taus:
        hist = exponential_history(kdown, dt_s, tau)
        for lag in lags:
            shifted = apply_lag(hist, t, lag)
            x = detrend(shifted, t)
            y = detrend(tg, t)
            ok = np.isfinite(x) & np.isfinite(y)
            r = float(stats.pearsonr(x[ok], y[ok]).statistic) if ok.sum() >= 3 else np.nan
            row = {"tau_s": tau, "lag_s": lag, "r": r}
            rows.append(row)
            if np.isfinite(r) and (best is None or r > best["r"]):
                best = row
    return best, rows


def unreliable_history_windows(t_s, tau_s, gap_max_s=20.0, fabricated_weight_limit=0.20):
    """Intervals where a large video gap would fabricate too much history.

    This preserves the notebook-30 safeguard: after a gap, history is marked
    unreliable while the artificial weight assigned to the first observed
    morphology remains above the configured threshold.
    """
    t = np.asarray(t_s, float)
    windows = []
    for i, dt in enumerate(np.diff(t), start=1):
        if dt <= gap_max_s:
            continue
        initial_weight = 1.0 - np.exp(-dt / tau_s)
        if initial_weight <= fabricated_weight_limit:
            continue
        duration = tau_s * np.log(initial_weight / fabricated_weight_limit)
        windows.append((t[i], t[i] + duration))
    return windows


def reliable_after_gaps(t_query, windows, lag_s=0.0):
    """Boolean reliability mask, shifted consistently with the applied lag."""
    t_query = np.asarray(t_query, float)
    reliable = np.ones(len(t_query), dtype=bool)
    for start, stop in windows:
        reliable &= ~((t_query >= start + lag_s) & (t_query < stop + lag_s))
    return reliable
