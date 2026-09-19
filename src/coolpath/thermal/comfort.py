from __future__ import annotations
import numpy as np

SIGMA_SB = 5.670374419e-8


def tmrt_iso7726(Tg, Ta, va, diameter=.04, emissivity=.95):
    Tg = np.asarray(Tg, float)
    Ta = np.asarray(Ta, float)
    va = np.maximum(np.asarray(va, float), 1e-3)
    h_free = 1.4 * (np.abs(Tg - Ta) / diameter)**0.25
    h_forced = 6.3 * (va**0.6) / (diameter**0.4)
    h = np.maximum(h_free, h_forced)
    t4 = (Tg + 273.15)**4 + h / (emissivity * SIGMA_SB) * (Tg - Ta)
    return np.maximum(t4, 0)**0.25 - 273.15


def wind_to_10m(ws, z=1.5, z0=.01):
    ws = np.asarray(ws, float)
    return ws * np.log(10 / z0) / np.log(z / z0)


def compute_utci(ta, tmrt, rh, v10):
    from pythermalcomfort.models import utci
    out = []
    for a, m, h, v in zip(np.asarray(ta), np.asarray(tmrt), np.asarray(rh),
                          np.asarray(v10)):
        if not np.all(np.isfinite([a, m, h, v])):
            out.append(np.nan)
            continue
        value = utci(tdb=float(a),
                     tr=float(m),
                     v=float(np.clip(v, .5, 17)),
                     rh=float(h),
                     limit_inputs=True)
        out.append(float(getattr(value, 'utci', value)))
    return np.asarray(out, float)
