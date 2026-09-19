import numpy as np
from coolpath.indicators.geometry import svf_row_weights, compute_svf, compute_gvi


def band(h, w, lo, hi):
    theta, _, _ = svf_row_weights(h)
    m = np.zeros((h, w), bool)
    m[(theta >= np.radians(lo)) & (theta <= np.radians(hi))] = True
    return m


def test_svf_analytic_cases():
    h, w = 720, 1440
    assert abs(compute_svf(band(h, w, 0, 90)) - 1) < 0.005
    assert abs(compute_svf(band(h, w, 0, 30)) - 0.25) < 0.01
    assert abs(compute_svf(band(h, w, 0, 45)) - 0.50) < 0.01
    assert abs(compute_svf(band(h, w, 0, 60)) - 0.75) < 0.01


def test_solid_angle_gvi_differs_near_pole():
    h, w = 180, 360
    m = np.zeros((h, w), bool)
    m[:20] = True
    raw, omega = compute_gvi(m, np.ones_like(m, bool))
    assert omega < raw
