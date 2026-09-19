from __future__ import annotations
import numpy as np


def angles_equirect(height: int, width: int):
    theta = (np.arange(height) + 0.5) / height * np.pi
    phi = (np.arange(width) + 0.5) / width * 2 * np.pi
    return theta, phi


def solid_angle_row_weights(height: int) -> np.ndarray:
    theta, _ = angles_equirect(height, 1)
    return np.sin(theta)


def svf_row_weights(height: int):
    theta, _ = angles_equirect(height, 1)
    upper = theta <= np.pi / 2
    weights = np.where(upper, np.cos(theta) * np.sin(theta), 0.0)
    return theta, weights, upper


def compute_svf(sky_mask: np.ndarray) -> float:
    h, w = sky_mask.shape
    _, weights, upper = svf_row_weights(h)
    frac = sky_mask[upper].sum(axis=1) / w
    return float(np.sum(frac * weights[upper]) / np.sum(weights[upper]))


def compute_gvi(mask: np.ndarray,
                valid_mask: np.ndarray | None = None) -> tuple[float, float]:
    if valid_mask is None: valid_mask = np.ones_like(mask, dtype=bool)
    raw = float(mask[valid_mask].mean() * 100) if valid_mask.any() else np.nan
    h, w = mask.shape
    weights = np.sin((np.arange(h) + 0.5) / h * np.pi)[:, None]
    weighted = float((mask * valid_mask * weights).sum() / max(
        (valid_mask * weights).sum(), 1e-12) * 100)
    return raw, weighted
