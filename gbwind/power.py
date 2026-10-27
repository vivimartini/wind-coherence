"""Wind to power. 100 m wind goes straight into the curve. Hub-height shear is only used when 100 m wind is missing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gbwind import config

CURVES_PATH = Path(__file__).resolve().parents[1] / "config" / "power_curves.json"

MIN_SHEAR, MAX_SHEAR, NEUTRAL_SHEAR = 0.0, 0.6, 1.0 / 7.0


def hub_wind_speed(u100, v100, u10, v10, hub_height_m):
    """Power law extrapolation from 100 m, with the exponent estimated from 10 m and 100 m.

    Where 10 m wind is too weak to give a stable exponent, the neutral 1/7 is used.
    """
    ws100 = np.hypot(u100, v100)
    ws10 = np.hypot(u10, v10)
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.log(ws100 / ws10) / np.log(10.0)
    alpha = np.where((ws10 > 0.5) & np.isfinite(alpha), alpha, NEUTRAL_SHEAR)
    alpha = np.clip(alpha, MIN_SHEAR, MAX_SHEAR)
    return ws100 * (np.asarray(hub_height_m) / 100.0) ** alpha


def curve_wind_speed(u100, v100, u10, v10, hub_height_m):
    """100 m speed, or 10 m extrapolated with the neutral one-seventh power."""
    speed = np.hypot(np.asarray(u100, dtype=float), np.asarray(v100, dtype=float))
    have = np.isfinite(speed)
    if np.all(have):
        return speed
    ws10 = np.hypot(np.asarray(u10, dtype=float), np.asarray(v10, dtype=float))
    fallback = ws10 * (np.asarray(hub_height_m, dtype=float) / 10.0) ** NEUTRAL_SHEAR
    return np.where(have, speed, fallback)


def smooth_capacity_factor(ws, cut_in=config.CUT_IN_MS, rated=config.RATED_MS, sharpness=6.0):
    """Logistic curve. No cut-out: a hard zero would block gradients, and almost no hours reach 25 m/s."""
    ws = np.asarray(ws, dtype=float)
    midpoint = 0.5 * (cut_in + rated)
    scale = (rated - cut_in) / sharpness
    z = (ws - midpoint) / scale
    return 1.0 / (1.0 + np.exp(-z))


def capacity_factor(ws, cut_in=config.CUT_IN_MS, rated=config.RATED_MS, cut_out=config.CUT_OUT_MS):
    ws = np.asarray(ws, dtype=float)
    ramp = (ws**3 - cut_in**3) / (rated**3 - cut_in**3)
    cf = np.where(ws < cut_in, 0.0, np.where(ws < rated, ramp, 1.0))
    return np.where(ws >= cut_out, 0.0, cf)


def load_curves(path: Path = CURVES_PATH) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("farms", {})


def farm_power_mw(u100, v100, u10, v10, hub_height_m, capacity_mw, cut_in=None, rated=None, sharpness=None):
    """MW. Pass cut-in, rated and sharpness for a fitted farm; otherwise the idealised curve."""
    ws = curve_wind_speed(u100, v100, u10, v10, hub_height_m)
    capacity = np.asarray(capacity_mw, dtype=float)
    if cut_in is None:
        return capacity_factor(ws) * capacity
    fitted = smooth_capacity_factor(ws, cut_in, rated, sharpness) * capacity
    use = np.isfinite(np.asarray(cut_in, dtype=float))
    return np.where(use, fitted, capacity_factor(ws) * capacity)
