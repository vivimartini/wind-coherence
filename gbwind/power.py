"""Wind to power. One idealised curve, fixed in advance, nothing fitted."""

from __future__ import annotations

import numpy as np

from gbwind import config

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


def smooth_capacity_factor(ws, cut_in=config.CUT_IN_MS, rated=config.RATED_MS, sharpness=6.0):
    """Differentiable stand-in for the power curve, used only on the training path.

    A logistic rises through the cubic region and saturates at rated speed. Cut-out
    is omitted: the hard drop to zero would block gradients, and most of these
    farms rarely reach it. The exact curve in ``capacity_factor`` stays the
    evaluation curve. Both must stay fixed across experimental arms.
    """
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


def farm_power_mw(u100, v100, u10, v10, hub_height_m, capacity_mw):
    return capacity_factor(hub_wind_speed(u100, v100, u10, v10, hub_height_m)) * np.asarray(capacity_mw)
