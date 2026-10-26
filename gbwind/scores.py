"""Proper scores and calibration diagnostics for ensembles."""

from __future__ import annotations

import numpy as np


def crps_ensemble(ens: np.ndarray, obs: np.ndarray | float) -> np.ndarray:
    """CRPS of the empirical ensemble distribution. ens has members on the last axis.

    CRPS = E|X - y| - 0.5 E|X - X'|, with the pair term from the sorted ensemble.
    """
    ens = np.asarray(ens, dtype=float)
    obs = np.asarray(obs, dtype=float)
    m = ens.shape[-1]
    term1 = np.mean(np.abs(ens - obs[..., None]), axis=-1)
    xs = np.sort(ens, axis=-1)
    w = 2.0 * np.arange(1, m + 1) - m - 1
    term2 = np.sum(w * xs, axis=-1) / m**2
    return term1 - term2


def energy_score(ens: np.ndarray, obs: np.ndarray) -> float:
    """ens (members, dims), obs (dims)."""
    ens = np.asarray(ens, dtype=float)
    d_obs = np.linalg.norm(ens - obs[None, :], axis=1).mean()
    diff = ens[:, None, :] - ens[None, :, :]
    d_pair = np.linalg.norm(diff, axis=2).mean()
    return float(d_obs - 0.5 * d_pair)


def variogram_score(ens: np.ndarray, obs: np.ndarray, p: float = 0.5, weights: np.ndarray | None = None) -> float:
    """Scheuerer and Hamill (2015). ens (members, dims), obs (dims)."""
    ens = np.asarray(ens, dtype=float)
    d = ens.shape[1]
    iu = np.triu_indices(d, 1)
    v_obs = np.abs(obs[:, None] - obs[None, :]) ** p
    v_ens = (np.abs(ens[:, :, None] - ens[:, None, :]) ** p).mean(axis=0)
    w = np.ones((d, d)) if weights is None else weights
    return float(np.sum(w[iu] * (v_obs[iu] - v_ens[iu]) ** 2))


def central_interval_hit(ens: np.ndarray, obs: float, level: float) -> bool:
    lo, hi = np.quantile(ens, [(1 - level) / 2, (1 + level) / 2])
    return bool(lo <= obs <= hi)


def rank_of_obs(ens: np.ndarray, obs: float, rng: np.random.Generator) -> int:
    """Rank 0..m of the observation in the ensemble, ties broken at random."""
    below = int(np.sum(ens < obs))
    ties = int(np.sum(ens == obs))
    return below + int(rng.integers(0, ties + 1))
