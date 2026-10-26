"""Circular block bootstrap over issue days."""

from __future__ import annotations

import numpy as np


def block_indices(n_days: int, block_length: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """(n_boot, n_days) day indices built from circular blocks of consecutive days."""
    if n_days == 0:
        return np.zeros((n_boot, 0), dtype=int)
    block_length = max(1, min(block_length, n_days))
    n_blocks = int(np.ceil(n_days / block_length))
    starts = rng.integers(0, n_days, size=(n_boot, n_blocks))
    offsets = np.arange(block_length)
    idx = (starts[:, :, None] + offsets[None, None, :]) % n_days
    return idx.reshape(n_boot, -1)[:, :n_days]


def mean_with_realisations(
    day_scores: np.ndarray, idx: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Replicate means where each resampled day also draws one coupling realisation.

    day_scores is (days, realisations); NaN realisations are never chosen.
    """
    n_boot, n = idx.shape
    k = day_scores.shape[1]
    if k == 1:
        return np.nanmean(day_scores[idx, 0], axis=1)
    pick = rng.integers(0, k, size=(n_boot, n))
    return np.nanmean(day_scores[idx, pick], axis=1)


def difference_interval(
    treat: np.ndarray,
    ref: np.ndarray,
    block_length: int,
    n_boot: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> dict:
    """Mean of treat minus ref over days, with a block bootstrap percentile interval.

    treat and ref are (days, realisations) aligned on days. Days where either is
    entirely missing are dropped first.
    """
    ok = np.isfinite(treat).any(axis=1) & np.isfinite(ref).any(axis=1)
    treat, ref = treat[ok], ref[ok]
    n = treat.shape[0]
    if n == 0:
        return {"estimate": np.nan, "lo": np.nan, "hi": np.nan, "n_days": 0}
    est = float(np.mean(np.nanmean(treat, axis=1) - np.nanmean(ref, axis=1)))
    idx = block_indices(n, block_length, n_boot, rng)
    reps = mean_with_realisations(treat, idx, rng) - mean_with_realisations(ref, idx, rng)
    lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
    return {"estimate": est, "lo": float(lo), "hi": float(hi), "n_days": int(n)}


def mean_interval(scores: np.ndarray, block_length: int, n_boot: int, rng: np.random.Generator, alpha: float = 0.05) -> dict:
    ok = np.isfinite(scores).any(axis=1)
    scores = scores[ok]
    if scores.shape[0] == 0:
        return {"estimate": np.nan, "lo": np.nan, "hi": np.nan, "n_days": 0}
    idx = block_indices(scores.shape[0], block_length, n_boot, rng)
    reps = mean_with_realisations(scores, idx, rng)
    lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
    return {"estimate": float(np.nanmean(scores)), "lo": float(lo), "hi": float(hi), "n_days": int(scores.shape[0])}
