"""The four couplings. Each takes an ensemble X of shape (members, farms) and
returns an array whose every column is a permutation of the same column of X.

That property is the whole experiment: marginals fixed, pairing changed.
"""

from __future__ import annotations

import numpy as np


def random_tiebreak_ranks(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Ranks 0..n-1 of a 1-d array, ties broken uniformly at random."""
    order = np.lexsort((rng.random(values.shape[0]), values))
    ranks = np.empty(values.shape[0], dtype=np.int64)
    ranks[order] = np.arange(values.shape[0])
    return ranks


def native(X: np.ndarray) -> np.ndarray:
    return np.array(X, copy=True)


def independent(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.empty_like(X)
    for f in range(X.shape[1]):
        out[:, f] = X[rng.permutation(X.shape[0]), f]
    return out


def reorder_to_template(X: np.ndarray, template: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Schaake shuffle. Column f of the output carries the rank pattern of template[:, f]."""
    if X.shape != template.shape:
        raise ValueError(f"ensemble {X.shape} and template {template.shape} differ in shape")
    out = np.empty_like(X)
    for f in range(X.shape[1]):
        out[:, f] = np.sort(X[:, f])[random_tiebreak_ranks(template[:, f], rng)]
    return out


def historical(X: np.ndarray, history_rows: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """history_rows: (members, farms) metered outputs at the drawn template timestamps."""
    return reorder_to_template(X, history_rows, rng)


def cross_model(X: np.ndarray, other: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Ranks from the other system's native ensemble at the same farms and valid time."""
    return reorder_to_template(X, other, rng)


def tie_fraction(template: np.ndarray) -> np.ndarray:
    """Per column fraction of values that share their value with another value."""
    out = np.empty(template.shape[1])
    for f in range(template.shape[1]):
        _, inv, counts = np.unique(template[:, f], return_inverse=True, return_counts=True)
        out[f] = np.mean(counts[inv] > 1)
    return out


def mean_pairwise_spearman(X: np.ndarray) -> float:
    """Realised average cross farm rank correlation of a coupled ensemble."""
    if X.shape[1] < 2:
        return float("nan")
    ranks = np.argsort(np.argsort(X, axis=0, kind="stable"), axis=0).astype(float)
    ranks -= ranks.mean(axis=0)
    sd = ranks.std(axis=0)
    keep = sd > 0
    if keep.sum() < 2:
        return float("nan")
    r = ranks[:, keep] / sd[keep]
    c = (r.T @ r) / r.shape[0]
    iu = np.triu_indices(c.shape[0], 1)
    return float(np.mean(c[iu]))
