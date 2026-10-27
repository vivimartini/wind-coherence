"""Per-farm CRPS is identical across couplings."""

import numpy as np
import pytest

from gbwind import couplings, scores

M, F = 51, 9


@pytest.fixture
def ens():
    rng = np.random.default_rng(0)
    X = rng.gamma(2.0, 100.0, size=(M, F))
    X[rng.random((M, F)) < 0.2] = 0.0  # ties at zero output
    X[:, 0] = np.minimum(X[:, 0], 250.0)  # ties at rated
    return X


@pytest.fixture
def template():
    rng = np.random.default_rng(1)
    T = rng.gamma(2.0, 100.0, size=(M, F))
    T[rng.random((M, F)) < 0.3] = 0.0
    return T


def all_couplings(X, T):
    rng = np.random.default_rng(2)
    other = np.random.default_rng(3).normal(size=X.shape)
    return {
        "native": couplings.native(X),
        "historical": couplings.historical(X, T, rng),
        "independent": couplings.independent(X, rng),
        "cross": couplings.cross_model(X, other, rng),
    }


def test_every_coupling_preserves_each_farm_multiset(ens, template):
    for name, Y in all_couplings(ens, template).items():
        assert np.array_equal(np.sort(Y, axis=0), np.sort(ens, axis=0)), name


def test_per_farm_crps_identical_across_couplings(ens, template):
    obs = np.random.default_rng(4).gamma(2.0, 100.0, size=F)
    ref = scores.crps_ensemble(ens.T, obs)
    for name, Y in all_couplings(ens, template).items():
        np.testing.assert_allclose(scores.crps_ensemble(Y.T, obs), ref, rtol=0, atol=1e-9, err_msg=name)


def test_portfolio_crps_does_change(ens, template):
    obs = np.full(F, 150.0)
    c = {k: float(scores.crps_ensemble(Y.sum(axis=1), obs.sum())) for k, Y in all_couplings(ens, template).items()}
    assert len({round(v, 6) for v in c.values()}) > 1


def test_schaake_imposes_template_ranks_without_ties():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(M, F))
    T = rng.normal(size=(M, F))
    Y = couplings.reorder_to_template(X, T, rng)
    assert np.array_equal(np.argsort(np.argsort(Y, axis=0), axis=0), np.argsort(np.argsort(T, axis=0), axis=0))


def test_schaake_reproducible_under_seed(ens, template):
    a = couplings.historical(ens, template, np.random.default_rng(9))
    b = couplings.historical(ens, template, np.random.default_rng(9))
    assert np.array_equal(a, b)


def test_ties_in_template_broken_at_random():
    X = np.arange(M * 2, dtype=float).reshape(M, 2)
    T = np.zeros((M, 2))
    outs = {tuple(couplings.reorder_to_template(X, T, np.random.default_rng(s))[:, 0]) for s in range(5)}
    assert len(outs) > 1


def test_perfectly_dependent_template_gives_comonotone_output(ens):
    T = np.repeat(np.arange(M, dtype=float)[:, None], F, axis=1)
    Y = couplings.reorder_to_template(ens + np.arange(F) * 1e-6, T, np.random.default_rng(0))
    assert couplings.mean_pairwise_spearman(Y) > 0.999


def test_tie_fraction():
    T = np.array([[0.0, 1.0], [0.0, 2.0], [5.0, 3.0], [6.0, 3.0]])
    np.testing.assert_allclose(couplings.tie_fraction(T), [0.5, 0.5])


def test_shape_mismatch_rejected(ens):
    with pytest.raises(ValueError):
        couplings.reorder_to_template(ens, ens[:10], np.random.default_rng(0))
