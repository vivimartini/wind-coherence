import numpy as np

from gbwind import scores


def brute_crps(x, y):
    x = np.asarray(x, float)
    return np.mean(np.abs(x - y)) - 0.5 * np.mean(np.abs(x[:, None] - x[None, :]))


def test_crps_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.normal(size=51)
        y = rng.normal()
        assert np.isclose(scores.crps_ensemble(x, y), brute_crps(x, y))


def test_crps_vectorised_over_leading_axes():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 3, 51))
    y = rng.normal(size=(4, 3))
    out = scores.crps_ensemble(x, y)
    assert out.shape == (4, 3)
    assert np.isclose(out[2, 1], brute_crps(x[2, 1], y[2, 1]))


def test_crps_point_ensemble_is_absolute_error():
    assert np.isclose(scores.crps_ensemble(np.full(51, 3.0), 5.0), 2.0)


def test_energy_score_in_one_dimension_equals_crps():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(51, 1))
    y = np.array([0.3])
    assert np.isclose(scores.energy_score(x, y), scores.crps_ensemble(x[:, 0], 0.3))


def test_variogram_score_zero_for_matching_point_ensemble():
    obs = np.array([0.1, 0.5, 0.9])
    assert np.isclose(scores.variogram_score(np.repeat(obs[None, :], 51, axis=0), obs), 0.0, atol=1e-20)


def test_variogram_score_prefers_correct_dependence():
    rng = np.random.default_rng(3)
    n, d = 400, 4
    cov = 0.9 * np.ones((d, d)) + 0.1 * np.eye(d)
    L = np.linalg.cholesky(cov)
    good, bad = 0.0, 0.0
    for _ in range(n):
        y = L @ rng.normal(size=d)
        ens_good = rng.normal(size=(51, d)) @ L.T
        ens_bad = rng.normal(size=(51, d))
        good += scores.variogram_score(ens_good, y)
        bad += scores.variogram_score(ens_bad, y)
    assert good < bad


def test_rank_of_obs_bounds_and_ties():
    rng = np.random.default_rng(4)
    ens = np.zeros(51)
    ranks = {scores.rank_of_obs(ens, 0.0, rng) for _ in range(500)}
    assert min(ranks) == 0 and max(ranks) == 51
    assert scores.rank_of_obs(np.arange(51.0), 100.0, rng) == 51


def test_interval_hit():
    ens = np.linspace(0, 1, 101)
    assert scores.central_interval_hit(ens, 0.5, 0.5)
    assert not scores.central_interval_hit(ens, 0.95, 0.5)
