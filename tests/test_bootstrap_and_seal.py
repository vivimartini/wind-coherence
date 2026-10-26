import numpy as np
import pytest

from gbwind import bootstrap, seal
from gbwind.config import Paths


def test_blocks_are_consecutive_and_circular():
    idx = bootstrap.block_indices(10, 4, 50, np.random.default_rng(0))
    assert idx.shape == (50, 10)
    for row in idx:
        for b in range(0, 8, 4):
            blk = row[b:b + 4]
            assert np.all((np.diff(blk) % 10) == 1)


def test_difference_interval_covers_true_shift():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(200, 1))
    treat = ref - 0.3 + rng.normal(scale=0.05, size=(200, 20))
    r = bootstrap.difference_interval(treat, ref, 5, 1000, np.random.default_rng(2))
    assert r["lo"] < -0.3 < r["hi"]
    assert r["hi"] < 0


def test_realisation_variability_widens_interval():
    rng = np.random.default_rng(3)
    ref = rng.normal(size=(100, 1))
    tight = ref + rng.normal(scale=0.01, size=(100, 1)).repeat(20, axis=1)
    loose = ref + rng.normal(scale=0.5, size=(100, 20))
    a = bootstrap.difference_interval(tight, ref, 5, 1000, np.random.default_rng(4))
    b = bootstrap.difference_interval(loose, ref, 5, 1000, np.random.default_rng(4))
    assert (b["hi"] - b["lo"]) > (a["hi"] - a["lo"])


def test_sealed_period_needs_flag_and_opens_once(tmp_path):
    paths = Paths(data=tmp_path / "data", results=tmp_path / "results")
    with pytest.raises(seal.SealError):
        seal.open_period(paths, "test", "2026-09-15", confirm=False)
    rec = seal.open_period(paths, "test", "2026-09-15", confirm=True)
    assert rec["end_date"] == "2026-09-15" and seal.is_opened(paths, "test")
    with pytest.raises(seal.SealError):
        seal.open_period(paths, "test", "2026-09-15", confirm=True)


def test_unsealed_periods_open_freely(tmp_path):
    paths = Paths(data=tmp_path / "data", results=tmp_path / "results")
    assert seal.open_period(paths, "dev", "x", confirm=False)["sealed"] is False
