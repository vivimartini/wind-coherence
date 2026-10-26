"""Leakage and admissibility of the historical template (DESIGN.md section 4)."""

import numpy as np
import pandas as pd
import pytest

from gbwind import config
from gbwind.template import TemplateHistory, is_available, publication_date

FARMS = ["a", "b", "c"]


@pytest.fixture(scope="module")
def history():
    t = pd.date_range("2021-01-01", "2026-07-15", freq="h")
    rng = np.random.default_rng(0)
    values = pd.DataFrame(rng.random((len(t), 3)) * 100, index=t, columns=FARMS)
    adm = pd.DataFrame(True, index=t, columns=FARMS)
    adm.iloc[rng.random(len(t)) < 0.1, 1] = False
    return TemplateHistory(values, adm, FARMS), values, adm


def test_publication_lag_counts_working_days_and_bank_holidays():
    # Monday 25 May 2026 is the Spring bank holiday.
    assert publication_date(np.array(["2026-05-22"], dtype="datetime64[D]"))[0] == np.datetime64("2026-06-01")
    assert publication_date(np.array(["2026-05-21"], dtype="datetime64[D]"))[0] == np.datetime64("2026-05-29")


def test_issue_time_boundary():
    issue = pd.Timestamp("2026-06-01 00:00")
    assert is_available(pd.DatetimeIndex(["2026-05-21 12:00"]), issue)[0]
    assert not is_available(pd.DatetimeIndex(["2026-05-22 12:00"]), issue)[0]


def test_delivery_date_is_local_settlement_date():
    # 23:00 UTC in summer covers 22:30 to 23:30 UTC, whose later half is 00:00 BST on the next day.
    issue = pd.Timestamp("2026-06-01 00:00")
    assert is_available(pd.DatetimeIndex(["2026-05-21 22:00"]), issue)[0]
    assert not is_available(pd.DatetimeIndex(["2026-05-21 23:00"]), issue)[0]


@pytest.mark.parametrize("issue", pd.date_range("2025-12-15", "2026-07-10", freq="9D"))
def test_pool_never_contains_unpublished_or_future_data(history, issue):
    h, _, _ = history
    for lead in config.LEAD_HOURS_PRIMARY:
        target = issue + pd.Timedelta(hours=lead)
        r = h.pool_with_widening(target, issue)
        assert np.all(h.published[r.indices] < np.datetime64(issue.date(), "D"))
        assert np.all(h.times[r.indices] < issue)
        assert is_available(h.times[r.indices], issue).all()


def test_draws_pass_the_runtime_leak_assertion(history):
    h, _, _ = history
    issue = pd.Timestamp("2026-06-10")
    r = h.pool_with_widening(issue + pd.Timedelta(hours=24), issue)
    drawn = h.draw(r.indices, 5, config.N_MEMBERS, np.random.default_rng(0))
    h.assert_no_leak(drawn, issue)


def test_runtime_leak_assertion_fires(history):
    h, _, _ = history
    issue = pd.Timestamp("2026-06-10")
    recent = np.flatnonzero(h.times >= pd.Timestamp("2026-06-08"))[: config.N_MEMBERS]
    with pytest.raises(AssertionError):
        h.assert_no_leak(recent[None, :], issue)


def test_same_time_of_day(history):
    h, _, _ = history
    target = pd.Timestamp("2026-06-02 06:00")
    idx = h.pool(target, pd.Timestamp("2026-06-01"), 21)
    assert idx.size > 0
    assert set(h.times[idx].hour) == {6}


def test_seasonal_window_wraps_across_new_year(history):
    h, _, _ = history
    target = pd.Timestamp("2026-01-03 12:00")
    idx = h.pool(target, pd.Timestamp("2026-01-02"), 21)
    months = set(h.times[idx].month)
    assert months == {12, 1}
    doy = h.times[idx].dayofyear.to_numpy()
    d = np.minimum(np.abs(doy - target.dayofyear), 365.25 - np.abs(doy - target.dayofyear))
    assert d.max() <= 21


def test_timestamp_with_one_inadmissible_farm_is_dropped(history):
    h, values, adm = history
    bad = adm.index[~adm.all(axis=1)]
    assert len(bad) > 0
    assert not np.isin(bad.values, h.times.values).any()
    assert len(h) == int(adm.all(axis=1).sum())


def test_pool_is_widened_not_portfolio_shrunk():
    t = pd.date_range("2025-01-01", "2026-06-30", freq="h")
    values = pd.DataFrame(1.0, index=t, columns=FARMS)
    adm = pd.DataFrame(True, index=t, columns=FARMS)
    # Leave only a sparse scatter of admissible timestamps so 21 days cannot fill 51.
    adm.iloc[np.arange(len(t)) % 7 != 0] = False
    h = TemplateHistory(values, adm, FARMS)
    issue = pd.Timestamp("2026-06-10")
    r = h.pool_with_widening(issue + pd.Timedelta(hours=24), issue)
    assert r.base_size < config.N_MEMBERS
    assert r.half_window_days > config.SEASONAL_HALF_WINDOW_DAYS
    assert r.indices.size >= config.N_MEMBERS or r.half_window_days == config.SEASONAL_HALF_WINDOW_MAX_DAYS
    assert h.farms == FARMS


def test_draws_are_without_replacement_and_seeded(history):
    h, _, _ = history
    issue = pd.Timestamp("2026-06-10")
    r = h.pool_with_widening(issue + pd.Timedelta(hours=30), issue)
    a = h.draw(r.indices, 4, config.N_MEMBERS, np.random.default_rng(3))
    b = h.draw(r.indices, 4, config.N_MEMBERS, np.random.default_rng(3))
    assert np.array_equal(a, b)
    assert all(len(set(row)) == config.N_MEMBERS for row in a)
