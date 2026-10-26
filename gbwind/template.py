"""Historical dependence template (DESIGN.md section 4).

A timestamp is admissible for a target only if
  1. it has the same UTC time of day as the target,
  2. its day of year is within the seasonal half window of the target's,
  3. every portfolio farm is metered and uncurtailed (enforced upstream through
     the admissibility table),
  4. its metered data was published before the issue time. B1610 publishes about
     five working days after delivery, so publication date = delivery date plus
     five England and Wales working days, and it must fall strictly before the
     issue date because issue is at 00 UTC.

If the pool is smaller than the ensemble, the seasonal window is widened, never
the portfolio shrunk.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import holidays
import numpy as np
import pandas as pd

from gbwind import config


@lru_cache(maxsize=1)
def _bank_holidays() -> np.ndarray:
    days = holidays.country_holidays("GB", subdiv="ENG", years=range(2015, 2031))
    return np.array(sorted(days), dtype="datetime64[D]")


def delivery_date(t: pd.DatetimeIndex) -> np.ndarray:
    """Local settlement date of the later settlement period of a centred hourly value at t."""
    last_sp = pd.DatetimeIndex(t) + pd.Timedelta(minutes=29)
    local = last_sp.tz_localize("UTC").tz_convert("Europe/London")
    return local.tz_localize(None).normalize().values.astype("datetime64[D]")


def publication_date(delivery: np.ndarray, lag_working_days: int = config.PUBLICATION_LAG_WORKING_DAYS) -> np.ndarray:
    return np.busday_offset(
        np.asarray(delivery, dtype="datetime64[D]"), lag_working_days, roll="forward", holidays=_bank_holidays()
    )


def is_available(t: pd.DatetimeIndex, issue_time: pd.Timestamp, lag_working_days: int = config.PUBLICATION_LAG_WORKING_DAYS) -> np.ndarray:
    issue_day = np.datetime64(pd.Timestamp(issue_time).normalize().date(), "D")
    return publication_date(delivery_date(t), lag_working_days) < issue_day


def _circular_doy_distance(doy: np.ndarray, target_doy: int) -> np.ndarray:
    d = np.abs(doy - target_doy)
    return np.minimum(d, 365.25 - d)


@dataclass
class PoolResult:
    indices: np.ndarray
    half_window_days: int
    base_size: int

    @property
    def feasible(self) -> bool:
        return self.indices.size >= config.N_MEMBERS


class TemplateHistory:
    """Metered history restricted to timestamps admissible at every farm."""

    def __init__(
        self,
        values: pd.DataFrame,
        admissible: pd.DataFrame,
        farms: list[str],
        lag_working_days: int = config.PUBLICATION_LAG_WORKING_DAYS,
    ):
        v = values.reindex(columns=farms)
        a = admissible.reindex(index=v.index, columns=farms).fillna(False).astype(bool)
        keep = a.all(axis=1) & v.notna().all(axis=1)
        self.farms = list(farms)
        self.times = pd.DatetimeIndex(v.index[keep.to_numpy()])
        self.values = v.loc[keep.to_numpy()].to_numpy(dtype=float)
        self.lag = lag_working_days
        self.published = publication_date(delivery_date(self.times), lag_working_days)
        self.doy = self.times.dayofyear.to_numpy()
        self.tod = (self.times.hour * 60 + self.times.minute).to_numpy()

    def __len__(self) -> int:
        return len(self.times)

    def pool(self, target: pd.Timestamp, issue_time: pd.Timestamp, half_window_days: int) -> np.ndarray:
        target, issue_time = pd.Timestamp(target), pd.Timestamp(issue_time)
        issue_day = np.datetime64(issue_time.normalize().date(), "D")
        mask = self.tod == target.hour * 60 + target.minute
        mask &= _circular_doy_distance(self.doy, target.dayofyear) <= half_window_days
        mask &= self.published < issue_day
        mask &= self.times.values < np.datetime64(issue_time)
        return np.flatnonzero(mask)

    def pool_with_widening(self, target, issue_time, n_required: int = config.N_MEMBERS) -> PoolResult:
        hw = config.SEASONAL_HALF_WINDOW_DAYS
        idx = self.pool(target, issue_time, hw)
        base = idx.size
        while idx.size < n_required and hw < config.SEASONAL_HALF_WINDOW_MAX_DAYS:
            hw = min(hw + config.SEASONAL_WIDEN_STEP_DAYS, config.SEASONAL_HALF_WINDOW_MAX_DAYS)
            idx = self.pool(target, issue_time, hw)
        return PoolResult(indices=idx, half_window_days=hw, base_size=base)

    def draw(self, pool: np.ndarray, n_draws: int, n_members: int, rng: np.random.Generator) -> np.ndarray:
        """(n_draws, n_members) indices into the history, each row without replacement."""
        if pool.size < n_members:
            raise ValueError(f"pool of {pool.size} cannot supply {n_members} members")
        return np.stack([rng.choice(pool, size=n_members, replace=False) for _ in range(n_draws)])

    def assert_no_leak(self, drawn: np.ndarray, issue_time: pd.Timestamp) -> None:
        issue_day = np.datetime64(pd.Timestamp(issue_time).normalize().date(), "D")
        if np.any(self.published[drawn] >= issue_day):
            bad = self.times[np.unique(drawn[self.published[drawn] >= issue_day])]
            raise AssertionError(f"template leak at issue {issue_time}: {list(bad[:5])}")


def feasibility(history: TemplateHistory, cases: list[tuple[pd.Timestamp, pd.Timestamp]]) -> pd.DataFrame:
    """Pool size per (issue, valid) target, before any result."""
    rows = []
    for issue, valid in cases:
        r = history.pool_with_widening(valid, issue)
        rows.append(
            {
                "issue_time": issue,
                "valid_time": valid,
                "pool_at_21d": r.base_size,
                "half_window_used": r.half_window_days,
                "pool_used": r.indices.size,
                "feasible": r.feasible,
            }
        )
    return pd.DataFrame(rows)
