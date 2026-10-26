"""Farm level settlement period table with metering, curtailment and price flags.

A farm's settlement period is
  metered         if every one of its BM units has a B1610 value,
  curtailed       if any of its BM units was instructed below (1 - 15%) of its
                  physical notification by an accepted bid-offer acceptance,
  negative_price  if the market index price was negative (system wide), since
                  farms may self-curtail then without any accepted instruction.
any_boa records whether any acceptance touched the period at all, for sensitivity.

The centred hourly table is what both evaluation and the template use: the
value at time t is the mean of the settlement periods starting at t - 30 min and
at t, admissible only if both are.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gbwind import config, elexon
from gbwind.config import DEFAULT_PATHS, Paths

SP = pd.Timedelta(minutes=30)


def _overlap_slices(grid: np.ndarray, t0: np.ndarray, t1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index range of periods [s, s + 30 min) overlapping each interval [t0, t1]."""
    lo = np.searchsorted(grid, t0 - SP.to_timedelta64(), side="right")
    hi = np.searchsorted(grid, t1, side="left")
    return lo, hi


def boa_min_level(boalf: pd.DataFrame, sp_index: pd.DatetimeIndex, units: list[str]) -> pd.DataFrame:
    """Lowest accepted level per (period, unit); NaN where no acceptance touches the period."""
    out = pd.DataFrame(np.nan, index=sp_index, columns=units)
    if not len(boalf):
        return out
    grid = sp_index.values
    for unit, g in boalf[boalf["bmunit"].isin(units)].groupby("bmunit"):
        col = np.full(len(grid), np.nan)
        lo, hi = _overlap_slices(grid, g["time_from"].values, g["time_to"].values)
        level = np.fmin(g["level_from"].to_numpy(float), g["level_to"].to_numpy(float))
        for a, b, v in zip(lo, hi, level, strict=True):
            col[a:b] = np.fmin(col[a:b], v)
        out[unit] = col
    return out


def pn_level(pn: pd.DataFrame, sp_index: pd.DatetimeIndex, units: list[str]) -> pd.DataFrame:
    """Duration weighted mean physical notification per (period, unit)."""
    if not len(pn):
        return pd.DataFrame(np.nan, index=sp_index, columns=units)
    p = pn[pn["bmunit"].isin(units)].copy()
    p["sp_start"] = p["time_from"].dt.floor("30min")
    p["w"] = (p["time_to"] - p["time_from"]).dt.total_seconds().clip(lower=1.0)
    p["wl"] = p["w"] * (p["level_from"] + p["level_to"]) / 2.0
    g = p.groupby(["sp_start", "bmunit"])[["wl", "w"]].sum()
    lvl = (g["wl"] / g["w"]).unstack("bmunit")
    return lvl.reindex(index=sp_index, columns=units)


def curtailed_units(boa_min: pd.DataFrame, pn: pd.DataFrame, threshold: float = config.CURTAILMENT_THRESHOLD) -> pd.DataFrame:
    """True where an acceptance takes a unit below (1 - threshold) of its notification.

    An acceptance in a period with no notification is flagged, conservatively.
    """
    has_boa = boa_min.notna()
    below = boa_min < (1.0 - threshold) * pn.clip(lower=0.0)
    reduces = pn > 0
    return has_boa & ((below & reduces) | pn.isna())


def negative_price_flags(mid: pd.DataFrame, sp_index: pd.DatetimeIndex) -> pd.Series:
    if mid is None or not len(mid):
        return pd.Series(False, index=sp_index)
    traded = mid[mid["volume"] > 0]
    neg = traded.groupby("sp_start")["price"].min() < config.NEGATIVE_PRICE_THRESHOLD
    return neg.reindex(sp_index, fill_value=False).astype(bool)


def build_farm_sp(
    b1610: pd.DataFrame,
    boalf: pd.DataFrame,
    farms: dict[str, list[str]],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    pn: pd.DataFrame | None = None,
    mid: pd.DataFrame | None = None,
    threshold: float = config.CURTAILMENT_THRESHOLD,
) -> pd.DataFrame:
    """Long table: farm, sp_start, mw, metered, curtailed, any_boa, negative_price."""
    sp_index = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="30min", inclusive="left")
    b = b1610.drop_duplicates(["bmunit", "sp_start"], keep="last")
    mwh = b.pivot(index="sp_start", columns="bmunit", values="mwh").reindex(sp_index)
    all_units = sorted({u for us in farms.values() for u in us})
    boa = boa_min_level(boalf, sp_index, all_units)
    pn_l = pn_level(pn if pn is not None else pd.DataFrame(), sp_index, all_units)
    cur = curtailed_units(boa, pn_l, threshold)
    neg = negative_price_flags(mid, sp_index).to_numpy()

    out = []
    for farm, units in farms.items():
        m = mwh.reindex(columns=units)
        out.append(
            pd.DataFrame(
                {
                    "farm": farm,
                    "sp_start": sp_index,
                    "mw": (m.sum(axis=1, min_count=len(units)) * 2.0).to_numpy(),
                    "metered": m.notna().all(axis=1).to_numpy(),
                    "curtailed": cur[units].any(axis=1).to_numpy(),
                    "any_boa": boa[units].notna().any(axis=1).to_numpy(),
                    "negative_price": neg,
                }
            )
        )
    return pd.concat(out, ignore_index=True)


def admissible_mask(farm_sp: pd.DataFrame, include_curtailed: bool = False) -> pd.Series:
    ok = farm_sp["metered"].astype(bool)
    if not include_curtailed:
        ok &= ~farm_sp["curtailed"].astype(bool)
        if "negative_price" in farm_sp:
            ok &= ~farm_sp["negative_price"].astype(bool)
    return ok


def centred_hourly(farm_sp: pd.DataFrame, include_curtailed: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide tables indexed by instant t: mean MW over [t - 30 min, t + 30 min), and admissibility.

    Only instants on the hour are kept, since forecasts are valid on the hour.
    """
    mw = farm_sp.pivot(index="sp_start", columns="farm", values="mw").sort_index()
    adm = (
        farm_sp.assign(ok=admissible_mask(farm_sp, include_curtailed).to_numpy())
        .pivot(index="sp_start", columns="farm", values="ok")
        .sort_index()
        .reindex(mw.index)
        .fillna(False)
        .astype(bool)
    )
    prev_mw, prev_adm = mw.shift(1, freq="30min"), adm.shift(1, freq="30min")
    idx = mw.index[mw.index.minute == 0]
    idx = idx[idx.isin(prev_mw.index)]
    val = (mw.loc[idx] + prev_mw.reindex(idx)) / 2.0
    both = adm.loc[idx] & prev_adm.reindex(idx).fillna(False).astype(bool)
    both &= val.notna()
    return val, both


def report(farm_sp: pd.DataFrame) -> pd.DataFrame:
    g = farm_sp.assign(admissible=admissible_mask(farm_sp).to_numpy()).groupby("farm")
    cols = {"metered_fraction": g["metered"].mean(), "curtailed_fraction": g["curtailed"].mean()}
    if "any_boa" in farm_sp:
        cols["any_boa_fraction"] = g["any_boa"].mean()
    cols["admissible_fraction"] = g["admissible"].mean()
    return pd.DataFrame(cols)


def run(paths: Paths = DEFAULT_PATHS, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    from gbwind.portfolio import load_yaml

    farms = {f["id"]: f["bmunits"] for f in load_yaml(paths.portfolio_yaml)["farms"]}
    b1610 = elexon.load("B1610", paths)
    optional = {}
    for ds in ("BOALF", "PN", "MID"):
        try:
            optional[ds] = elexon.load(ds, paths)
        except FileNotFoundError:
            print(f"warning: no {ds} data, its flags will be empty")
            optional[ds] = pd.DataFrame()
    start = start or config.HISTORY_START
    end = end or (b1610["sp_start"].max() + SP)
    table = build_farm_sp(b1610, optional["BOALF"], farms, start, end, pn=optional["PN"], mid=optional["MID"])
    paths.interim.mkdir(parents=True, exist_ok=True)
    table.to_parquet(paths.farm_sp, index=False)
    print(report(table).round(4).to_string())
    return table
