"""Elexon Insights ingestion: B1610 metered output, PN and BOALF, BM unit reference.

    python -m gbwind.elexon check                 # one probe per endpoint
    python -m gbwind.elexon units                 # list wind BM units
    python -m gbwind.elexon fetch --start 2020-09-01 --end 2026-09-18

All timestamps leaving this module are tz-naive UTC. Settlement period start
times are computed from settlement date and period in Europe/London, so clock
change days with 46 or 50 periods are handled.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import requests

from gbwind import config
from gbwind.config import DEFAULT_PATHS, ELEXON_API, Paths

# Later settlement runs supersede earlier ones.
RUN_ORDER = {"II": 0, "SF": 1, "R1": 2, "R2": 3, "R3": 4, "RF": 5, "DF": 6}

# Days per request. Halved automatically if the API rejects a range.
CHUNK_DAYS = {"B1610": 7, "PN": 1, "BOALF": 1, "MID": 7}

# Datasets that are system wide rather than per BM unit.
SYSTEM_DATASETS = {"MID"}


class ElexonClient:
    def __init__(self, base: str = ELEXON_API, pause_s: float = 0.1, retries: int = 4):
        self.base = base.rstrip("/")
        self.pause_s = pause_s
        self.retries = retries
        self.session = requests.Session()

    def get(self, path: str, params: list[tuple[str, str]] | dict | None = None):
        url = f"{self.base}/{path.lstrip('/')}"
        for attempt in range(self.retries):
            try:
                r = self.session.get(url, params=params, timeout=60)
            except requests.RequestException:
                if attempt == self.retries - 1:
                    raise
                time.sleep(2**attempt)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2**attempt)
                continue
            r.raise_for_status()
            time.sleep(self.pause_s)
            body = r.json()
            return body["data"] if isinstance(body, dict) and "data" in body else body
        r.raise_for_status()
        return None

    def stream(self, dataset: str, start: pd.Timestamp, end: pd.Timestamp, bmunits: Iterable[str]):
        params = [("from", _iso(start)), ("to", _iso(end))]
        params += [("bmUnit", u) for u in bmunits]
        return self.get(f"datasets/{dataset}/stream", params)

    def bmunits(self) -> list[dict]:
        return self.get("reference/bmunits/all")


def _iso(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%MZ")


def sp_start_utc(settlement_date, settlement_period) -> pd.Series:
    """UTC start of each settlement period.

    Settlement day starts at local midnight in Europe/London and runs in 30 minute
    steps of absolute time, so on clock change days it has 46 or 50 periods.
    """
    dates = pd.to_datetime(pd.Series(settlement_date).astype(str))
    midnight = dates.dt.tz_localize("Europe/London").dt.tz_convert("UTC")
    offset = pd.to_timedelta((pd.Series(settlement_period).astype(int).to_numpy() - 1) * 30, "min")
    return (midnight + offset).dt.tz_localize(None).reset_index(drop=True)


def _utc_naive(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True).dt.tz_localize(None)


def tidy_b1610(rows: list[dict]) -> pd.DataFrame:
    cols = ["bmunit", "sp_start", "mwh", "run"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    out = pd.DataFrame(
        {
            "bmunit": df["bmUnit"].astype(str),
            "sp_start": sp_start_utc(df["settlementDate"], df["settlementPeriod"]),
            "mwh": pd.to_numeric(df["quantity"], errors="coerce"),
            "run": df.get("settlementRunType", pd.Series(["?"] * len(df))).astype(str),
        }
    )
    out["_rank"] = out["run"].map(RUN_ORDER).fillna(-1)
    out = out.sort_values("_rank").drop_duplicates(["bmunit", "sp_start"], keep="last")
    return out.drop(columns="_rank").sort_values(["bmunit", "sp_start"]).reset_index(drop=True)


def tidy_levels(rows: list[dict], dataset: str) -> pd.DataFrame:
    cols = ["bmunit", "time_from", "time_to", "level_from", "level_to"]
    if dataset == "BOALF":
        cols += ["acceptance_number", "so_flag"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    out = pd.DataFrame(
        {
            "bmunit": df["bmUnit"].astype(str),
            "time_from": _utc_naive(df["timeFrom"]),
            "time_to": _utc_naive(df["timeTo"]),
            "level_from": pd.to_numeric(df["levelFrom"], errors="coerce"),
            "level_to": pd.to_numeric(df["levelTo"], errors="coerce"),
        }
    )
    if dataset == "BOALF":
        out["acceptance_number"] = df["acceptanceNumber"]
        out["so_flag"] = df.get("soFlag", False)
        out = out.drop_duplicates(["bmunit", "acceptance_number", "time_from", "time_to"])
    else:
        out = out.drop_duplicates(["bmunit", "time_from", "time_to"], keep="last")
    return out.sort_values(["bmunit", "time_from"]).reset_index(drop=True)


def tidy_mid(rows: list[dict]) -> pd.DataFrame:
    """Market index price per settlement period and provider."""
    cols = ["sp_start", "provider", "price", "volume"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    out = pd.DataFrame(
        {
            "sp_start": sp_start_utc(df["settlementDate"], df["settlementPeriod"]),
            "provider": df["dataProvider"].astype(str),
            "price": pd.to_numeric(df["price"], errors="coerce"),
            "volume": pd.to_numeric(df["volume"], errors="coerce"),
        }
    )
    return out.drop_duplicates(["sp_start", "provider"], keep="last").sort_values("sp_start").reset_index(drop=True)


def fetch_range(
    client: ElexonClient, dataset: str, start: pd.Timestamp, end: pd.Timestamp, bmunits: list[str]
) -> list[dict]:
    """Fetch [start, end) in chunks, halving a chunk the API refuses."""
    rows: list[dict] = []
    step = pd.Timedelta(days=CHUNK_DAYS[dataset])
    t = start
    while t < end:
        t2 = min(t + step, end)
        try:
            got = client.stream(dataset, t, t2, bmunits)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 400 and step > pd.Timedelta(hours=6):
                step = step / 2
                continue
            raise
        rows.extend(got or [])
        t = t2
    return rows


def month_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    return list(pd.date_range(start.to_period("M").to_timestamp(), end, freq="MS"))


def fetch_all(
    bmunits: list[str],
    start: str,
    end: str,
    paths: Paths = DEFAULT_PATHS,
    datasets: tuple[str, ...] = ("B1610", "PN", "BOALF", "MID"),
    refresh_recent_days: int = 40,
    client: ElexonClient | None = None,
) -> None:
    """Download monthly parquet files per dataset. Completed old months are skipped."""
    client = client or ElexonClient()
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    for ds in datasets:
        outdir = paths.raw_elexon / ds
        outdir.mkdir(parents=True, exist_ok=True)
        for m in month_starts(t0, t1):
            m_end = min(m + pd.offsets.MonthBegin(1), t1 + pd.Timedelta(days=1))
            f = outdir / f"{m:%Y-%m}.parquet"
            if f.exists() and m_end < now - pd.Timedelta(days=refresh_recent_days):
                continue
            rows = fetch_range(client, ds, max(m, t0), m_end, [] if ds in SYSTEM_DATASETS else bmunits)
            if ds == "B1610":
                df = tidy_b1610(rows)
            elif ds == "MID":
                df = tidy_mid(rows)
            else:
                df = tidy_levels(rows, ds)
            df.to_parquet(f, index=False)
            print(f"{ds} {m:%Y-%m}: {len(df)} rows", flush=True)


def load(dataset: str, paths: Paths = DEFAULT_PATHS) -> pd.DataFrame:
    files = sorted((paths.raw_elexon / dataset).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {dataset} files under {paths.raw_elexon}; run gbwind.elexon fetch")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def wind_units(client: ElexonClient | None = None) -> pd.DataFrame:
    client = client or ElexonClient()
    df = pd.DataFrame(client.bmunits())
    df = df[df["fuelType"] == "WIND"].copy()
    df["generationCapacity"] = pd.to_numeric(df["generationCapacity"], errors="coerce")
    cols = ["elexonBmUnit", "bmUnitName", "leadPartyName", "bmUnitType", "generationCapacity"]
    return df[cols].drop_duplicates("elexonBmUnit").sort_values("elexonBmUnit").reset_index(drop=True)


def save_bmunit_reference(paths: Paths = DEFAULT_PATHS, client: ElexonClient | None = None) -> Path:
    client = client or ElexonClient()
    paths.raw_bmunits.parent.mkdir(parents=True, exist_ok=True)
    paths.raw_bmunits.write_text(json.dumps(client.bmunits()))
    return paths.raw_bmunits


def check(client: ElexonClient | None = None) -> bool:
    """One probe per endpoint, plus a consistency check on settlement period timing."""
    client = client or ElexonClient()
    unit = "T_WHILW-1"
    day = pd.Timestamp("2024-07-10")
    ok = True
    probes = {
        "B1610": lambda: client.stream("B1610", day, day + pd.Timedelta(hours=2), [unit]),
        "PN": lambda: client.stream("PN", day, day + pd.Timedelta(hours=2), [unit]),
        "BOALF": lambda: client.stream("BOALF", day, day + pd.Timedelta(hours=2), []),
        "reference/bmunits": client.bmunits,
    }
    for name, probe in probes.items():
        try:
            rows = probe()
            print(f"ok   {name:<18} {len(rows)} rows  fields: {sorted(rows[0])[:8] if rows else []}")
        except Exception as e:
            ok = False
            print(f"FAIL {name:<18} {e}")
    try:
        rows = client.stream("B1610", day, day + pd.Timedelta(hours=2), [unit])
        df = pd.DataFrame(rows)
        ours = sp_start_utc(df["settlementDate"], df["settlementPeriod"]) + pd.Timedelta(minutes=30)
        theirs = pd.to_datetime(df["halfHourEndTime"])
        offset = (theirs - ours).dt.total_seconds().div(3600).unique()
        print(f"info B1610 halfHourEndTime minus computed UTC end, hours: {offset.tolist()} "
              "(1.0 in summer means Elexon reports local time)")
    except Exception as e:
        print(f"warn timing check skipped: {e}")
    return ok


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m gbwind.elexon")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    sub.add_parser("units")
    f = sub.add_parser("fetch")
    f.add_argument("--start", default=config.HISTORY_START)
    f.add_argument("--end", default=pd.Timestamp.now().strftime("%Y-%m-%d"))
    f.add_argument("--datasets", nargs="+", default=["B1610", "PN", "BOALF", "MID"])
    a = p.parse_args(argv)

    if a.cmd == "check":
        return 0 if check() else 1
    if a.cmd == "units":
        df = wind_units()
        print(df.to_string(index=False))
        print(f"\n{len(df)} wind BM units")
        return 0
    if a.cmd == "fetch":
        from gbwind.portfolio import load_yaml

        units = [u for farm in load_yaml(DEFAULT_PATHS.portfolio_yaml)["farms"] for u in farm["bmunits"]]
        save_bmunit_reference()
        fetch_all(units, a.start, a.end, datasets=tuple(a.datasets))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
