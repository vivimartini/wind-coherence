"""One cycle in, one object out.

The scheduled task and a backfill call this same function. The object key is
the cycle, so running it twice does not create a second copy.
"""

from __future__ import annotations

import io

import pandas as pd

from gbwind.object_store import ObjectStore, open_store

SOURCES = ("aifs-ens", "ifs-ens")
# Open-data archives lag the nominal issue time. Eight hours is long enough
# for a 6-hourly cycle to be present, and short enough that the task still
# archives that cycle on the same day.
PUBLISH_LAG = pd.Timedelta(hours=8)


def due_cycle(now, lag: pd.Timedelta = PUBLISH_LAG) -> pd.Timestamp:
    """Latest 00/06/12/18 UTC cycle old enough to be in the open archive."""
    stamp = pd.Timestamp(now)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    ready = (stamp - lag).floor("h")
    hour = (int(ready.hour) // 6) * 6
    return ready.replace(hour=hour, minute=0, second=0, microsecond=0, nanosecond=0)


def cycle_key(source: str, cycle: pd.Timestamp) -> str:
    if source not in SOURCES:
        raise ValueError(f"unknown source {source}")
    stamp = pd.Timestamp(cycle).strftime("%Y-%m-%dT%H")
    return f"archive/source={source}/init={stamp}.parquet"


def frame_bytes(frame: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    frame.to_parquet(buf, index=False)
    return buf.getvalue()


def publish_frame(store: ObjectStore, source: str, cycle: pd.Timestamp, frame: pd.DataFrame) -> str:
    return store.put_bytes(cycle_key(source, cycle), frame_bytes(frame))


def resolve_store(url: str | None, default_root) -> ObjectStore:
    return open_store(url or str(default_root))
