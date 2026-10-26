"""One cycle in, one object out.

The scheduled task and a backfill call this same function. The object key is
the cycle, so running it twice does not create a second copy.

Which cycles to ingest comes from the source archive, not the clock: the
archive publishes each cycle hours after its nominal time, and a run that
fires early or is missed must not lose a cycle. Each run therefore archives
every cycle the source holds from the lookback window that is not already in
the store, oldest first, and "no new cycle" is a normal outcome.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

from gbwind.object_store import ObjectStore, open_store

SOURCES = ("aifs-ens", "ifs-ens")
LOOKBACK = pd.Timedelta(days=2)


def _naive_utc(stamp) -> pd.Timestamp:
    stamp = pd.Timestamp(stamp)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp


def pending_cycles(source: str, available: Iterable, store: ObjectStore, now, lookback: pd.Timedelta = LOOKBACK):
    """Cycles the source holds within the lookback window and the store lacks, oldest first."""
    now = _naive_utc(now)
    stamps = sorted({_naive_utc(t) for t in pd.DatetimeIndex(list(available))})
    recent = [t for t in stamps if now - lookback <= t <= now]
    return [t for t in recent if not store.exists(cycle_key(source, t))]


def ingest_pending(
    store: ObjectStore,
    source: str,
    available: Iterable,
    extract: Callable[[pd.Timestamp], pd.DataFrame | None],
    now,
    lookback: pd.Timedelta = LOOKBACK,
) -> list[dict]:
    """Archive each pending cycle. A cycle that is unreadable or not fully written yet is left for the next run."""
    todo = pending_cycles(source, available, store, now, lookback)
    if not todo:
        return [{"event": "ingest", "source": source, "status": "no new cycle"}]
    events = []
    for cycle in todo:
        frame = extract(cycle)
        key = cycle_key(source, cycle)
        if frame is None:
            events.append({"event": "ingest", "key": key, "status": "unreadable, retried next run"})
            continue
        values = frame.select_dtypes("floating").to_numpy()
        if not np.isfinite(values).all():
            events.append({"event": "ingest", "key": key, "status": "incomplete, retried next run"})
            continue
        events.append({"event": "ingest", "key": key, "status": publish_frame(store, source, cycle, frame)})
    return events


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
