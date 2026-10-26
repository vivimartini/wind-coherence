"""The sealed test period is opened once.

Opening writes a lock recording when, at which commit, and with which end date.
Any later attempt to evaluate a sealed period refuses, so a second look at the
test data has to be a visible, deliberate act (deleting the lock) rather than an
accident.
"""

from __future__ import annotations

import json
import subprocess

import pandas as pd

from gbwind.config import ROOT, SEALED_PERIODS, Paths


class SealError(RuntimeError):
    pass


def lock_path(paths: Paths, period: str):
    return paths.seal_dir / f"{period}.opened.json"


def is_opened(paths: Paths, period: str) -> bool:
    return lock_path(paths, period).exists()


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def open_period(paths: Paths, period: str, end_date: str, confirm: bool) -> dict:
    """Return the lock record, creating it on first open. Refuses a second open."""
    if period not in SEALED_PERIODS:
        return {"period": period, "sealed": False}
    if not confirm:
        raise SealError(f"{period} is sealed. Pass --open-sealed to open it, once.")
    lock = lock_path(paths, period)
    if lock.exists():
        raise SealError(f"{period} was already opened: {lock.read_text()}")
    record = {
        "period": period,
        "opened_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "commit": _commit(),
        "end_date": end_date,
    }
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps(record, indent=2))
    return record
