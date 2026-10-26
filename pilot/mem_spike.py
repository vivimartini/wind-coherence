"""Peak memory of one training-shaped step, on one GPU.

Two members and a 42 h rollout is the step that already fits. Four members is
the next width. The peak on this one device is what decides whether that width
needs member sharding.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m1_suite

ROOT = m1_suite.ROOT
OUT = ROOT / "results" / "spike"
LOG = ROOT / "logs"


def main() -> int:
    m1_suite.OUT = OUT
    m1_suite.LOG = LOG
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    journal = OUT / "spike.jsonl"
    rows = []
    # 7 steps is 42 h at a 6 h step. Rematerialisation is on, which is the only
    # way the 42 h step has fitted so far.
    configs = (
        ("spike_m2_s7", ["--mode", "grad", "--members", "2", "--steps", "7", "--checkpointing", "1"]),
        ("spike_m4_s7", ["--mode", "grad", "--members", "4", "--steps", "7", "--checkpointing", "1"]),
    )
    for name, args in configs:
        record = m1_suite.run_probe(name, args, timeout_s=2400)
        rows.append(record)
        with journal.open("a") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
        if m1_suite.oom(record) and name == "spike_m2_s7":
            rows.append({"name": "spike_m4_s7", "status": "skipped", "reason": "two members already exhausted one GPU"})
            break
    (OUT / "summary.json").write_text(json.dumps(rows, indent=2, default=str))
    print(json.dumps(rows, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
