"""Milestone 1 suite. Each configuration is a subprocess so an OOM is a data point.

Writes one JSON object per line as soon as that configuration finishes, so a
later timeout still leaves the earlier measurements.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/rds/user/vm481/hpc-work/wind-coherence")
REPO = Path("/home/vm481/hpc-work/wind-coherence")
OUT = ROOT / "results" / "m1"
LOG = ROOT / "logs"
PYTHON = ROOT / "venv-wn" / "bin" / "python"
PROBE = REPO / "pilot" / "m1_probe.py"

# Inherit the driver's CUDA libraries. Do not preallocate the whole GPU, or the
# memory curve is just the preallocation.
ENV = os.environ.copy()
ENV["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
ENV["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
ENV["JAX_COMPILATION_CACHE_DIR"] = str(ROOT / "jax-cache")
ENV["WN_ROOT"] = str(ROOT)


def gpu_used() -> tuple[float | None, float | None, str | None]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,name", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None, None, None
    used, total, name = [p.strip() for p in out.split(",")]
    return float(used), float(total), name


def run_probe(name: str, args: list[str], timeout_s: int) -> dict:
    out_path = OUT / f"{name}.json"
    cmd = [str(PYTHON), str(PROBE), "--out", str(out_path), *args]
    log_path = LOG / f"{name}.log"
    print(f"\n=== {name} ===", flush=True)
    print(" ".join(cmd), flush=True)
    t0 = time.perf_counter()
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=ENV)
        peak = 0.0
        total = None
        gpu_name = None
        while proc.poll() is None:
            used, tot, nm = gpu_used()
            if used is not None:
                peak = max(peak, used)
                total = tot
                gpu_name = nm
            if time.perf_counter() - t0 > timeout_s:
                proc.kill()
                proc.wait()
                record = {
                    "name": name,
                    "status": "timeout",
                    "args": args,
                    "wall_seconds": round(time.perf_counter() - t0, 3),
                    "peak_memory_mib": peak,
                    "gpu_memory_total_mib": total,
                    "gpu_name": gpu_name,
                    "log": str(log_path),
                }
                print(json.dumps(record), flush=True)
                return record
            time.sleep(2)
        used, tot, nm = gpu_used()
        if used is not None:
            peak = max(peak, used)
            total = tot
            gpu_name = nm
    wall = time.perf_counter() - t0
    payload = {}
    if out_path.is_file():
        try:
            payload = json.loads(out_path.read_text())
        except json.JSONDecodeError:
            payload = {"status": "bad_json"}
    status = payload.get("status", "crashed")
    if proc.returncode != 0 and status == "ok":
        status = "crashed"
    if proc.returncode < 0 and status != "ok":
        status = "killed"
    record = {
        "name": name,
        "status": status if proc.returncode == 0 else status,
        "returncode": proc.returncode,
        "args": args,
        "wall_seconds": round(wall, 3),
        "peak_memory_mib": peak,
        "gpu_memory_total_mib": total,
        "gpu_name": gpu_name,
        "loss_finite": payload.get("loss_finite"),
        "loss": payload.get("loss"),
        "first_call_seconds": payload.get("first_call_seconds"),
        "steady_call_seconds": payload.get("steady_call_seconds"),
        "compile_overhead_seconds": payload.get("compile_overhead_seconds"),
        "update": (payload.get("gradients") or {}).get("update"),
        "hundred_metre_wind_is_a_model_target": (payload.get("wind") or {}).get("hundred_metre_wind_is_a_model_target"),
        "error_type": payload.get("error_type"),
        "error": payload.get("error"),
        "log": str(log_path),
        "detail": str(out_path) if out_path.is_file() else None,
    }
    print(json.dumps({k: record[k] for k in record if k != "update"}, default=str), flush=True)
    if record.get("update"):
        print("update " + json.dumps(record["update"], default=str), flush=True)
    return record


def oom(record: dict) -> bool:
    if record["status"] in {"killed", "timeout"}:
        return True
    text = f"{record.get('error_type', '')} {record.get('error', '')}".lower()
    return any(token in text for token in ("out of memory", "resource_exhausted", "oom"))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    (ROOT / "jax-cache").mkdir(parents=True, exist_ok=True)
    freeze = subprocess.check_output([str(PYTHON), "-m", "pip", "freeze"], text=True)
    (OUT / "pip-freeze.txt").write_text(freeze)
    journal = OUT / "suite.jsonl"
    rows = []

    def keep(record: dict) -> None:
        rows.append(record)
        with journal.open("a") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    keep(run_probe("inventory", ["--mode", "inventory"], timeout_s=1800))
    if rows[-1]["status"] != "ok":
        print("inventory failed; not starting gradient profiles", flush=True)
        return 1

    # Single-step gradient. Unbiased CRPS is undefined at 1 member; 2 is the
    # released default. The update check runs on that default.
    member_ok = True
    for members in (1, 2, 4, 8):
        if not member_ok:
            keep({"name": f"grad_m{members}_s1", "status": "skipped", "reason": "smaller ensemble exhausted memory"})
            continue
        update = ["--update"] if members == 2 else []
        record = run_probe(
            f"grad_m{members}_s1",
            ["--mode", "grad", "--members", str(members), "--steps", "1", *update],
            timeout_s=3600,
        )
        keep(record)
        if oom(record):
            member_ok = False

    # 4 steps = 24 h, 7 steps = 42 h, at the model's 6 h step. Try rematerialisation
    # even when the uncheckpointed rollout dies: that is the comparison.
    for steps in (4, 7):
        step_dead = False
        for members in (1, 2, 4):
            if step_dead:
                keep({"name": f"roll_m{members}_s{steps}", "status": "skipped", "reason": "smaller rollout exhausted memory with and without rematerialisation"})
                continue
            both_oom = True
            for flag, label in ((0, "off"), (1, "on")):
                record = run_probe(
                    f"roll_m{members}_s{steps}_ckpt{label}",
                    [
                        "--mode", "grad",
                        "--members", str(members),
                        "--steps", str(steps),
                        "--checkpointing", str(flag),
                    ],
                    timeout_s=3600,
                )
                keep(record)
                if not oom(record):
                    both_oom = False
            if both_oom:
                step_dead = True

    summary = {
        "n": len(rows),
        "ok": sum(r.get("status") == "ok" for r in rows),
        "oom_or_killed": sum(oom(r) for r in rows if r.get("status") != "skipped"),
        "journal": str(journal),
    }
    (OUT / "summary.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, default=str))
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
