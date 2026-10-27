"""One AIFS ENS v2 member, one 6 h step, from ECMWF open data. Records peak memory."""

from __future__ import annotations

import datetime
import json
import os
import traceback
from collections import defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("WN_ROOT", "/rds/user/vm481/hpc-work/wind-coherence"))
OUT = ROOT / "results" / "aifs-spike"
LSM = ROOT / "data" / "aifs" / "lsm.grib"
LEAD_HOURS = 6
# AIFS_DATE, for example 2026-05-20T00, picks an archived cycle; AIFS_SOURCE is
# ecmwf (recent runs only), aws (archive from 18 Jan 2023) or azure.
DATE = os.environ.get("AIFS_DATE")
SOURCE = os.environ.get("AIFS_SOURCE", "ecmwf")
WIND_AND_CORE = ["100u", "100v", "10u", "10v", "2t", "msl", "z_500", "t_850"]

PARAM_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "sd"]
PARAM_SFC_FC = ["lsm", "z", "slor", "sdor"]
PARAM_SOIL = ["vsw", "sot"]
PARAM_WAVE = ["wmb", "h1012", "h1214", "h1417", "h1721", "h2125", "h2530", "mwd", "cdww", "mwp", "swh"]
PARAM_PL = ["gh", "t", "u", "v", "w", "q"]
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50, 10]
SOIL_LEVELS = [1, 2]
SOIL_NAMES = {"sot_1": "stl1", "sot_2": "stl2", "vsw_1": "swvl1", "vsw_2": "swvl2"}


def write(record: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "step.json").write_text(json.dumps(record, indent=2, default=str))
    print(json.dumps(record, default=str), flush=True)


def main() -> int:
    record: dict = {
        "model": "ecmwf/aifs-ens-2.0",
        "lead_hours": LEAD_HOURS,
        "members": 1,
        "status": "started",
    }
    write(record)
    try:
        import numpy as np
        import torch

        record["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        if not torch.cuda.is_available():
            record["status"] = "no_cuda"
            write(record)
            return 1
        torch.cuda.reset_peak_memory_stats()
        fields, date = initial_state()
        record["init_source"] = SOURCE
        record["input_fields"] = sorted(fields)
        from aifs_guards import check_static_fields, install_nan_guard
        from anemoi.inference.runners.simple import SimpleRunner

        runner = SimpleRunner(str(ROOT / "data" / "aifs" / "aifs-ens-crps-2.0.ckpt"))
        static = check_static_fields(fields, runner.model)
        record["static_fields"] = static
        if not all(entry["ok"] for entry in static.values()):
            record["status"] = "static_field_check_failed"
            write(record)
            return 1
        guard_log: list = []
        install_nan_guard(runner.model.model, guard_log)
        last = None
        step_checks = []
        for state in runner.run(input_state={"date": date, "fields": fields}, lead_time=LEAD_HOURS):
            finite = {
                name: bool(np.isfinite(state["fields"][name]).all()) for name in WIND_AND_CORE if name in state["fields"]
            }
            step_checks.append({"valid": str(state.get("date")), **finite})
            if not all(finite.values()):
                record.update({"status": "nan_in_output", "step_checks": step_checks})
                write(record)
                return 1
            last = state
        names = sorted(last["fields"])
        record.update(
            {
                "status": "ok",
                "nan_guard": {"model_calls": len(guard_log), "all_finite": True},
                "step_checks": step_checks,
                "init": str(date),
                "valid": str(last.get("date")),
                "n_fields": len(names),
                "has_100u": "100u" in names,
                "has_100v": "100v" in names,
                "has_10u": "10u" in names,
                "has_10v": "10v" in names,
                "field_shape": list(np.shape(last["fields"]["100u"])) if "100u" in names else None,
                "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / (1024**2), 1),
                "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / (1024**2), 1),
            }
        )
        write(record)
        return 0
    except Exception as exc:
        record["status"] = "error"
        record["error_type"] = type(exc).__name__
        record["error"] = f"{exc}\n{traceback.format_exc()}"[-4000:]
        try:
            import torch

            if torch.cuda.is_available():
                record["peak_allocated_mib"] = round(torch.cuda.max_memory_allocated() / (1024**2), 1)
        except Exception:
            pass
        write(record)
        return 1


def initial_state():
    import earthkit.data as ekd
    import earthkit.regrid as ekr
    import numpy as np
    from ecmwf.opendata import Client as OpendataClient

    ekd.config.set("user-cache-directory", str(ROOT / "cache" / "earthkit"))
    date = datetime.datetime.fromisoformat(DATE) if DATE else OpendataClient(SOURCE).latest()
    number = 1

    def fetch(param, levelist=None, number=None, **kwargs):
        fields = defaultdict(list)
        for when in (date - datetime.timedelta(hours=6), date):
            kwargs_now = dict(kwargs)
            if number is None:
                data = ekd.from_source(
                    "ecmwf-open-data", date=when, param=param, levelist=levelist or [], source=SOURCE, **kwargs_now
                )
            else:
                kwargs_now.setdefault("stream", "enfo")
                data = ekd.from_source(
                    "ecmwf-open-data",
                    date=when,
                    param=param,
                    levelist=levelist or [],
                    number=[number],
                    source=SOURCE,
                    **kwargs_now,
                )
            for field in data:
                assert field.to_numpy().shape == (721, 1440)
                values = np.roll(field.to_numpy(), -field.shape[1] // 2, axis=1)
                values = ekr.interpolate(values, {"grid": (0.25, 0.25)}, {"grid": "N320"})
                name = f"{field.metadata('param')}_{field.metadata('levelist')}" if levelist else field.metadata("param")
                fields[name].append(values)
        return {name: np.stack(values) for name, values in fields.items()}

    fields = {}
    fields.update(fetch(PARAM_SFC, number=number, levtype="sfc"))
    fields.update(fetch(PARAM_SFC_FC, levtype="sfc"))
    fields.update(fetch(PARAM_WAVE, stream="waef", number=number))
    soil = fetch(PARAM_SOIL, levelist=SOIL_LEVELS, number=number)
    for key, name in SOIL_NAMES.items():
        fields[name] = soil[key]
    fields.update(fetch(PARAM_PL, levelist=LEVELS, number=number))
    fields.pop("q_10", None)
    mwd = np.deg2rad(fields.pop("mwd"))
    fields["cos_mwd"] = np.cos(mwd)
    fields["sin_mwd"] = np.sin(mwd)
    for level in LEVELS:
        fields[f"z_{level}"] = fields.pop(f"gh_{level}") * 9.80665
    mask = np.equal(ekd.from_source("file", str(LSM))[0].to_numpy(flatten=True), 0)
    for name in ("sd", "swvl1", "swvl2"):
        fields[name][:, mask] = np.nan
    return fields, date


if __name__ == "__main__":
    raise SystemExit(main())
