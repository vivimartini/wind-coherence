"""End to end orchestration of one period."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from gbwind import archive, config, evaluate, seal
from gbwind.config import Paths
from gbwind.curtailment import centred_hourly
from gbwind.portfolio import load_selected
from gbwind.template import TemplateHistory, _bank_holidays


def load_inputs(paths: Paths, include_curtailed: bool = False) -> evaluate.Inputs:
    portfolio = load_selected(paths)
    farms = portfolio["farm"].tolist()
    farm_sp = pd.read_parquet(paths.farm_sp)
    farm_sp = farm_sp[farm_sp["farm"].isin(farms)]
    values, admissible = centred_hourly(farm_sp, include_curtailed=include_curtailed)
    history = TemplateHistory(values, admissible, farms)
    forecasts = {
        s: archive.load_forecasts(s, paths) for s in config.SYSTEMS if paths.forecast_store(s).exists()
    }
    if not forecasts:
        raise FileNotFoundError(f"no forecast stores under {paths.forecasts}; run gbwind archive first")
    return evaluate.Inputs(portfolio, forecasts, values, admissible, history)


def default_test_end(today: pd.Timestamp | None = None) -> str:
    """Last issue date whose furthest valid time has published metering."""
    today = pd.Timestamp(today or pd.Timestamp.now()).normalize()
    last_published = np.busday_offset(
        np.datetime64(today.date(), "D"), -config.PUBLICATION_LAG_WORKING_DAYS - 1,
        roll="backward", holidays=_bank_holidays(),
    )
    return str(pd.Timestamp(last_published) - pd.Timedelta(days=2))[:10]


def run_period(
    period: str,
    paths: Paths,
    open_sealed: bool = False,
    test_end: str | None = None,
    sensitivities: bool = True,
) -> dict:
    if period in config.SEALED_PERIODS:
        test_end = test_end or default_test_end()
        seal.open_period(paths, period, test_end, confirm=open_sealed)
    start, end = config.period_bounds(period, test_end)
    outroot = paths.period_results(period)
    report: dict = {"period": period, "issue_start": str(start.date()), "issue_end": str(end.date()), "runs": {}}

    inp = load_inputs(paths)
    runs = [(s, s, config.LEAD_HOURS_PRIMARY, inp) for s in inp.forecasts]
    if sensitivities:
        if "ifs-ens" in inp.forecasts:
            runs.append(("ifs-ens_3h", "ifs-ens", config.LEAD_HOURS_IFS_SENSITIVITY, inp))
        inp_c = load_inputs(paths, include_curtailed=True)
        runs += [(f"{s}_with_curtailed", s, config.LEAD_HOURS_PRIMARY, inp_c) for s in inp_c.forecasts]

    for label, system, leads, data in runs:
        print(f"[{period}] {label}: evaluating", flush=True)
        result = evaluate.evaluate_system(system, data, start, end, leads)
        summary = evaluate.summarise(result)
        summary.update({"system": system, "label": label, "leads_h": list(leads)})
        if system == "aifs-ens":
            summary["aifs_versions"] = sorted({config.aifs_version(t) for t in result["scores"].get("init_time", [])})
        evaluate.save(result, summary, outroot / label)
        if label in inp.forecasts and sensitivities:
            for bl in config.BLOCK_LENGTH_SENSITIVITY_DAYS:
                alt = evaluate.summarise(result, block_length=bl)
                (outroot / label / f"summary_block{bl}.json").write_text(json.dumps(alt, indent=2, default=str))
        report["runs"][label] = _headline(summary)

    outroot.mkdir(parents=True, exist_ok=True)
    (outroot / "report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def _headline(summary: dict) -> dict:
    if not summary.get("n_cases"):
        return summary
    d = summary["differences_vs_native"]["crps"]
    return {
        "n_days": summary["n_days"],
        "n_cases": summary["n_cases"],
        "native_crps": summary["metrics"]["crps"]["native"]["estimate"],
        "crps_minus_native": {c: {k: round(v, 6) if isinstance(v, float) else v for k, v in r.items()} for c, r in d.items()},
    }
