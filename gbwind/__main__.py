"""Command line entry point.

    python -m gbwind portfolio build          # REPD join, BM unit validation
    python -m gbwind curtailment              # farm level metering and curtailment flags
    python -m gbwind portfolio select         # metering rule, freezes the portfolio
    python -m gbwind archive --system aifs-ens --start 2025-07-02 --end 2026-09-20
    python -m gbwind verify --system aifs-ens # members present at every farm
    python -m gbwind feasibility --period dev # template pool sizes, before any result
    python -m gbwind evaluate --period dev
    python -m gbwind figures --period dev
    python -m gbwind page --period dev
    python -m gbwind run --period dev         # everything above, one command
    python -m gbwind demo                     # synthetic world, end to end, offline
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from gbwind import config
from gbwind.config import DEFAULT_PATHS, SYNTHETIC_PATHS, Paths


def _paths(a) -> Paths:
    return SYNTHETIC_PATHS if getattr(a, "synthetic", False) else DEFAULT_PATHS


def cmd_feasibility(paths: Paths, period: str, test_end: str | None) -> str:
    """Feasibility report. Uses metering and flags only, never forecasts or scores."""
    from gbwind.feasibility import build_report

    periods = tuple(dict.fromkeys(["dev", "v1", period]))
    text = build_report(paths, periods)
    print(text)
    print(f"written to {paths.results / 'feasibility' / 'REPORT.md'}")
    return text


def _cmd_live(a) -> int:
    """Ingest writes one object per cycle. Score is the next piece and does not invent numbers."""
    import os

    from gbwind import archive
    from gbwind.live import cycle_key, due_cycle, publish_frame, resolve_store

    url = a.archive or os.environ.get("GBWIND_ARCHIVE")
    store = resolve_store(url, DEFAULT_PATHS.data / "live")
    if a.live_cmd == "score":
        print('{"event": "score", "status": "waiting_on_settled_metering"}')
        return 0
    if not DEFAULT_PATHS.portfolio_csv.exists():
        print('{"event": "ingest", "status": "missing_portfolio"}')
        return 1
    farms = pd.read_csv(DEFAULT_PATHS.portfolio_csv)
    cycle = pd.Timestamp(a.cycle) if a.cycle else due_cycle(pd.Timestamp.now("UTC"))
    day = cycle.strftime("%Y-%m-%d")
    archive.extract(a.source, day, day, farms, DEFAULT_PATHS)
    ds = archive.load_forecasts(a.source, DEFAULT_PATHS).sel(init_time=cycle.to_datetime64())
    frame = ds.to_dataframe().reset_index()
    status = publish_frame(store, a.source, cycle, frame)
    print(f'{{"event": "ingest", "status": "{status}", "key": "{cycle_key(a.source, cycle)}"}}')
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m gbwind")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("portfolio")
    pp.add_argument("action", choices=["build", "select"])

    sub.add_parser("curtailment")

    ar = sub.add_parser("archive")
    ar.add_argument("--system", choices=list(config.SYSTEMS), required=True)
    ar.add_argument("--start", default=config.PERIODS["v1"][0])
    ar.add_argument("--end", default=pd.Timestamp.now().strftime("%Y-%m-%d"))
    ar.add_argument("--workers", type=int, default=4)

    ve = sub.add_parser("verify")
    ve.add_argument("--system", choices=list(config.SYSTEMS), required=True)

    live = sub.add_parser("live", help="operational ingest and scoring")
    live_sub = live.add_subparsers(dest="live_cmd", required=True)
    ing = live_sub.add_parser("ingest", help="archive one forecast cycle, idempotently")
    ing.add_argument("--source", choices=list(config.SYSTEMS), required=True)
    ing.add_argument(
        "--cycle",
        default=None,
        help="initialisation time, for example 2026-07-01T00; default is the latest published 6-hourly cycle",
    )
    ing.add_argument("--archive", default=None, help="local directory or s3://bucket/prefix")
    score = live_sub.add_parser("score", help="score settled cycles")
    score.add_argument("--archive", default=None)

    for name in ("feasibility", "evaluate", "figures", "page", "run"):
        s = sub.add_parser(name)
        s.add_argument("--period", choices=list(config.PERIODS), default="dev")
        s.add_argument("--synthetic", action="store_true", help="use the synthetic data root")
        s.add_argument("--test-end", default=None, help="last issue date for the open ended test period")
        if name in ("evaluate", "run"):
            s.add_argument("--open-sealed", action="store_true", help="open the sealed test period, once")
            s.add_argument("--no-sensitivities", action="store_true")

    sub.add_parser("demo")
    rp = sub.add_parser("reproduce", help="download everything, then run a period")
    rp.add_argument("--period", choices=[k for k in config.PERIODS if k not in config.SEALED_PERIODS], default="dev")
    a = p.parse_args(argv)

    if a.cmd == "reproduce":
        from gbwind import archive, curtailment, elexon, portfolio

        units = [u for f in portfolio.load_yaml(DEFAULT_PATHS.portfolio_yaml)["farms"] for u in f["bmunits"]]
        elexon.save_bmunit_reference(DEFAULT_PATHS)
        elexon.fetch_all(units, config.HISTORY_START, pd.Timestamp.now().strftime("%Y-%m-%d"), DEFAULT_PATHS)
        portfolio.build(DEFAULT_PATHS)
        curtailment.run(DEFAULT_PATHS)
        portfolio.select(DEFAULT_PATHS)
        start, end = config.period_bounds(a.period)
        farms = pd.read_csv(DEFAULT_PATHS.portfolio_csv)
        for system in config.SYSTEMS:
            archive.extract(system, str(start.date()), str(end.date()), farms, DEFAULT_PATHS)
        return main(["run", "--period", a.period])

    if a.cmd == "portfolio":
        from gbwind import portfolio

        df = portfolio.build(DEFAULT_PATHS) if a.action == "build" else portfolio.select(DEFAULT_PATHS)
        print(df.to_string(index=False))
        return 0
    if a.cmd == "curtailment":
        from gbwind import curtailment

        curtailment.run(DEFAULT_PATHS)
        return 0
    if a.cmd == "archive":
        from gbwind import archive

        farms = pd.read_csv(DEFAULT_PATHS.portfolio_csv)
        archive.extract(a.system, a.start, a.end, farms, DEFAULT_PATHS, workers=a.workers)
        return 0
    if a.cmd == "verify":
        from gbwind import archive

        v = archive.verify(a.system, DEFAULT_PATHS)
        print(v.describe().to_string())
        print(f"inits with any missing member value: {(v['finite_fraction'] < 1).sum()} of {len(v)}")
        return 0
    if a.cmd == "live":
        return _cmd_live(a)

    paths = SYNTHETIC_PATHS if a.cmd == "demo" else _paths(a)
    if a.cmd == "demo":
        from gbwind import synthetic

        print("Generating synthetic world (planted: AIFS-like arm has too weak cross farm dependence)")
        synthetic.generate(paths)
        a.period, a.test_end, a.open_sealed, a.no_sensitivities, a.synthetic = "dev", None, False, False, True
        a.cmd = "run"

    if a.cmd == "feasibility":
        cmd_feasibility(paths, a.period, a.test_end)
        return 0
    if a.cmd in ("evaluate", "run"):
        from gbwind.pipeline import run_period

        if a.cmd == "run" and not a.synthetic and a.period not in config.SEALED_PERIODS:
            cmd_feasibility(paths, a.period, a.test_end)
        report = run_period(a.period, paths, open_sealed=a.open_sealed, test_end=a.test_end,
                            sensitivities=not a.no_sensitivities)
        print(json.dumps(report, indent=2, default=str))
    if a.cmd in ("figures", "run"):
        from gbwind import figures

        for f in figures.make_all(paths.period_results(a.period), synthetic=a.synthetic):
            print(f"figure: {f}")
    if a.cmd in ("page", "run"):
        from gbwind import page
        from gbwind.portfolio import load_selected

        out = page.build(paths.period_results(a.period), load_selected(paths), synthetic=a.synthetic)
        print(f"results page: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
