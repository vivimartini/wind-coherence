"""Feasibility report, before any result (DESIGN.md section 4).

Uses metering, curtailment and price flags only. No forecast is scored, so it can
be read before the development period is evaluated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gbwind import config
from gbwind.config import Paths
from gbwind.curtailment import admissible_mask, centred_hourly
from gbwind.template import TemplateHistory, feasibility


def _md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float | np.floating) else str(v) for v in row]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def joint_attribution(farm_sp: pd.DataFrame) -> pd.DataFrame:
    """Why centred hours fail joint admissibility, one reason per hour in priority order."""
    fs = farm_sp.copy()
    flags = {}
    for name, mask in (
        ("unmetered", ~fs["metered"].astype(bool)),
        ("curtailed", fs["curtailed"].astype(bool)),
        ("negative_price", fs.get("negative_price", pd.Series(False, index=fs.index)).astype(bool)),
    ):
        flags[name] = fs.assign(f=mask.to_numpy()).pivot(index="sp_start", columns="farm", values="f").any(axis=1)
    f = pd.DataFrame(flags).sort_index()
    prev = f.shift(1, freq="30min")
    idx = f.index[f.index.minute == 0]
    idx = idx[idx.isin(prev.index)]
    hour = f.loc[idx] | prev.reindex(idx).fillna(False).astype(bool)
    reason = pd.Series("admissible", index=idx)
    for name in ("negative_price", "curtailed", "unmetered"):
        reason[hour[name].to_numpy()] = name
    return reason.to_frame("reason")


CALM_BIAS_NOTE = (
    "Curtailed hours carry reduced metered output, so the all hours row understates the "
    "windy tail. A gap in the same direction is therefore a lower bound on the calm bias "
    "that the admissibility rule introduces into the template."
)


def calm_bias(values: pd.DataFrame, admissible: pd.DataFrame, capacity: pd.Series) -> pd.DataFrame:
    """Portfolio capacity factor over all fully metered hours against the admissible subset."""
    cap = capacity.reindex(values.columns)
    full = values.notna().all(axis=1)
    cf = values.sum(axis=1) / cap.sum()
    adm = admissible.all(axis=1) & full
    rows = {}
    for name, m in (("all fully metered hours", full), ("admissible at every farm", adm)):
        x = cf[m]
        rows[name] = {
            "hours": int(m.sum()),
            "mean_cf": float(x.mean()),
            "p10": float(x.quantile(0.1)),
            "median": float(x.median()),
            "p90": float(x.quantile(0.9)),
            "share_cf_above_0.6": float((x > 0.6).mean()),
            "share_cf_below_0.1": float((x < 0.1).mean()),
        }
    out = pd.DataFrame(rows).T
    out.index.name = "subset"
    return out


def pool_table(history: TemplateHistory, period: str, test_end: str | None) -> tuple[pd.DataFrame, dict]:
    start, end = config.period_bounds(period, test_end)
    issues = pd.date_range(start, end, freq="D") + pd.Timedelta(hours=config.ISSUE_HOUR_UTC)
    cases = [(i, i + pd.Timedelta(hours=h)) for i in issues for h in config.LEAD_HOURS_PRIMARY]
    df = feasibility(history, cases)
    s = {
        "targets": len(df),
        "pool_at_21d_min": int(df["pool_at_21d"].min()),
        "pool_at_21d_p05": float(df["pool_at_21d"].quantile(0.05)),
        "pool_at_21d_median": float(df["pool_at_21d"].median()),
        "pool_at_21d_max": int(df["pool_at_21d"].max()),
        "widened_share": float((df["half_window_used"] > config.SEASONAL_HALF_WINDOW_DAYS).mean()),
        "max_half_window_used": int(df["half_window_used"].max()),
        "infeasible_share": float((~df["feasible"]).mean()),
    }
    return df, s


def build_report(paths: Paths, periods: tuple[str, ...] = ("dev", "v1")) -> str:
    portfolio = pd.read_csv(paths.portfolio_csv)
    sel = portfolio[portfolio["selected"].astype(bool)] if "selected" in portfolio else portfolio
    farms = sel["farm"].tolist()
    farm_sp = pd.read_parquet(paths.farm_sp)
    farm_sp = farm_sp[farm_sp["farm"].isin(farms)]
    values, admissible = centred_hourly(farm_sp)
    history = TemplateHistory(values, admissible, farms)
    outdir = paths.results / "feasibility"
    outdir.mkdir(parents=True, exist_ok=True)

    g = farm_sp.assign(adm=admissible_mask(farm_sp).to_numpy()).groupby("farm")
    zero = farm_sp[admissible_mask(farm_sp)].groupby("farm")["mw"].apply(lambda s: float((s <= 0).mean()))
    farm_tab = sel.set_index("farm")[
        [c for c in ("type", "registered_capacity_mw", "repd_capacity_mw", "effective_capacity_mw", "hub_height_m", "hub_height_source") if c in sel]
    ].copy()
    farm_tab["effective_over_registered"] = farm_tab.get("effective_capacity_mw", np.nan) / farm_tab.get("registered_capacity_mw", np.nan)
    farm_tab["metered"] = g["metered"].mean()
    farm_tab["curtailed_15pct"] = g["curtailed"].mean()
    if "any_boa" in farm_sp:
        farm_tab["any_acceptance"] = g["any_boa"].mean()
    farm_tab["admissible"] = g["adm"].mean()
    farm_tab["zero_output_share"] = zero
    farm_tab.index.name = "farm"
    yearly = sel.set_index("farm")[[c for c in sel.columns if c.startswith("q995_")]]
    yearly.index.name = "farm"

    reasons = joint_attribution(farm_sp)
    by_year = reasons.groupby(reasons.index.year)["reason"].value_counts(normalize=True).unstack(fill_value=0.0)
    by_year.index.name = "year"
    neg_share = float(farm_sp.drop_duplicates("sp_start")["negative_price"].mean()) if "negative_price" in farm_sp else float("nan")
    bias = calm_bias(values, admissible, sel.set_index("farm")["capacity_mw"])

    pools = {}
    for period in periods:
        test_end = None
        if period in config.SEALED_PERIODS:
            from gbwind.pipeline import default_test_end

            test_end = default_test_end()
        df, s = pool_table(history, period, test_end)
        df.to_csv(outdir / f"pool_{period}.csv", index=False)
        pools[period] = s
    pool_df = pd.DataFrame(pools).T
    pool_df.index.name = "period"

    farm_tab.to_csv(outdir / "farms.csv")
    by_year.to_csv(outdir / "joint_admissibility_by_year.csv")
    bias.to_csv(outdir / "calm_bias.csv")

    admissible_days = pd.DatetimeIndex(history.times).normalize().nunique()
    text = f"""# Feasibility report

Generated {pd.Timestamp.now(tz="UTC"):%Y-%m-%d %H:%M} UTC, before any forecast is scored.
Metered history {farm_sp["sp_start"].min():%Y-%m-%d} to {farm_sp["sp_start"].max():%Y-%m-%d}, {len(farms)} farms.

Rules applied: capacity is the {config.CAPACITY_QUANTILE:.1%} quantile of admissible metered output before
{config.CAPACITY_WINDOW_END}; a unit is curtailed when an accepted instruction takes it below
{1 - config.CURTAILMENT_THRESHOLD:.0%} of its physical notification; settlement periods with a negative market
index price are excluded at every farm.

## Template history

- Centred hours admissible at every farm: **{len(history):,}** of {len(values):,} ({len(history) / max(1, len(values)):.1%})
- Days with at least one admissible hour: **{admissible_days:,}**
- Settlement periods with a negative market index price: {neg_share:.2%}

## Template pool size per target (need at least {config.N_MEMBERS})

{_md(pool_df, "{:.3f}")}

## Is the admissible history biased towards calm weather?

{_md(bias)}

{CALM_BIAS_NOTE}

## Why hours fail joint admissibility, by year

Priority order when several apply: unmetered, then curtailed, then negative price.

{_md(by_year)}

## Farms

{_md(farm_tab)}

## Effective capacity by year ({config.CAPACITY_QUANTILE:.1%} quantile, MW)

{_md(yearly, "{:.0f}")}
"""
    (outdir / "REPORT.md").write_text(text)
    return text
