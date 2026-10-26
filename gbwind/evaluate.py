"""Evaluation: couple, score, check invariance, bootstrap, save.

A case is one (system, 00 UTC issue, lead). For each case the power ensemble X
(members x farms) is recoupled in four ways. Every coupled ensemble is checked to
leave per farm CRPS exactly unchanged; any difference voids the run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from gbwind import bootstrap, config, couplings, power, scores
from gbwind.template import TemplateHistory, feasibility


class VoidRun(RuntimeError):
    """Per farm CRPS changed under a coupling. The shuffle has leaked."""


@dataclass
class Inputs:
    portfolio: pd.DataFrame
    forecasts: dict[str, xr.Dataset]
    obs_values: pd.DataFrame
    obs_admissible: pd.DataFrame
    history: TemplateHistory
    leads: dict[str, tuple[int, ...]] = field(default_factory=dict)


def power_ensemble(ds: xr.Dataset, portfolio: pd.DataFrame) -> xr.DataArray:
    """(init_time, lead_time, member, farm) in MW, farms in portfolio order."""
    ds = ds.sel(farm=portfolio["farm"].to_numpy())
    hub = xr.DataArray(portfolio["hub_height_m"].to_numpy(float), dims="farm")
    cap = xr.DataArray(portfolio["capacity_mw"].to_numpy(float), dims="farm")
    mw = xr.apply_ufunc(
        power.farm_power_mw,
        ds["wind_u_100m"], ds["wind_v_100m"], ds["wind_u_10m"], ds["wind_v_10m"], hub, cap,
    )
    return mw.transpose("init_time", "lead_time", "member", "farm").astype("float64")


def aggregation_subsets(n_farms: int, per_size: int, seed: int) -> dict[int, list[np.ndarray]]:
    """Fixed farm subsets per size, shared by every case and coupling."""
    rng = np.random.default_rng([seed, 7])
    out: dict[int, list[np.ndarray]] = {}
    for k in range(1, n_farms + 1):
        if k == n_farms:
            out[k] = [np.arange(n_farms)]
            continue
        seen: set[tuple[int, ...]] = set()
        subs = []
        for _ in range(per_size * 5):
            s = tuple(sorted(rng.choice(n_farms, size=k, replace=False)))
            if s not in seen:
                seen.add(s)
                subs.append(np.array(s))
            if len(subs) == per_size:
                break
        out[k] = subs
    return out


def _case_rng(system: str, init: pd.Timestamp, lead: int, salt: int) -> np.random.Generator:
    sys_id = sum(map(ord, system))
    return np.random.default_rng([config.SEED, sys_id, int(pd.Timestamp(init).value // 10**9), lead, salt])


def score_realisation(Y, obs, cap, subsets, rng) -> tuple[dict, np.ndarray, dict[int, float]]:
    total = Y.sum(axis=1)
    obs_total = float(obs.sum())
    cap_total = float(cap.sum())
    cf, obs_cf = Y / cap, obs / cap
    rec = {
        "crps": float(scores.crps_ensemble(total, obs_total)) / cap_total,
        "es": scores.energy_score(cf, obs_cf),
        "vs": scores.variogram_score(cf, obs_cf, p=config.VARIOGRAM_ORDER),
        "rank": scores.rank_of_obs(total, obs_total, rng),
        "spearman": couplings.mean_pairwise_spearman(Y),
        "obs_total": obs_total,
        "ens_mean_total": float(total.mean()),
        "ens_sd_total": float(total.std()),
    }
    for lvl in config.COVERAGE_LEVELS:
        rec[f"cov{int(lvl * 100)}"] = scores.central_interval_hit(total, obs_total, lvl)
    per_farm = scores.crps_ensemble(Y.T, obs)
    agg = {
        k: float(np.mean([scores.crps_ensemble(Y[:, s].sum(axis=1), obs[s].sum()) / cap[s].sum() for s in subs]))
        for k, subs in subsets.items()
    }
    return rec, per_farm, agg


def evaluate_system(
    system: str,
    inp: Inputs,
    issue_start: pd.Timestamp,
    issue_end: pd.Timestamp,
    leads: tuple[int, ...],
    n_draws: int = config.TEMPLATE_DRAWS,
) -> dict:
    portfolio = inp.portfolio
    farms = portfolio["farm"].tolist()
    cap = portfolio["capacity_mw"].to_numpy(float)
    P = power_ensemble(inp.forecasts[system], portfolio)
    partner = config.CROSS_PARTNER.get(system)
    P_other = power_ensemble(inp.forecasts[partner], portfolio) if partner in inp.forecasts else None
    subsets = aggregation_subsets(len(farms), config.AGGREGATION_SUBSETS_PER_SIZE, config.SEED)

    inits = pd.DatetimeIndex(P["init_time"].values)
    inits = inits[(inits >= issue_start) & (inits <= issue_end + pd.Timedelta(hours=23))]
    inits = inits[inits.hour == config.ISSUE_HOUR_UTC]
    lead_index = pd.TimedeltaIndex(P["lead_time"].values)

    rows, agg_rows, excluded, feas_cases, ties = [], [], [], [], []
    saved = {"case": [], "coupled": [], "template_times": []}

    for init in inits:
        for lead in leads:
            valid = init + pd.Timedelta(hours=lead)
            key = {"system": system, "init_time": init, "lead_h": lead, "valid_time": valid}
            if pd.Timedelta(hours=lead) not in lead_index:
                excluded.append({**key, "reason": "lead not in archive"})
                continue
            X = P.sel(init_time=init, lead_time=pd.Timedelta(hours=lead)).to_numpy()
            if not np.isfinite(X).all():
                excluded.append({**key, "reason": "forecast missing"})
                continue
            if valid not in inp.obs_values.index or not inp.obs_admissible.loc[valid, farms].all():
                excluded.append({**key, "reason": "observation missing or curtailed"})
                continue
            obs = inp.obs_values.loc[valid, farms].to_numpy(float)
            feas_cases.append((init, valid))
            pool = inp.history.pool_with_widening(valid, init)
            if not pool.feasible:
                excluded.append({**key, "reason": f"template pool {pool.indices.size} < {config.N_MEMBERS}"})
                continue

            rng_t = _case_rng(system, init, lead, 1)
            drawn = inp.history.draw(pool.indices, n_draws, X.shape[0], rng_t)
            inp.history.assert_no_leak(drawn, init)

            realisations: list[tuple[str, int, np.ndarray]] = [("native", 0, couplings.native(X))]
            rng_c = _case_rng(system, init, lead, 2)
            for d in range(n_draws):
                H = inp.history.values[drawn[d]]
                ties.append(couplings.tie_fraction(H))
                realisations.append(("historical", d, couplings.historical(X, H, rng_c)))
            rng_i = _case_rng(system, init, lead, 3)
            for d in range(n_draws):
                realisations.append(("independent", d, couplings.independent(X, rng_i)))
            if P_other is not None:
                try:
                    Z = P_other.sel(init_time=init, lead_time=pd.Timedelta(hours=lead)).to_numpy()
                except KeyError:
                    Z = None
                if Z is not None and np.isfinite(Z).all() and Z.shape == X.shape:
                    realisations.append(("cross", 0, couplings.cross_model(X, Z, _case_rng(system, init, lead, 4))))

            rng_r = _case_rng(system, init, lead, 5)
            ref_per_farm = None
            for name, d, Y in realisations:
                rec, per_farm, agg = score_realisation(Y, obs, cap, subsets, rng_r)
                if ref_per_farm is None:
                    ref_per_farm = per_farm
                elif not np.allclose(per_farm, ref_per_farm, rtol=0, atol=config.INVARIANCE_ATOL):
                    raise VoidRun(f"{system} {init} +{lead}h {name}[{d}]: per farm CRPS changed")
                rows.append({**key, "coupling": name, "realisation": d,
                             "half_window_days": pool.half_window_days, "pool_size": int(pool.indices.size), **rec})
                agg_rows.extend({**key, "coupling": name, "realisation": d, "size": k, "crps": v} for k, v in agg.items())

            saved["case"].append((init, lead))
            first = {n: Y for n, d, Y in realisations if d == 0}
            saved["coupled"].append(np.stack([first.get(c, np.full_like(X, np.nan)) for c in config.COUPLINGS]))
            saved["template_times"].append(inp.history.times.values[drawn])

    return {
        "scores": pd.DataFrame(rows),
        "aggregation": pd.DataFrame(agg_rows),
        "excluded": pd.DataFrame(excluded),
        "feasibility": feasibility(inp.history, feas_cases),
        "tie_fraction": pd.Series(np.mean(ties, axis=0) if ties else np.nan, index=farms),
        "saved": saved,
        "farms": farms,
    }


def day_matrix(scores_df: pd.DataFrame, coupling: str, metric: str, days: pd.DatetimeIndex) -> np.ndarray:
    """(days, realisations): per issue day, mean over leads, one column per realisation."""
    d = scores_df[scores_df["coupling"] == coupling]
    if d.empty:
        return np.full((len(days), 1), np.nan)
    m = d.groupby(["init_time", "realisation"])[metric].mean().unstack("realisation")
    return m.reindex(days).to_numpy(float)


def summarise(result: dict, block_length: int = config.BLOCK_LENGTH_DAYS, n_boot: int = config.N_BOOTSTRAP) -> dict:
    s, a = result["scores"], result["aggregation"]
    if s.empty:
        return {"n_cases": 0, "note": "no evaluable cases"}
    days = pd.DatetimeIndex(sorted(s["init_time"].unique()))
    present = [c for c in config.COUPLINGS if (s["coupling"] == c).any()]
    rng = np.random.default_rng([config.SEED, 99, block_length])

    out: dict = {
        "n_days": len(days),
        "n_cases": int(s[s["coupling"] == "native"].shape[0]),
        "block_length_days": block_length,
        "n_bootstrap": n_boot,
        "couplings": present,
        "metrics": {},
        "differences_vs_native": {},
        "aggregation": {},
    }
    metrics = ["crps", "es", "vs"] + [f"cov{int(lv * 100)}" for lv in config.COVERAGE_LEVELS]
    for metric in metrics:
        out["metrics"][metric] = {
            c: bootstrap.mean_interval(day_matrix(s, c, metric, days), block_length, n_boot, rng) for c in present
        }
        ref = day_matrix(s, "native", metric, days)
        out["differences_vs_native"][metric] = {
            c: bootstrap.difference_interval(day_matrix(s, c, metric, days), ref, block_length, n_boot, rng)
            for c in present if c != "native"
        }
    for size in sorted(a["size"].unique()):
        sub = a[a["size"] == size]
        ref = day_matrix(sub, "native", "crps", days)
        out["aggregation"][int(size)] = {
            "native": float(np.nanmean(ref)),
            **{c: bootstrap.difference_interval(day_matrix(sub, c, "crps", days), ref, block_length, n_boot, rng)
               for c in present if c != "native"},
        }

    m = config.N_MEMBERS
    out["rank_histogram"] = {
        c: (np.bincount(s.loc[s["coupling"] == c, "rank"].astype(int), minlength=m + 1)
            / max(1, s.loc[s["coupling"] == c, "realisation"].nunique())).round(3).tolist()
        for c in present
    }
    out["realised_spearman"] = {c: float(s.loc[s["coupling"] == c, "spearman"].mean()) for c in present}
    f = result["feasibility"]
    out["template_pool"] = {
        "min_at_21d": int(f["pool_at_21d"].min()) if len(f) else None,
        "median_at_21d": float(f["pool_at_21d"].median()) if len(f) else None,
        "fraction_widened": float((f["half_window_used"] > config.SEASONAL_HALF_WINDOW_DAYS).mean()) if len(f) else None,
        "max_half_window_used": int(f["half_window_used"].max()) if len(f) else None,
        "fraction_infeasible": float((~f["feasible"]).mean()) if len(f) else None,
    }
    out["tie_fraction"] = result["tie_fraction"].round(4).to_dict()
    ex = result["excluded"]
    out["excluded"] = ex["reason"].value_counts().to_dict() if len(ex) else {}
    return out


def save(result: dict, summary: dict, outdir) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    result["scores"].to_parquet(outdir / "scores.parquet", index=False)
    result["aggregation"].to_parquet(outdir / "aggregation.parquet", index=False)
    result["excluded"].to_csv(outdir / "excluded.csv", index=False)
    result["feasibility"].to_csv(outdir / "template_pool.csv", index=False)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    sv = result["saved"]
    if sv["case"]:
        inits = np.array([np.datetime64(c[0], "ns") for c in sv["case"]])
        leads = np.array([c[1] for c in sv["case"]])
        ds = xr.Dataset(
            {
                "coupled_mw": (("case", "coupling", "member", "farm"), np.stack(sv["coupled"]).astype("float32")),
                "template_times": (("case", "draw", "member"), np.stack(sv["template_times"])),
            },
            coords={
                "init_time": ("case", inits),
                "lead_h": ("case", leads),
                "coupling": list(config.COUPLINGS),
                "farm": result["farms"],
            },
            attrs={"note": "Realisation 0 of each coupling. template_times are every drawn template, all draws."},
        )
        ds.to_zarr(outdir / "couplings.zarr", mode="w")
