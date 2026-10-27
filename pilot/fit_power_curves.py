"""Fit one smooth power curve per farm.

ERA5 100 m wind, bilinear from the four surrounding 0.25 degree points, same
interpolation as `gbwind.archive`. No hub-height step. Fit on uncurtailed hours
before 2025-07-02. Rated speed is pulled toward 12.5 m/s.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from gbwind import config
from gbwind.curtailment import centred_hourly
from gbwind.portfolio import effective_capacity
from gbwind.power import capacity_factor, smooth_capacity_factor

ROOT = Path(__file__).resolve().parents[1]
ERA5 = Path("/rds/user/vm481/hpc-work/wind-coherence/data/era5")
OUT = Path("/rds/user/vm481/hpc-work/wind-coherence/results/power-curves")
PARAMS = ROOT / "config" / "power_curves.json"
FIT_END = pd.Timestamp("2025-07-02")
DEV_START, DEV_END = pd.Timestamp("2026-05-12"), pd.Timestamp("2026-06-30 23:00")
ERA5_SPAN = "2020-09-01/2026-06-30"
GRID = 0.25
BINS = [0, 3, 5, 7, 9, 11, 13, 16, 20, 25, 40]
VARIABLES = [
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]
PRIOR_RATED_MS = config.RATED_MS
PRIOR_RATED_SD = 2.0
PRIOR_HOURS = 100


def corners(lat: float, lon: float) -> list[tuple[float, float]]:
    lat0 = np.floor(round(lat / GRID, 8)) * GRID
    lon0 = np.floor(round(lon / GRID, 8)) * GRID
    return [
        (round(float(lat0 + i * GRID), 2), round(float(lon0 + j * GRID), 2))
        for i in (0, 1)
        for j in (0, 1)
    ]


def point_path(lat: float, lon: float) -> Path:
    return ERA5 / f"{lat:.2f}_{lon:.2f}.zip"


def read_series(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()
    if raw[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            name = next(n for n in zf.namelist() if n.endswith(".csv"))
            raw = zf.read(name)
    frame = pd.read_csv(io.BytesIO(raw), parse_dates=["valid_time"]).set_index("valid_time").sort_index()
    frame.index = frame.index.tz_localize(None)
    return frame


def adopt_existing() -> None:
    """The first fit saved one nearest-point file per farm. Keep any that sit on a corner."""
    for path in ERA5.glob("*.zip"):
        if "_" in path.stem:
            continue
        frame = read_series(path)
        lat, lon = float(frame["latitude"].iloc[0]), float(frame["longitude"].iloc[0])
        dest = point_path(lat, lon)
        if not dest.exists():
            dest.write_bytes(path.read_bytes())
            print(f"kept {path.name} as {dest.name}", flush=True)


def download(lat: float, lon: float) -> Path:
    dest = point_path(lat, lon)
    if dest.exists() and dest.stat().st_size > 100_000:
        return dest
    import cdsapi

    ERA5.mkdir(parents=True, exist_ok=True)
    cdsapi.Client(quiet=True, progress=False, debug=False).retrieve(
        "reanalysis-era5-single-levels-timeseries",
        {
            "variable": VARIABLES,
            "location": {"longitude": lon, "latitude": lat},
            "date": [ERA5_SPAN],
            "data_format": "csv",
        },
        str(dest),
    )
    print(f"{dest.name}: {dest.stat().st_size} bytes", flush=True)
    return dest


def interpolate(frames: dict[tuple[float, float], pd.DataFrame], lat: float, lon: float) -> pd.DataFrame:
    """Bilinear weights on the 0.25 degree cell. Same result as xarray's linear interp."""
    lats = sorted({p[0] for p in frames})
    lons = sorted({p[1] for p in frames})
    times = next(iter(frames.values())).index
    ty = 0.0 if lats[1] == lats[0] else (lat - lats[0]) / (lats[1] - lats[0])
    tx = 0.0 if lons[1] == lons[0] else (lon - lons[0]) / (lons[1] - lons[0])
    weights = {
        (lats[0], lons[0]): (1.0 - ty) * (1.0 - tx),
        (lats[0], lons[1]): (1.0 - ty) * tx,
        (lats[1], lons[0]): ty * (1.0 - tx),
        (lats[1], lons[1]): ty * tx,
    }
    out = {}
    for var in ("u100", "v100", "u10", "v10"):
        acc = None
        for pt, w in weights.items():
            col = frames[pt][var].reindex(times).to_numpy(dtype=np.float64)
            acc = col * w if acc is None else acc + col * w
        out[var] = acc
    return pd.DataFrame(out, index=times)


def fit_curve(ws: np.ndarray, cf: np.ndarray) -> np.ndarray:
    def data_residual(p):
        return smooth_capacity_factor(ws, *p) - cf

    bounds = ([0.5, 6.0, 1.5], [8.0, 20.0, 20.0])
    rough = least_squares(data_residual, x0=[3.0, PRIOR_RATED_MS, 6.0], bounds=bounds)
    sigma = max(float(np.std(data_residual(rough.x))), 1e-3)

    def residual(p):
        prior = (p[1] - PRIOR_RATED_MS) / PRIOR_RATED_SD * sigma * np.sqrt(PRIOR_HOURS)
        return np.append(data_residual(p), prior)

    fitted = least_squares(residual, x0=rough.x, bounds=bounds).x
    return fitted, rough.x


def scores(mw: np.ndarray, pred: np.ndarray, capacity: float) -> dict:
    err = pred - mw
    return {
        "hours": int(len(mw)),
        "mean_error_mw": round(float(np.mean(err)), 2),
        "fraction_overpredicted": round(float(np.mean(err > 0)), 3),
        "mae_mw": round(float(np.mean(np.abs(err))), 2),
        "rmse_mw": round(float(np.sqrt(np.mean(err**2))), 2),
        "mae_cf": round(float(np.mean(np.abs(err)) / capacity), 4),
    }


def one_farm(farm: pd.Series, wind: pd.DataFrame, metered: pd.Series, admissible: pd.Series) -> dict:
    joined = pd.DataFrame({"mw": metered, "ok": admissible}).join(wind, how="inner")
    joined = joined.assign(ws=np.hypot(joined["u100"], joined["v100"]), cf=joined["mw"] / farm["capacity_mw"])
    fit = joined[(joined.index < FIT_END) & joined["ok"] & np.isfinite(joined["ws"])]
    params, rough = fit_curve(fit["ws"].to_numpy(), np.clip(fit["cf"].to_numpy(), 0, 1.5))
    pred = pd.Series(smooth_capacity_factor(joined["ws"].to_numpy(), *params) * farm["capacity_mw"], index=joined.index)
    dev = joined[(joined.index >= DEV_START) & (joined.index <= DEV_END) & joined["ok"]]
    counts = pd.cut(fit["ws"], BINS, right=False).value_counts().sort_index()
    record = {
        "farm": farm["farm"],
        "wind": "ERA5 100 m, bilinear from the four surrounding 0.25 degree points",
        "hub_height_applied": False,
        "capacity_mw": float(farm["capacity_mw"]),
        "fit_hours": int(len(fit)),
        "cut_in_ms": round(float(params[0]), 3),
        "rated_ms": round(float(params[1]), 3),
        "sharpness": round(float(params[2]), 3),
        "unpenalised_rated_ms": round(float(rough[1]), 3),
        "cut_out_ms": None,
        "rated_prior_ms": PRIOR_RATED_MS,
        "rated_prior_sd_ms": PRIOR_RATED_SD,
        "rated_prior_hours": PRIOR_HOURS,
        "uncurtailed_hours_by_wind_bin": {str(k): int(v) for k, v in counts.items()},
        "fit_set": scores(fit["mw"].to_numpy(), pred.loc[fit.index].to_numpy(), farm["capacity_mw"]),
        "development": scores(dev["mw"].to_numpy(), pred.loc[dev.index].to_numpy(), farm["capacity_mw"]),
    }
    figure(farm["farm"], fit, params, record)
    return record


def figure(name: str, fit: pd.DataFrame, params: np.ndarray, record: dict) -> None:
    centres = np.arange(0, 30, 0.25)
    binned = fit.groupby(pd.cut(fit["ws"], BINS, right=False), observed=True)["cf"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    ax = axes[0]
    ax.plot(centres, capacity_factor(centres), color="0.6", label="idealised")
    ax.plot(centres, smooth_capacity_factor(centres, *params), color="C0", label="fitted")
    ax.plot([iv.mid for iv in binned.index], binned.to_numpy(), "o", color="C1", label="uncurtailed mean")
    ax.set_xlim(0, 30)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("ERA5 100 m wind, m/s")
    ax.set_ylabel("capacity factor")
    ax.set_title(name)
    ax.legend(frameon=False, fontsize=8)
    err = record["development"]
    axes[1].hist(smooth_capacity_factor(fit["ws"], *params) - fit["cf"].clip(0, 1.5), bins=40, color="C0")
    axes[1].set_xlabel("fitted minus metered, capacity factor")
    axes[1].set_ylabel("fitting hours")
    axes[1].set_title(f"dev MAE {err['mae_mw']:.0f} MW, {100 * err['mae_cf']:.1f}% of capacity")
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=120)
    plt.close(fig)


def main() -> None:
    print("start", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    farms = pd.read_csv(ROOT / "data" / "interim" / "portfolio.csv")
    needed = {pt for lat, lon in zip(farms["lat"], farms["lon"], strict=True) for pt in corners(lat, lon)}
    if any(not point_path(*pt).exists() for pt in needed):
        adopt_existing()
        for lat, lon in sorted(needed):
            download(lat, lon)
    metering = pd.read_parquet(config.DEFAULT_PATHS.farm_sp)
    values, admissible = centred_hourly(metering)
    capacity = effective_capacity(metering)["effective_capacity_mw"]
    records = []
    loaded: dict[tuple[float, float], pd.DataFrame] = {}
    for farm in farms.itertuples(index=False):
        row = farms[farms["farm"] == farm.farm].iloc[0].copy()
        row["capacity_mw"] = float(capacity.loc[farm.farm])
        for pt in corners(farm.lat, farm.lon):
            loaded.setdefault(pt, read_series(point_path(*pt)))
        wind = interpolate({pt: loaded[pt] for pt in corners(farm.lat, farm.lon)}, farm.lat, farm.lon)
        record = one_farm(row, wind, values[farm.farm], admissible[farm.farm])
        records.append(record)
        brief = ("farm", "fit_hours", "cut_in_ms", "rated_ms", "unpenalised_rated_ms", "development")
        print(json.dumps({k: record[k] for k in brief}), flush=True)
    payload = {
        "wind": "ERA5 100 m bilinear, no hub-height extrapolation",
        "fit_end": str(FIT_END.date()),
        "floor": "development-date error with ERA5 100 m wind as the input",
        "cut_out": "not fitted and not applied; the curve stays smooth. The idealised reference cuts out at 25 m/s",
        "farms": {r["farm"]: r for r in records},
    }
    text = json.dumps(payload, indent=2)
    (OUT / "curves.json").write_text(text)
    PARAMS.write_text(text)
    print(f"wrote {OUT / 'curves.json'}", flush=True)


if __name__ == "__main__":
    main()
