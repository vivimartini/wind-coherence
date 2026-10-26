"""Portfolio definition: REPD join, BM unit validation, metering based selection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml
from pyproj import Transformer

from gbwind import config
from gbwind.config import DEFAULT_PATHS, Paths

_OSGB_TO_WGS84 = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def load_repd(paths: Paths = DEFAULT_PATHS, url: str = config.REPD_URL) -> pd.DataFrame:
    if not paths.raw_repd.exists():
        paths.raw_repd.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        paths.raw_repd.write_bytes(r.content)
    df = pd.read_csv(paths.raw_repd, encoding="latin1", low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def hub_height(farm_type: str, tip_height_m: float | None, override: float | None) -> tuple[float, str]:
    if override is not None:
        return float(override), "portfolio.yaml"
    if tip_height_m is not None and np.isfinite(tip_height_m) and tip_height_m > 0:
        return config.HUB_FROM_TIP_RATIO * float(tip_height_m), "0.62 x REPD tip height"
    return config.DEFAULT_HUB_HEIGHT_M[farm_type], f"default {farm_type}"


def build(paths: Paths = DEFAULT_PATHS) -> pd.DataFrame:
    """Resolve coordinates, capacity and hub height for every candidate farm."""
    spec = load_yaml(paths.portfolio_yaml)
    repd = load_repd(paths)
    repd["Ref ID"] = pd.to_numeric(repd["Ref ID"], errors="coerce")
    bmu = pd.DataFrame(json.loads(paths.raw_bmunits.read_text()))
    bmu = bmu.drop_duplicates("elexonBmUnit").set_index("elexonBmUnit")

    rows = []
    for farm in spec["farms"]:
        missing = [u for u in farm["bmunits"] if u not in bmu.index]
        not_wind = [u for u in farm["bmunits"] if u in bmu.index and bmu.loc[u, "fuelType"] != "WIND"]
        if missing or not_wind:
            raise ValueError(f"{farm['id']}: unknown BM units {missing}, non wind {not_wind}")
        cap = float(pd.to_numeric(bmu.loc[farm["bmunits"], "generationCapacity"]).sum())

        r = repd[repd["Ref ID"].isin(farm["repd_refs"])]
        if len(r) != len(farm["repd_refs"]):
            raise ValueError(f"{farm['id']}: REPD refs {farm['repd_refs']} not all found")
        w = pd.to_numeric(r["Installed Capacity (MWelec)"], errors="coerce").fillna(1.0).to_numpy()
        x = pd.to_numeric(r["X-coordinate"], errors="coerce").to_numpy()
        y = pd.to_numeric(r["Y-coordinate"], errors="coerce").to_numpy()
        lon, lat = _OSGB_TO_WGS84.transform(np.average(x, weights=w), np.average(y, weights=w))
        tip = pd.to_numeric(r["Height of Turbines (m)"], errors="coerce").max()
        hub, hub_src = hub_height(farm["type"], tip, farm.get("hub_height_m"))
        rows.append(
            {
                "farm": farm["id"],
                "name": farm["name"],
                "type": farm["type"],
                "lat": round(float(lat), 4),
                "lon": round(float(lon), 4),
                "capacity_mw": cap,
                "registered_capacity_mw": cap,
                "repd_capacity_mw": float(w.sum()),
                "tip_height_m": None if not np.isfinite(tip) else float(tip),
                "hub_height_m": round(hub, 1),
                "hub_height_source": hub_src,
                "bmunits": " ".join(farm["bmunits"]),
            }
        )
    df = pd.DataFrame(rows)
    paths.interim.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.portfolio_csv, index=False)
    return df


def effective_capacity(
    farm_sp: pd.DataFrame,
    quantile: float = config.CAPACITY_QUANTILE,
    window_end: str = config.CAPACITY_WINDOW_END,
) -> pd.DataFrame:
    """Per farm quantile of admissible metered output before the development period.

    Also the same quantile per calendar year, so drift from repowering, export limits
    or partial commissioning is visible rather than averaged away.
    """
    from gbwind.curtailment import admissible_mask

    d = farm_sp[(farm_sp["sp_start"] < pd.Timestamp(window_end)) & admissible_mask(farm_sp)]
    d = d[d["mw"].notna()]
    overall = d.groupby("farm")["mw"].quantile(quantile).rename("effective_capacity_mw")
    yearly = d.groupby(["farm", d["sp_start"].dt.year])["mw"].quantile(quantile).unstack()
    yearly.columns = [f"q{int(quantile * 1000)}_{y}" for y in yearly.columns]
    return pd.concat([overall, yearly], axis=1)


def select(paths: Paths = DEFAULT_PATHS, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Apply the metering rule, set effective capacity, and freeze the portfolio.

    Uses only metering, curtailment and price flags, never forecasts, so it can run
    before any score is computed.
    """
    spec = load_yaml(paths.portfolio_yaml)
    thresh = spec["selection"]["min_metered_fraction"]
    portfolio = pd.read_csv(paths.portfolio_csv)
    portfolio = portfolio[[c for c in portfolio.columns if not c.startswith("q995_")]]
    portfolio = portfolio.drop(
        columns=["metered_fraction", "curtailed_fraction", "effective_capacity_mw", "selected"], errors="ignore"
    )
    farm_sp = pd.read_parquet(paths.farm_sp)
    if start:
        farm_sp = farm_sp[farm_sp["sp_start"] >= pd.Timestamp(start)]
    if end:
        farm_sp = farm_sp[farm_sp["sp_start"] < pd.Timestamp(end)]
    g = farm_sp.groupby("farm")
    stats = pd.DataFrame(
        {"metered_fraction": g["metered"].mean(), "curtailed_fraction": g["curtailed"].mean()}
    )
    portfolio = portfolio.merge(stats, left_on="farm", right_index=True, how="left")
    portfolio = portfolio.merge(effective_capacity(farm_sp), left_on="farm", right_index=True, how="left")
    portfolio["capacity_mw"] = portfolio["effective_capacity_mw"]
    portfolio["selected"] = (portfolio["metered_fraction"] >= thresh) & portfolio["capacity_mw"].gt(0)
    portfolio.to_csv(paths.portfolio_csv, index=False)
    return portfolio


def load_selected(paths: Paths = DEFAULT_PATHS) -> pd.DataFrame:
    df = pd.read_csv(paths.portfolio_csv)
    if "selected" in df:
        df = df[df["selected"].astype(bool)]
    return df.reset_index(drop=True)
