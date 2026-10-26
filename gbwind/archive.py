"""dynamical.org Icechunk archives: open, extract members at farm locations, save.

Each 00 UTC run is read as one GB box per variable (chunks are 8 degrees square, so
this touches a handful of chunks), bilinearly interpolated to farm locations, and
appended to a local Zarr store with dims (init_time, lead_time, member, farm).
Saved forecasts are a deliverable, so the store is kept in the repository data.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr

from gbwind import config
from gbwind.config import DEFAULT_PATHS, Paths


def open_system(system: str) -> xr.Dataset:
    import icechunk
    import pystac

    catalog = pystac.Catalog.from_file(config.STAC_CATALOG)
    collection = catalog.get_child(config.SYSTEMS[system])
    if collection is None:
        raise KeyError(f"{config.SYSTEMS[system]} not in dynamical.org STAC catalog")
    repo = icechunk.Repository.open(icechunk.http_storage(collection.assets["icechunk-https"].href))
    return xr.open_zarr(repo.readonly_session("main").store, chunks=None)


def _gb_box(ds: xr.Dataset) -> xr.Dataset:
    (la0, la1), (lo0, lo1) = config.GB_BBOX["lat"], config.GB_BBOX["lon"]
    lat = ds["latitude"]
    lat_slice = slice(la1, la0) if float(lat[0]) > float(lat[-1]) else slice(la0, la1)
    return ds.sel(latitude=lat_slice, longitude=slice(lo0, lo1))


def extract_init(ds: xr.Dataset, init_time: pd.Timestamp, leads_h, farms: pd.DataFrame) -> xr.Dataset | None:
    leads = pd.to_timedelta(list(leads_h), unit="h")
    try:
        sub = _gb_box(ds[list(config.WIND_VARIABLES)].sel(init_time=init_time, lead_time=leads))
    except KeyError:
        return None
    sub = sub.load().sortby("latitude")
    pts = sub.interp(
        latitude=xr.DataArray(farms["lat"].to_numpy(), dims="farm"),
        longitude=xr.DataArray(farms["lon"].to_numpy(), dims="farm"),
        method="linear",
    )
    pts = pts.drop_vars(["latitude", "longitude"], errors="ignore").assign_coords(
        farm=farms["farm"].to_numpy()
    )
    pts = pts.rename({"ensemble_member": "member"}).expand_dims(init_time=[np.datetime64(init_time, "ns")])
    return pts.transpose("init_time", "lead_time", "member", "farm").astype("float32")


def existing_inits(store) -> set[pd.Timestamp]:
    try:
        return set(pd.DatetimeIndex(xr.open_zarr(store)["init_time"].values))
    except Exception:
        return set()


def extract(
    system: str,
    start: str,
    end: str,
    farms: pd.DataFrame,
    paths: Paths = DEFAULT_PATHS,
    leads_h=None,
    workers: int = 4,
) -> None:
    """Append every 00 UTC init in [start, end] that is not already saved."""
    if leads_h is None:
        leads_h = config.LEAD_HOURS_IFS_SENSITIVITY if system == "ifs-ens" else config.LEAD_HOURS_PRIMARY
    store = paths.forecast_store(system)
    store.parent.mkdir(parents=True, exist_ok=True)
    ds = open_system(system)
    available = pd.DatetimeIndex(ds["init_time"].values)
    wanted = pd.date_range(start, end, freq="D") + pd.Timedelta(hours=config.ISSUE_HOUR_UTC)
    todo = [t for t in wanted if t in available and t not in existing_inits(store)]
    missing = [t for t in wanted if t not in available]
    if missing:
        print(f"{system}: {len(missing)} inits absent from archive, e.g. {missing[:3]}")

    def job(t):
        return t, extract_init(ds, t, leads_h, farms)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for t, piece in pool.map(job, todo):
            if piece is None:
                print(f"{system} {t:%Y-%m-%d}: not readable, skipped")
                continue
            n_nan = int(piece["wind_u_100m"].isnull().sum())
            first = not store.exists()
            piece.to_zarr(store, mode="w" if first else "a", append_dim=None if first else "init_time")
            print(f"{system} {t:%Y-%m-%d}: saved, {n_nan} missing member values", flush=True)


def load_forecasts(system: str, paths: Paths = DEFAULT_PATHS) -> xr.Dataset:
    ds = xr.open_zarr(paths.forecast_store(system)).load()
    return ds.sortby("init_time")


def verify(system: str, paths: Paths = DEFAULT_PATHS) -> pd.DataFrame:
    """Members present and finite at every farm, per init."""
    ds = load_forecasts(system, paths)
    u = ds["wind_u_100m"]
    return pd.DataFrame(
        {
            "members": int(ds.sizes["member"]),
            "finite_fraction": u.notnull().mean(dim=("lead_time", "member", "farm")).to_series(),
        }
    )
