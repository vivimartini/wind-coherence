"""A synthetic world with a planted answer, for tests and the demo.

Nothing produced here is a result about real forecasts. Everything written under
the synthetic data root is labelled synthetic, and the results page shows a
banner when built from it.

Truth is a latent Gaussian field over farms with exponential spatial correlation
(length TRUE_LENGTH_KM) and AR(1) persistence, mapped to wind speed through a
Weibull marginal and to power through the project power curve. For each case a
predictable signal s is drawn so that truth = rho * s + sqrt(1 - rho^2) * e0 with
e0 at the true length scale, and members are rho * s + sqrt(1 - rho^2) * e_m,
where e_m has the system's own length scale. With the true length scale, truth
and members are exchangeable given s, so the ensemble is calibrated in both
marginals and dependence. With a shorter one the marginals are identical in distribution but
cross farm dependence is too weak, which the historical coupling should detect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import ndtr

from gbwind import config
from gbwind.config import SYNTHETIC_PATHS, Paths

TRUE_LENGTH_KM = 450.0
SYSTEM_LENGTH_KM = {"ifs-ens": TRUE_LENGTH_KM, "aifs-ens": 60.0}
RHO = 0.8
AR_PHI_PER_SP = 0.993
WEIBULL_K, WEIBULL_SCALE = 2.0, 9.5
SHEAR_10M_RATIO = 0.72

FARMS = [
    ("syn_moray", "offshore", 57.9, -2.9, 600.0),
    ("syn_forth", "offshore", 56.6, -1.8, 800.0),
    ("syn_hornsea", "offshore", 53.9, 1.9, 1200.0),
    ("syn_lincs", "offshore", 53.2, 0.9, 500.0),
    ("syn_thames", "offshore", 51.6, 1.4, 630.0),
    ("syn_irish", "offshore", 54.0, -3.6, 650.0),
    ("syn_wales", "offshore", 53.45, -3.6, 570.0),
    ("syn_sussex", "offshore", 50.7, -0.3, 400.0),
    ("syn_ayrshire", "onshore", 55.7, -4.3, 500.0),
    ("syn_highland", "onshore", 57.1, -4.6, 230.0),
    ("syn_borders", "onshore", 55.8, -2.6, 150.0),
    ("syn_valleys", "onshore", 51.7, -3.6, 230.0),
]


def portfolio_frame() -> pd.DataFrame:
    df = pd.DataFrame(FARMS, columns=["farm", "type", "lat", "lon", "capacity_mw"])
    df["name"] = df["farm"].str.replace("syn_", "Synthetic ").str.title()
    df["hub_height_m"] = 100.0
    df["hub_height_source"] = "synthetic"
    df["selected"] = True
    return df


def distance_km(lat, lon) -> np.ndarray:
    la, lo = np.radians(lat), np.radians(lon)
    dla = la[:, None] - la[None, :]
    dlo = lo[:, None] - lo[None, :]
    a = np.sin(dla / 2) ** 2 + np.cos(la[:, None]) * np.cos(la[None, :]) * np.sin(dlo / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def chol(dist: np.ndarray, length_km: float) -> np.ndarray:
    return np.linalg.cholesky(np.exp(-dist / length_km) + 1e-9 * np.eye(dist.shape[0]))


def wind_speed(z: np.ndarray) -> np.ndarray:
    u = np.clip(ndtr(z), 1e-9, 1 - 1e-9)
    return WEIBULL_SCALE * (-np.log1p(-u)) ** (1 / WEIBULL_K)


def generate(
    paths: Paths = SYNTHETIC_PATHS,
    history_start: str = "2022-01-01",
    end: str = "2026-09-10",
    init_start: str = "2025-07-02",
    init_end: str = "2026-09-01",
    curtail_rate: float = 0.02,
    seed: int = config.SEED,
) -> None:
    from gbwind.power import capacity_factor

    rng = np.random.default_rng([seed, 42])
    pf = portfolio_frame()
    F = len(pf)
    dist = distance_km(pf["lat"].to_numpy(), pf["lon"].to_numpy())
    L_true = chol(dist, TRUE_LENGTH_KM)

    sp = pd.date_range(history_start, end, freq="30min", inclusive="left")
    z = np.empty((len(sp), F))
    z[0] = L_true @ rng.standard_normal(F)
    innov = rng.standard_normal((len(sp), F)) @ L_true.T * np.sqrt(1 - AR_PHI_PER_SP**2)
    for t in range(1, len(sp)):
        z[t] = AR_PHI_PER_SP * z[t - 1] + innov[t]
    cap = pf["capacity_mw"].to_numpy()
    mw = capacity_factor(wind_speed(z)) * cap

    metered = rng.random(z.shape) > 0.003
    # Curtailment in contiguous 3 hour blocks, more likely in strong wind.
    starts = (rng.random(z.shape) < curtail_rate / 6) & (z > 0.5)
    curtailed = np.zeros_like(starts)
    for k in range(6):
        curtailed[k:] |= starts[: len(sp) - k]
    mw_obs = np.where(curtailed, mw * 0.4, mw)
    farm_sp = pd.DataFrame(
        {
            "farm": np.repeat(pf["farm"].to_numpy()[None, :], len(sp), axis=0).ravel(order="F"),
            "sp_start": np.tile(sp.values, F),
            "mw": np.where(metered, mw_obs, np.nan).ravel(order="F"),
            "metered": metered.ravel(order="F"),
            "curtailed": curtailed.ravel(order="F"),
        }
    )
    paths.interim.mkdir(parents=True, exist_ok=True)
    farm_sp.to_parquet(paths.farm_sp, index=False)
    pf.to_csv(paths.portfolio_csv, index=False)

    t_index = pd.Series(np.arange(len(sp)), index=sp)
    inits = pd.date_range(init_start, init_end, freq="D")
    paths.forecasts.mkdir(parents=True, exist_ok=True)
    for system, leads in (("aifs-ens", config.LEAD_HOURS_PRIMARY), ("ifs-ens", config.LEAD_HOURS_IFS_SENSITIVITY)):
        L_sys = chol(dist, SYSTEM_LENGTH_KM[system])
        srng = np.random.default_rng([seed, sum(map(ord, system))])
        ws = np.empty((len(inits), len(leads), config.N_MEMBERS, F), dtype="float32")
        for i, init in enumerate(inits):
            for j, lead in enumerate(leads):
                zt = z[t_index[init + pd.Timedelta(hours=lead)]]
                signal = RHO * zt + np.sqrt(1 - RHO**2) * (L_true @ srng.standard_normal(F))
                noise = srng.standard_normal((config.N_MEMBERS, F)) @ L_sys.T
                ws[i, j] = wind_speed(RHO * signal[None, :] + np.sqrt(1 - RHO**2) * noise)
        coords = {
            "init_time": inits.values,
            "lead_time": pd.to_timedelta(list(leads), unit="h").values,
            "member": np.arange(config.N_MEMBERS),
            "farm": pf["farm"].to_numpy(),
        }
        dims = ("init_time", "lead_time", "member", "farm")
        zeros = np.zeros_like(ws)
        ds = xr.Dataset(
            {
                "wind_u_100m": (dims, ws),
                "wind_v_100m": (dims, zeros),
                "wind_u_10m": (dims, ws * SHEAR_10M_RATIO),
                "wind_v_10m": (dims, zeros),
            },
            coords=coords,
            attrs={"synthetic": 1, "noise_length_km": SYSTEM_LENGTH_KM[system]},
        )
        ds.to_zarr(paths.forecast_store(system), mode="w")
