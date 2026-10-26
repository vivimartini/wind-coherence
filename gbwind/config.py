"""Every fixed choice in DESIGN.md, in one place.

Nothing here is tuned against results. Changing a value is an amendment and
belongs in the DESIGN.md log with a date.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

N_MEMBERS = 51
SEED = 20260925

# Section 5. The 00 UTC run, 24 to 47 hours ahead. The primary steps are the ones
# shared by AIFS ENS (6 hourly) and IFS ENS (3 hourly) inside that window.
ISSUE_HOUR_UTC = 0
LEAD_WINDOW_HOURS = (24, 47)
LEAD_HOURS_PRIMARY = (24, 30, 36, 42)
LEAD_HOURS_IFS_SENSITIVITY = (24, 27, 30, 33, 36, 39, 42, 45)

# Section 6. Periods. Test is sealed and opened once.
PERIODS: dict[str, tuple[str, str | None]] = {
    "v1": ("2025-07-02", "2026-04-30"),
    "dev": ("2026-05-12", "2026-06-30"),
    "test": ("2026-07-01", None),
}
SEALED_PERIODS = {"test"}

# Section 4. Template pool construction.
PUBLICATION_LAG_WORKING_DAYS = 5
SEASONAL_HALF_WINDOW_DAYS = 21
SEASONAL_WIDEN_STEP_DAYS = 7
SEASONAL_HALF_WINDOW_MAX_DAYS = 91
TEMPLATE_DRAWS = 20
HISTORY_START = "2020-09-01"

# Section 7. Uncertainty.
BLOCK_LENGTH_DAYS = 5
BLOCK_LENGTH_SENSITIVITY_DAYS = (3, 10)
N_BOOTSTRAP = 2000
COVERAGE_LEVELS = (0.5, 0.8, 0.9)
VARIOGRAM_ORDER = 0.5
AGGREGATION_SUBSETS_PER_SIZE = 20

# Power conversion, fixed in the execution note.
CUT_IN_MS = 3.0
RATED_MS = 12.5
CUT_OUT_MS = 25.0
HUB_FROM_TIP_RATIO = 0.62
DEFAULT_HUB_HEIGHT_M = {"offshore": 100.0, "onshore": 80.0}

# Capacity used to scale the power curve: this quantile of uncurtailed, positive
# price, metered output over the history before the development period.
CAPACITY_QUANTILE = 0.995
CAPACITY_WINDOW_END = "2026-05-12"

# A BM unit is curtailed in a settlement period when an accepted instruction takes
# it below (1 - threshold) of its physical notification.
CURTAILMENT_THRESHOLD = 0.15

# Settlement periods whose market index price is negative, in any provider with
# traded volume, are excluded at every farm.
NEGATIVE_PRICE_THRESHOLD = 0.0

# Invariance tolerance for the sanity condition, in MW of CRPS.
INVARIANCE_ATOL = 1e-9

# Data sources.
ELEXON_API = "https://data.elexon.co.uk/bmrs/api/v1"
STAC_CATALOG = "https://stac.dynamical.org/catalog.json"
REPD_URL = (
    "https://assets.publishing.service.gov.uk/media/6a6cbdc00c36759b5ccaa305/"
    "REPD_Publication_Q2_2026.csv"
)
SYSTEMS = {
    "aifs-ens": "ecmwf-aifs-ens-forecast",
    "ifs-ens": "ecmwf-ifs-ens-forecast-15-day-0-25-degree",
}
CROSS_PARTNER = {"aifs-ens": "ifs-ens", "ifs-ens": "aifs-ens"}
WIND_VARIABLES = ("wind_u_100m", "wind_v_100m", "wind_u_10m", "wind_v_10m")
GB_BBOX = {"lat": (49.0, 61.5), "lon": (-9.0, 3.0)}

COUPLINGS = ("native", "historical", "independent", "cross")


@dataclass(frozen=True)
class Paths:
    """All file locations derive from one root so the synthetic demo can run in isolation."""

    data: Path
    results: Path
    config: Path = ROOT / "config"

    @property
    def raw_elexon(self) -> Path:
        return self.data / "raw" / "elexon"

    @property
    def raw_repd(self) -> Path:
        return self.data / "raw" / "repd.csv"

    @property
    def raw_bmunits(self) -> Path:
        return self.data / "raw" / "bmunits.json"

    @property
    def interim(self) -> Path:
        return self.data / "interim"

    @property
    def portfolio_csv(self) -> Path:
        return self.interim / "portfolio.csv"

    @property
    def farm_sp(self) -> Path:
        return self.interim / "farm_sp.parquet"

    @property
    def forecasts(self) -> Path:
        return self.data / "forecasts"

    def forecast_store(self, system: str) -> Path:
        return self.forecasts / f"{system}.zarr"

    def period_results(self, period: str) -> Path:
        return self.results / period

    @property
    def seal_dir(self) -> Path:
        return self.data / "seal"

    @property
    def portfolio_yaml(self) -> Path:
        return self.config / "portfolio.yaml"


DEFAULT_PATHS = Paths(data=ROOT / "data", results=ROOT / "results")
SYNTHETIC_PATHS = Paths(data=ROOT / "data_synthetic", results=ROOT / "results" / "synthetic")


def period_bounds(period: str, test_end: str | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Inclusive issue date bounds for a period."""
    start, end = PERIODS[period]
    if end is None:
        if test_end is None:
            raise ValueError(f"period {period!r} is open ended, pass its end date explicitly")
        end = test_end
    return pd.Timestamp(start), pd.Timestamp(end)


def aifs_version(init_time: pd.Timestamp) -> str:
    """AIFS ENS v2 arrived with IFS cycle 50r1 on 12 May 2026."""
    return "v2" if pd.Timestamp(init_time) >= pd.Timestamp("2026-05-12") else "v1"
