"""End to end on a synthetic world with a planted answer.

The AIFS-like synthetic arm has members whose cross farm noise decorrelates over
60 km while the truth decorrelates over 450 km. Marginals are right, dependence is
too weak. The method must detect that (historical beats native), and must not
find a spurious gain on the IFS-like arm, whose dependence is correct.
"""

import json

import numpy as np
import pytest

from gbwind import config, synthetic
from gbwind.config import Paths
from gbwind.pipeline import run_period


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn")
    paths = Paths(data=root / "data", results=root / "results")
    synthetic.generate(paths, history_start="2023-01-01", end="2026-07-05",
                       init_start="2026-05-12", init_end="2026-06-30")
    report = run_period("dev", paths, sensitivities=False)
    return paths, report


def summary(paths, system):
    return json.loads((paths.period_results("dev") / system / "summary.json").read_text())


def test_run_completes_with_cases(world):
    paths, report = world
    for system in ("aifs-ens", "ifs-ens"):
        s = summary(paths, system)
        assert s["n_cases"] > 100
        assert set(s["couplings"]) == set(config.COUPLINGS)


def test_planted_weak_dependence_is_detected(world):
    paths, _ = world
    d = summary(paths, "aifs-ens")["differences_vs_native"]["crps"]["historical"]
    assert d["hi"] < 0, d


def test_no_spurious_gain_when_dependence_is_right(world):
    paths, _ = world
    d = summary(paths, "ifs-ens")["differences_vs_native"]["crps"]["historical"]
    assert d["lo"] < 0.002 and abs(d["estimate"]) < abs(
        summary(paths, "aifs-ens")["differences_vs_native"]["crps"]["historical"]["estimate"]
    )


def test_cross_model_transfers_better_dependence(world):
    paths, _ = world
    d = summary(paths, "aifs-ens")["differences_vs_native"]["crps"]["cross"]
    assert d["estimate"] < 0


def test_realised_correlation_ordering(world):
    paths, _ = world
    r = summary(paths, "aifs-ens")["realised_spearman"]
    assert r["independent"] < r["native"] < r["historical"]
    assert abs(r["independent"]) < 0.1


def test_single_farm_aggregation_is_invariant(world):
    paths, _ = world
    agg = summary(paths, "aifs-ens")["aggregation"]
    for c in ("historical", "independent", "cross"):
        assert np.isclose(agg["1"][c]["estimate"], 0.0, atol=1e-12)


def test_template_pool_feasible(world):
    paths, _ = world
    pool = summary(paths, "aifs-ens")["template_pool"]
    assert pool["fraction_infeasible"] == 0.0
    assert pool["min_at_21d"] >= config.N_MEMBERS


def test_figures_and_page_build(world):
    from gbwind import figures, page
    from gbwind.portfolio import load_selected

    paths, _ = world
    files = figures.make_all(paths.period_results("dev"), synthetic=True)
    assert all(f.exists() for f in files)
    out = page.build(paths.period_results("dev"), load_selected(paths), synthetic=True)
    html = out.read_text()
    assert "Synthetic demo" in html and '"synthetic":true' in html
