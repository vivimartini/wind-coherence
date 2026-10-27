# wind-coherence

Does an operational ensemble get the joint behaviour of a GB wind portfolio right, or only the forecast at each farm?

Each farm keeps its 51 forecast values. Only the pairing across farms changes. Portfolio CRPS moves when that pairing was wrong. Two farms that each do 0 or 100 MW half the time look the same farm by farm, whether they fail together (the total is 0 or 200) or offset (the total is always 100).

## Status, 27 Oct 2026

Seventeen farms, metered from September 2020, all still selected. Curves are in `config/power_curves.json`: ERA5 100 m wind, bilinear from the four surrounding 0.25° points, fit on uncurtailed hours before 2 July 2025. On the development dates the median absolute error is 10.2% of capacity. Hornsea 1 is 24%. Most of that is 3–21 June 2026, when the farm metered about zero and the curtailment flag missed it. It stays in.

AIFS ENS v2, one member per GPU. A 42 h step fits on one 80 GB A100 (55.5 GB, 26 s/step). The two-GPU step has not been timed. No development score and no sealed score. The live stack is in `infra/` and has not been deployed from this machine.

Hypothesis, rules, and the dated change log: [DESIGN.md](DESIGN.md). WeatherNext Mini memory: [docs/m1_feasibility.md](docs/m1_feasibility.md).

## Experiment

00 UTC run, leads 24 to 47 hours. One system, one portfolio.

| Coupling | Pairing |
| --- | --- |
| Native | Ensemble as issued |
| Historical | Schaake shuffle against metered output at the same time of day |
| Independent | Each farm shuffled on its own |

Per-farm CRPS has to match across couplings. `tests/test_couplings.py` checks it, and a mismatch voids the run.

[IFS ENS](https://dynamical.org/catalog/models/ecmwf-ifs-ens/) is the reference. [AIFS ENS](https://dynamical.org/catalog/ecmwf-aifs-ens-forecast/) is the model under test. The score is portfolio CRPS, with interval coverage and the [variogram score](https://journals.ametsoc.org/abstract/journals/mwre/143/4/mwr-d-14-00269.1.xml) alongside it.

## Data

IFS ENS and AIFS ENS from the dynamical.org Icechunk archives, 51 members at 0.25°. AIFS ENS starts 2 July 2025. Metered output is Elexon B1610. Curtailment uses physical notifications and accepted instructions. Turbine metadata is the DESNZ Renewable Energy Planning Database.

The forecast archive is for scoring. The dependence template is built from metered history. Training initial states come from ECMWF open data. ERA5 is only for the power curves. Where 100 m wind exists it goes straight into the curve. Shared leads in the 24–47 h window are 24, 30, 36 and 42 h.

## Progress

- [x] Elexon ingestion module
- [x] Curtailment flagging from notifications and instructions
- [x] Portfolio selection and turbine metadata join
- [x] Ensemble archive loader, members at farm locations
- [x] Power conversion and the three couplings, plus cross model
- [x] Marginal invariance test passing
- [x] Metered history, September 2020 onward, and per-farm power curves
- [ ] Template feasibility on the development dates
- [ ] Development period result
- [ ] Sealed period, opened once

`python -m gbwind demo` runs the same path on a synthetic world with a planted answer. Weak cross-farm dependence gets caught. A correctly coupled ensemble shows no gain. That checks the code. It is not a result about real forecasts.

## Run

```bash
pip install -r requirements.txt && pip install -e .
python -m gbwind.elexon check      # one probe per endpoint
python -m gbwind.elexon units      # list wind BM units
```

The data layer, in order:

```bash
python -m gbwind.elexon fetch                    # B1610, PN, BOALF for the candidate portfolio
python -m gbwind portfolio build                 # REPD join, BM unit validation, hub heights
python -m gbwind curtailment                     # farm level metering and curtailment flags
python -m gbwind portfolio select                # metering rule, freezes the portfolio
python -m gbwind archive --system aifs-ens --start 2025-07-02
python -m gbwind archive --system ifs-ens  --start 2025-07-02
python -m gbwind verify  --system aifs-ens       # all members present at every farm
python -m gbwind feasibility --period dev        # template pool sizes, before any result
```

Then one command per period:

```bash
python -m gbwind run --period dev                # evaluate, figures, results page
python -m gbwind run --period v1                 # AIFS ENS v1 robustness, never merged
python -m gbwind run --period test --open-sealed # once; a second attempt refuses
```

Or everything from nothing, including downloads: `python -m gbwind reproduce --period dev`,
or `docker build -t gbwind . && docker run -v $PWD/data:/app/data -v $PWD/results:/app/results gbwind`.

Offline, with no downloads: `python -m gbwind demo` builds the synthetic world and
writes results, figures and the results page under `results/synthetic/dev/`.

Tests: `pytest`. The leakage tests (`tests/test_template_leakage.py`) and the
marginal invariance tests (`tests/test_couplings.py`) run in CI on every push.

## Layout

| Path | What |
| --- | --- |
| `gbwind/config.py` | Constants. A change here is an amendment |
| `gbwind/elexon.py`, `curtailment.py`, `portfolio.py` | Metered output, flags, portfolio |
| `gbwind/archive.py` | dynamical.org archives, members at farm locations |
| `gbwind/power.py` | 100 m wind to power. Fitted curves in `config/power_curves.json` |
| `gbwind/couplings.py` | Native, historical, independent, cross model |
| `gbwind/template.py` | Pool admissibility, publication lag, widening, draws |
| `gbwind/scores.py`, `bootstrap.py` | CRPS, energy and variogram scores, block bootstrap |
| `gbwind/evaluate.py`, `pipeline.py` | Per case scoring, invariance check, summaries |
| `gbwind/seal.py` | The sealed test period, opened once |
| `gbwind/figures.py`, `page.py` | Three figures and one HTML page |
| `config/portfolio.yaml` | Candidate farms and BM units |
| `data/forecasts/` | Saved member forecasts at farm locations |
| `results/<period>/<system>/` | Scores, saved couplings and template timestamps, summaries |

## Prior work

The hypothesis, the template rules, and a dated log of what changed are in [DESIGN.md](./DESIGN.md).

[GenCast](https://arxiv.org/abs/2312.15796) already scores aggregate wind power, from analysis wind through an idealised curve, so it never sees metering or curtailment. Silurian (with Hydro-Québec) and Aardvark fine-tune weather models per site. This run asks whether a portfolio loss changes the coupling, separate from a gain that is only better marginals. Aurora 1.5 is a probabilistic ensemble. The system measured here is AIFS ENS.

The shuffle comes from a preprint on [coastal sea level](https://arxiv.org/abs/2608.08954). The [functional generative network](https://arxiv.org/abs/2506.10772) paper finds largely realistic joint structure from marginal training. That is the result most in tension with the hypothesis, and DESIGN.md treats it that way. Also [WeatherNext 3](https://arxiv.org/abs/2609.03582), [AIFS-CRPS](https://arxiv.org/abs/2412.15832) and [ensemble copula coupling](https://arxiv.org/abs/1302.7149). WeatherNext 3 is a possible later arm.

## Licence

MIT. Forecast data CC BY 4.0 and ECMWF Terms of Use, via dynamical.org. Elexon data
under the Elexon BMRS licence. REPD under the Open Government Licence.
