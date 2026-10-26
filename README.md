# gb-wind-physics

## Can weather forecasts correctly predict when wind farms will produce little electricity at the same time?

This project tests AI and physics based weather ensembles against metered
generation from a portfolio of GB wind farms. It keeps each farm's forecast
unchanged and alters only how scenarios are paired across farms, measuring the
effect on uncertainty in their combined output.

**Status.** The evaluation machinery is built. One WeatherNext Mini gradient step
has been measured on one 80 GB A100, on the official weather loss: two ensemble
members fit, four do not, and a 24 h or 42 h rollout fits only with
rematerialisation ([docs/m1_feasibility.md](docs/m1_feasibility.md)). That step
does not yet go through the power calculation. The live archive and the CDK
stack are in the repo (`gbwind/live.py`, `infra/`). Nothing is deployed: this
account has no AWS credentials, and no held-out forecast result exists yet.

## Why it matters

A geographically spread portfolio is valuable because its farms need not all
produce badly at once. A forecast can describe the uncertainty at each farm
correctly and still get the uncertainty of their combined output wrong, because
what matters for the portfolio is whether low output happens together.

Take two farms, each with a 50 per cent chance of producing 0 MW and a 50 per cent
chance of producing 100 MW.

- If they always move together, the total is either 0 or 200 MW.
- If they always offset, the total is always 100 MW.

Each farm has exactly the same individual forecast in both cases, and the expected
total is the same. The chance of the portfolio producing nothing is completely
different. Per site accuracy cannot distinguish these, and it is the distinction
that matters for anyone holding the portfolio.

Ensembles trained on marginal scoring rules have their per location uncertainty
constrained by the loss and their cross location dependence left emergent. Whether
that emergent dependence is right for a real portfolio is an open, application
specific question, and the evidence is mixed. The WeatherNext 3 authors report
that their architecture produces largely realistic joint structure despite the
marginal objective, while a preprint on coastal sea level finds a model with
positive skill at every individual station whose joint spatial structure is worse
than climatological draws.

## The experiment

One forecast system at a time. One portfolio. The 00 UTC run at lead times 24 to
47 hours, the window a day ahead position is taken on. Every farm keeps exactly
the same 51 forecast values. Only the pairing across farms changes.

| Coupling | Dependence source |
| --- | --- |
| Native | The ensemble as shipped |
| Historical | Schaake shuffle against simultaneously observed metered output |
| Independent | Each farm shuffled separately, as an independence reference |

Per farm CRPS is then identical across couplings, which a unit test asserts and
which voids the run if it fails. What the comparison measures is the effect of
changing dependence, conditional on fixed farm level forecasts.

Two systems. [IFS ENS](https://dynamical.org/catalog/models/ecmwf-ifs-ens/) as the
physics reference and [AIFS ENS](https://dynamical.org/catalog/ecmwf-aifs-ens-forecast/)
as the ML arm. Primary outcome is portfolio CRPS, with interval coverage and the
[variogram score](https://journals.ametsoc.org/abstract/journals/mwre/143/4/mwr-d-14-00269.1.xml)
as complementary dependence sensitive diagnostics.

## Data

All free. AIFS ENS and IFS ENS as Icechunk Zarr archives on the AWS Open Data
Registry via dynamical.org, 51 members at 0.25 degrees, opened directly with
Xarray, AIFS ENS covering 2 July 2025 onward. Elexon B1610 for per farm metered
output, with physical notifications and accepted instructions for curtailment
flagging. DESNZ Renewable Energy Planning Database for turbine metadata.

Nothing is trained on the forecast archive. The dependence template comes from
years of metered output, the marginals are the ensemble's own, and the archive is
used for evaluation only.

Checked on first open: 100 m wind is present in both archives, so every arm uses
100 m wind. The lead steps shared by both systems inside the window are 24, 30,
36 and 42 hours. Both are recorded as an execution note in DESIGN.md.

## Progress

- [x] Elexon ingestion module
- [x] Curtailment flagging from notifications and instructions
- [x] Portfolio selection and turbine metadata join
- [x] Ensemble archive loader, members at farm locations
- [x] Power conversion and the three couplings, plus cross model
- [x] Marginal invariance test passing
- [ ] Full metered history downloaded and template feasibility reported
- [ ] Development period result
- [ ] Sealed period, opened once

The machinery is checked end to end on a synthetic world with a planted answer
(`python -m gbwind demo`). There, an ensemble whose cross farm dependence is too
weak is caught by the historical coupling, and a correctly coupled one shows no
spurious gain. That is a check on the method, not a result about real forecasts.

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
| `gbwind/config.py` | Every fixed design choice, in one place |
| `gbwind/elexon.py`, `curtailment.py`, `portfolio.py` | Metered output, flags, portfolio |
| `gbwind/archive.py` | dynamical.org archives, members at farm locations |
| `gbwind/power.py` | Hub height extrapolation and the idealised power curve |
| `gbwind/couplings.py` | Native, historical, independent, cross model |
| `gbwind/template.py` | Pool admissibility, publication lag, widening, draws |
| `gbwind/scores.py`, `bootstrap.py` | CRPS, energy and variogram scores, block bootstrap |
| `gbwind/evaluate.py`, `pipeline.py` | Per case scoring, invariance check, summaries |
| `gbwind/seal.py` | The sealed test period, opened once |
| `gbwind/figures.py`, `page.py` | The three figures and the explorable results page |
| `config/portfolio.yaml` | Candidate farms and BM units |
| `data/forecasts/` | Saved member forecasts at farm locations |
| `results/<period>/<system>/` | Scores, saved couplings and template timestamps, summaries |

## Design and prior art

The full specification, the preregistered hypothesis, the template construction
rules, what is deliberately not claimed, and an amendment log recording where the
design changed and why, are in [DESIGN.md](./DESIGN.md).

Aggregate wind power evaluation of ML ensembles is not new.
[GenCast](https://arxiv.org/abs/2312.15796) converts wind to power through a
nonlinear curve, pools across farm groupings at 120, 240 and 480 km, and evaluates
aggregate CRPS in a section on joint distribution skill, using idealised power
from analysis wind. Fine-tuning a weather foundation model on energy assets is
not new either: Silurian with Hydro-Québec does it per site for hub-height wind
and icing, and Aardvark fine-tunes on station wind or temperature with a
deterministic loss. Restoring cross location dependence with copulas in
[day ahead electricity scenario generation](https://arxiv.org/abs/2409.07637) is
established practice, and
[probabilistic wind power forecasting on offshore farm data](https://arxiv.org/abs/2602.13010)
is an active area. To my knowledge the open question is narrower: whether
training on a portfolio power CRPS changes how a probabilistic weather model
couples farms, separated from a gain that is only better marginals. Aurora 1.5
is a probabilistic ensemble as of July 2026 and is related work, not the model
this live system measures.

The shuffle based approach is borrowed from a preprint on
[coastal sea level](https://arxiv.org/abs/2608.08954). The
[functional generative network](https://arxiv.org/abs/2506.10772) paper reports
largely realistic joint structure despite marginal training, which is the strongest
evidence against the hypothesis here and is treated as such in DESIGN.md. Further
context in [WeatherNext 3](https://arxiv.org/abs/2609.03582),
[AIFS-CRPS](https://arxiv.org/abs/2412.15832) and
[ensemble copula coupling](https://arxiv.org/abs/1302.7149).

The first experiment runs on IFS ENS and AIFS ENS, so its findings concern those
systems. WeatherNext 3 supplies motivation and becomes a third arm only as an
extension.

## Licence

MIT. Forecast data CC BY 4.0 and ECMWF Terms of Use, via dynamical.org. Elexon data
under the Elexon BMRS licence. REPD under the Open Government Licence.
