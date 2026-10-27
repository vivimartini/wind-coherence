# When do wind farms underproduce together?

**Does the dependence structure of an operational ensemble, as distinct from its marginal forecasts, limit portfolio-level wind power prediction in Great Britain?**

A spread-out portfolio is worth something because the farms do not all produce badly at once. The question is whether the forecast has that right. Underproduce together means low generation at the same time. It does not mean turbines breaking.

Question locked 25 September 2026. Later notes change how it is run. They do not change the question.

Tested by holding every farm's marginal forecast fixed and changing only how
values are paired across farms, so that any change in portfolio skill is
attributable to dependence alone.

Design fixed 25 September 2026, before any result is produced and before the
test period is opened. Amendments logged at the foot with dates.

---

## 1. The question in one paragraph

A forecast can describe the uncertainty at each wind farm correctly and still
get the uncertainty of their combined output wrong, because what matters for a
portfolio is whether low output events happen together. Ensembles trained on
marginal scoring rules have their marginals constrained by the loss and their
dependence left emergent. Whether that emergent dependence is good enough for
portfolio decisions is a separate question from marginal skill, and it is not
answered by marginal scores.

### What is being calculated

The predictive distribution of the **portfolio total**, meaning the sum over a
fixed set of farms of a nonlinear function of each farm's local wind, and its
expected CRPS. CRPS of a sum depends on how the summands move together. The average of the per-farm CRPS does not. The sum is the score for that reason.

The experiment measures **the effect of changing dependence, conditional on fixed
farm level forecasts.** It does not decompose portfolio CRPS into a marginal part
and a dependence part. An earlier draft claimed that, and it was wrong. Fixing
the marginals and permuting the coupling produces a score difference attributable
to the coupling change, which is not the same as a unique additive decomposition
of the score into marginal and dependence contributions, and no such decomposition
is claimed.

If a coupling improves portfolio CRPS, that
does not establish that it recovers the true dependence. A different coupling can
compensate for imperfect marginals, so the improvement may reflect error
cancellation rather than a better dependence estimate. The result is therefore
reported as which coupling serves the portfolio better under these marginals, not
as a measurement of true dependence.

---

## 2. Where this sits against existing work, corrected

**The GenCast paper already does aggregate wind power evaluation.** It converts
wind to power through a standard idealised power curve scaled by facility
capacity, sums across groupings of wind farms defined by pooling regions at 120,
240 and 480 km, and evaluates aggregate CRPS out to 10 days. It appears in a
section titled *Skill in predicting the joint distribution* and is explicitly
presented as a test of spatial coherence.

An earlier draft of this design claimed nobody had tested joint calibration
through a downstream aggregate quantity. That was wrong, and the claim is
withdrawn.

Three differences are left, and the project stands on these.

1. **GenCast's targets are idealised.** Power is derived from analysis 10 m wind
   times a standard curve times capacity. It never meets metered generation, so
   it never meets curtailment, turbine availability, real hub heights or real
   settlement periods.
2. **It measures aggregate skill, not separable dependence.** Showing that one
   ensemble scores better on pooled power than another does not isolate
   dependence, because the two also differ in their marginals. The decomposition
   in section 4 is what isolates it, and no paper found so far runs it for wind
   power.
3. **Current systems.** GenCast is the December 2024 generation. AIFS ENS v2 and
   WeatherNext 3 both postdate it.

Two further pieces of adjacent work, for honest positioning. *Do AI Forecast
Ensembles Sample the Correct Conditional Distribution?*, a preprint, finds a diffusion model
with positive marginal skill at every one of eight tide gauges whose joint
spatial structure is worse than climatological draws, using a shuffle based
decomposition, and reports that the failure is invisible to the energy score but
detected by the variogram score. That is the method this project borrows, applied
to sea level rather than power. *Good Forecast, Missing Physics* finds noise like
mesoscale structure in GenCast ensemble spread below roughly 400 km, as internal
physics diagnostics with no downstream application.

Fine-tuning a weather foundation model on energy assets is not new, and this
design does not claim that it is. Silurian AI with Hydro-Québec fine-tunes a
1.5 billion parameter model on utility data, including wind-farm met masts, for
hub-height wind and icing, with per-site regression losses: no ensemble, no
power, no portfolio. Aardvark fine-tunes the whole model on station 10 m wind or
temperature as the only target, scored by deterministic RMSE, with no
aggregation. LSEG has fed AIFS ENS into renewable production models, and a 2026
group has fine-tuned AIFS with an added decoder head. To my knowledge neither
trains on power or on a portfolio. The statistical wind-power literature already
shows that a calibrated aggregate forecast needs a model of spatial dependence,
which separate site forecasts do not reveal. That literature is statistical, not
a weather model.

To my knowledge, what is not already published is the narrower claim this
project actually makes: training a probabilistic weather model with a CRPS
defined on physically converted aggregate power, then testing whether that
changes the dependence between sites rather than the marginals. The search
behind that sentence was not exhaustive. The contribution is that portfolio
objective and that attribution, and nothing broader.

Aurora was earlier set aside as a deterministic model. That is out of date.
Microsoft released Aurora 1.5 on 9 July 2026, with probabilistic ensemble
forecasts, hourly output and 22 further variables, code on GitHub and
checkpoints on Hugging Face. How those members are formed, and how the model is
fine-tuned, has not been checked here. Aurora 1.5 belongs in the related work.
It is not the training target. The live system scores AIFS ENS, so fine-tuning
AIFS ENS is what closes that loop.

---

## 3. The hypothesis, stated as a hypothesis

**Preregistered directional hypothesis.** Replacing an ML ensemble's native
coupling with a historical dependence template will improve portfolio CRPS,
while leaving per farm CRPS unchanged by construction.

### The best evidence against this hypothesis

Stated here rather than in a discussion section, because a preregistered
hypothesis that hides the case against it is not preregistered in any useful
sense.

The WeatherNext 3 paper reports that despite being trained to optimise marginal
skill, the functional generative network architecture produces largely realistic
joint structures, citing spatial metrics showing strong performance across spatial
scales, and it demonstrates this through spatially pooled precipitation forecasts
that show no systematic degradation at larger pooling sizes. The FGN authors
discuss why architectural constraints may encourage useful joint structure even
though a marginal objective does not uniquely determine it.

That is a real and interesting claim, namely a capability emerging that the loss
does not identify. It makes a null result a live possibility rather than a
disappointing edge case, and it is the reason section 7 treats a null result as a
headline with competing explanations rather than as a failure.

It also bounds what this project can conclude. The first experiment runs on IFS
ENS and AIFS ENS, so its findings concern those systems. WeatherNext 3 supplies
the motivation and is not a target until X1.

### Adjacent applied literature

Restoring cross location and cross time dependence in day ahead electricity
scenarios, using copulas on top of separately produced marginals, is an
established practice evaluated on real power system data. Probabilistic wind power
forecasting on multi year offshore farm data with weather forecasts is likewise an
active area, focused on producing better per farm generation distributions.

This project sits inside that area rather than beside it. The distinction is that
the emphasis here is on how the distributions combine across farms, with the
marginals deliberately held fixed so that the coupling is the only thing varying.
No claim is made that copulas for power scenarios, or aggregate wind evaluation,
are new.

### What is not claimed about the hypothesis, and why

- Not claimed that marginal CRPS training entails weak correlations. A marginal
  loss leaves dependence unconstrained, which is not the same as biasing it in a
  known direction. The direction above is a guess informed by the mesoscale noise
  finding, and the experiment is designed to detect the opposite result equally
  well.
- Not claimed that the effect must grow with the number of farms. Farm spacing,
  capacity concentration and weather regime all bear on it, so aggregation level
  is reported as a curve rather than assumed monotone.
- Not claimed that the physics ensemble has correct dependence. Dynamical
  consistency within a member is not calibration of the dependence across
  locations. IFS ENS is a reference arm, not a ground truth.

---

## 4. The decisive experiment

Within a single forecast system, for a fixed portfolio and a fixed lead time,
every farm keeps exactly the same set of forecast values across members. Only
the pairing across farms changes.

| Coupling | How members are paired across farms | Role |
| --- | --- | --- |
| **Native** | As the ensemble delivers them, member *i* at every farm | The system as shipped |
| **Historical** | Schaake shuffle. Farm level values are reordered to match the rank pattern of a historical day drawn from metered output, so dependence comes from observed joint behaviour | The treatment |
| **Independent** | Each farm's values shuffled independently | An independence reference, not exact independence. With 51 members, residual sample correlation remains, so the realised correlation is reported rather than assumed to be zero. It is a scale marker, not a predicted loser. A portfolio whose farms genuinely decorrelate at this lead time could score well under it, which would itself be informative |
| **Cross model** | Ranks taken from the other forecast system, applied to these marginals | Extension. Isolates whose dependence is better |

Because marginals are identical across all four, per farm CRPS is identical by
construction, and the whole difference in portfolio CRPS is dependence. That is
the result the project exists to produce.

**Why plain ensemble copula coupling is not the treatment.** ECC re-imposes the
raw ensemble's own rank dependence on post processed marginals. Applying it with
the ensemble whose dependence is under suspicion would preserve exactly the thing
being tested. Hence the historical template and the cross model variant, which
both source dependence from outside the ensemble in question.

### How the historical template is actually constructed

"A historical day" was ambiguous in the previous draft. Specified now, because
these choices determine whether the template is valid.

The shuffle needs one historical timestamp per ensemble member, so 51 of them,
each supplying a vector of simultaneously observed outputs across every farm in
the portfolio. Farm by farm, those 51 historical values are ranked, and the 51
forecast values at that farm are reordered to carry the same rank pattern. The
historical joint rank structure is thereby imposed on the forecast marginals.

**Pool construction.** A timestamp is admissible only if all of the following
hold.

1. **Same time of day** as the target settlement period, since the diurnal cycle
   affects both stability and the shape of spatial correlation.
2. **Seasonal window.** Within plus or minus 21 days of the target date's day of
   year, across all available prior years. Wide enough to fill the pool, narrow
   enough that the correlation regime is comparable.
3. **Every farm in the portfolio uncurtailed and fully metered** in that
   settlement period. A timestamp with a single missing or curtailed farm is
   dropped rather than patched, because a patched value would carry no joint
   information.
4. **Available at issue time**, which is stricter than merely being in the past.
   B1610 publishes roughly five working days after delivery, so the pool must end
   at the issue time minus that publication lag. A template built from metered
   data that had not yet been published is a leak, and the leakage test asserts
   against it.

**Sampling.** The 51 timestamps are drawn without replacement from the admissible
pool with a fixed seed, and the draw is repeated across bootstrap replicates so
that template sampling variability appears in the reported intervals rather than
being hidden by a single lucky draw.

**Ties.** Metered wind output ties heavily, at zero below cut in and at rated
capacity in strong winds, and ties break the rank construction. Ties are broken at
random under the fixed seed, and the tie fraction is reported per farm, because a
high tie fraction means the template carries little rank information for that farm
and the result must be read accordingly.

**Feasibility check, before any result.** Confirm the admissible pool reaches at
least 51 timestamps for every target period, and report the pool size
distribution. Requirements 1 through 4 together are restrictive, and a pool that
falls short at some targets would force either a wider seasonal window or a
smaller portfolio. Which of those is chosen is recorded here before results are
seen, and the choice is a wider seasonal window, because shrinking the portfolio
would change the quantity under study.

---

## 5. Scope

**In.** Two forecast systems, IFS ENS as the physics reference and AIFS ENS v2
as the ML arm. One portfolio of GB offshore and onshore farms with reliable
metering and complete metadata. One lead time window fixed now, the 00 UTC run
at lead times 24 to 47 hours, which is the window a day ahead position is taken
on. One primary outcome, portfolio CRPS, supported by interval coverage and the
variogram score.

**Named honestly.** The target is *the selected portfolio's output*, not national
wind. National aggregation is claimed only if the fleet is actually covered.

**Out, for the core.** WeatherNext 3 as a third arm, a learned copula beyond the
Schaake shuffle already specified, temporal dependence, national aggregation,
cold start, and Aurora 1.5 as a training target. Aurora 1.5 is related work,
not a deterministic leftover, and it stays out until its ensemble mechanism has
been read. Settlement of a fixed nomination at the Elexon System Price is in
scope, as a second way to report the same arms. A trading strategy is not.

**The ramp experiment is dropped.** An earlier draft motivated it from the
temporal discontinuities in WeatherNext 3. That was a misreading. The paper's
section 4.6 reports discontinuities in the station output head and to some degree
the precipitation heads, and states explicitly that this does not apply to the
hourly analysis output, which has no noticeable discontinuities. The station head
predicts 2 m temperature and dewpoint. The 100 m wind sits in the hourly analysis
output. So the cited evidence does not support a wind ramp failure and the
experiment has no premise.

---

## 6. Data, and why the archive gate is closed

| Source | What | Coverage |
| --- | --- | --- |
| AIFS ENS via the dynamical.org Icechunk Zarr archive on the AWS Open Data Registry | 51 members, 0.25 degree, 10 m and 100 m wind components, 2 m temperature, 6 hourly initialisations | Forecasts initialised 2025-07-02 to present. Free, CC BY 4.0 |
| IFS ENS via the same archive route | 51 members, 0.25 degree, medium range | Free, same access pattern |
| Elexon B1610 | Metered volume per BM unit, half hourly | Years of history |
| Elexon PN and BOALF | Declared output plans and accepted instructions | Curtailment flagging |
| DESNZ Renewable Energy Planning Database | Location, capacity, turbine model, hub height | Static |
| ECMWF open data, IFS step 0, AWS bucket `ecmwf-forecasts` | Initial states for fine-tuning and rollouts: 13 pressure levels, surface and static fields, 00/12 UTC main runs and 06/18 UTC short cutoff runs | 0.4 degree from 18 Jan 2023, 0.25 degree from 1 Feb 2024. Free, CC BY 4.0 |
| ERA5, Google ARCO-ERA5 Zarr | Hourly 100 m and 10 m wind at farm locations | Power curve fitting only, never an initial state |

ECMWF's live open data service retains only recent runs. The AWS bucket keeps
older runs as GRIB, which is enough for initial states but would be a heavy route
to full ensemble forecasts. The dynamical.org archives serve those, opened
directly with Xarray rather than assembled from GRIB downloads.

**Verify on first open.** Whether 100 m wind is present in the IFS ENS archive as
well as AIFS ENS. If it is not, all arms use 10 m wind, which is what GenCast
did, and the comparison stays internally consistent.

### The archive length problem, and why it does not bite

AIFS ENS v2 arrived on 12 May 2026, alongside ECMWF cycle 50r1, which is a regime
change inside the archive. Restricting the primary analysis to v2 leaves roughly
four months, which would be far too little to train a model.

It is enough here, because **nothing is trained on the forecast archive.** The
historical dependence template is built from Elexon metered output, which goes
back years. The marginals are the ensemble's own. The forecast archive is used
only for evaluation. So the short v2 window is a sample size question for the
bootstrap, not a training data question.

Split accordingly. Development on 12 May to 30 June 2026. Sealed test from 1 July
2026, opened once. The v1 period from July 2025 to April 2026 is reported
separately as a robustness check on the earlier model version, never merged.

### Curtailment and shared operational effects

This is the trap that could invalidate the whole result. If two farms are
constrained together by the same network limit, their output errors are
correlated for reasons that have nothing to do with weather dependence, and a
historical template built from metered output would import that correlation and
then be credited for it.

So the historical dependence template is built **only from periods flagged
uncurtailed at every farm in the portfolio**, and the count of admissible days is
reported. Curtailed periods are excluded from evaluation as well as from the
template. A sensitivity check reports what happens when they are included, and
any difference is discussed as operational rather than meteorological dependence.

---

## 7. Evaluation protocol

- **Primary outcome.** Portfolio CRPS at the fixed lead window, normalised by
  portfolio rated capacity. Pairs are named by the checkpoint's own graph, read
  on 25 October 2026 from `aifs-ens-crps-2.0.ckpt` before any score. Hidden
  nodes are `ReducedGaussianGridNodes` with grid `o96`. Their coordinates are
  the 40,320 points in anemoi's `grid-o96.npz`, which is the file that builder
  loads, and the node id is the index in that file. The checkpoint does not
  store a hidden-to-hidden edge list. Its edges are `CutOffEdges` from the
  N320 data nodes stored in the checkpoint (542,080 points) onto those O96
  nodes, cutoff 250 km, at most 64 neighbours, radius 250/6371 on the unit
  sphere. Two O96 nodes are one hop apart when a data node connects to both.
  A hop count is the shortest path on that graph.
  Dudgeon and Race Bank are both node 3744, hop 0. East Anglia One, Greater
  Gabbard and London Array are all node 3921, hop 0. Whitelee and Clyde are
  both node 3402, hop 0. Walney Extension and Gwynt y Môr are both node 3742,
  hop 0. A hop of 0 should be nearly perfect coupling from the interpolation.
  If Dudgeon and Race Bank are not, the power chain has a bug and that pair is
  not evidence about the atmosphere.
  Whitelee (node 3402) and Griffin (node 3238) are one hop apart. Beatrice
  (node 2923) and Hornsea 1 (node 3573) are two hops apart. No pair in this
  roster is three hops: the encoder cutoff is wide enough that two is the
  maximum. The earlier division by 104.36 km is withdrawn. It put Whitelee and
  Clyde, which share a node, on different scales.
- **H1 is two-sided.** Aggregate calibration error can be underdispersion or
  overdispersion. Adamov, Fuhrer, Knutti and Schemm (MeteoSwiss and ETH Zurich,
  2026, arXiv:2609.08412) found per-pixel spread near calibrated while the
  spatial mean was overdispersed, because each member carried one coherent
  field offset. That paper is about weight noise added to deterministic
  models, not about AIFS ENS. It is the reason both signs count. The earlier
  wording, that AIFS would show a larger underdispersion gap than IFS, is
  withdrawn. Both signs are scored on the full portfolio, on the one-hop pair
  and on the two-hop pair.
- **Detectable effect, before arms C and D.** On development dates, arms A and
  B are used to estimate the spread of paired portfolio-CRPS differences under
  the block bootstrap, and from that the smallest relative effect the test
  window can detect. The number used as a yardstick is the STIPP comparison,
  where changing the training objective from CRPS to the energy score moved a
  station wind energy score from 9.52 to 9.46, about 0.6 percent. That is a
  station energy score, not a portfolio CRPS, so it is an anchor for what an
  objective change has been seen to move, not a prediction for this portfolio.
  If the smallest effect this window can detect is above 1.8 percent of the
  development-set portfolio CRPS, three times that anchor, the window is
  underpowered. Arms C and D do not spend GPU hours until that comparison is
  written down. An underpowered window is extended, or the weight of the
  project shifts onto H1 and the live system.
- **Cheap rival.** Beside arm B, the frozen model's power members are reordered
  with the Schaake shuffle already specified in section 4. That baseline trains
  nothing. Fine-tuning has to beat it as well as the per-farm correction. A
  coupling fitted to the evaluation scores is not used.
- **Pounds, and a score that sees the coupling.** The ensemble mean of member
  power does not. The mean of a sum is the sum of the means, so reordering
  members across farms leaves that nomination unchanged. Beside it, two
  quantities that do change when the coupling changes. The pinball loss of the
  portfolio total at the 10 percent and 90 percent quantiles of the member
  totals, members summed before the quantile is taken. And a fixed reserve
  nomination: the 10 percent quantile of those member totals, the level fixed
  in advance and not tuned. Both nominations, the mean and the reserve, are
  converted to MWh, use only what is known at issue time, and are cashed out
  the same way: metered MWh minus nominated MWh at the Elexon System Price.
  The window is reported in pounds per MW of portfolio capacity for each
  nomination, next to CRPS and the two pinball losses. The price is not an
  input to training.
- **Supporting.** Interval coverage at nominal levels, the variogram score as a
  complementary diagnostic sensitive to dependence errors, and a rank histogram of
  the portfolio total. The variogram score is not universally more sensitive than
  the energy score. It was the discriminating one in the coastal sea level study,
  which is the reason for including it alongside rather than instead.
- **Sanity condition.** Per farm CRPS must be identical across couplings to
  numerical precision. If it is not, the shuffle has leaked and the run is void.
- **Held out dates.** Chronological. Sealed test opened once.
- **Uncertainty.** Block bootstrap over days, with block length respecting wind
  autocorrelation, which runs to days. No difference reported without an
  interval, and with roughly four months of v2 the intervals will be wide. That
  is stated rather than hidden.
- **Failure reported.** If the historical coupling does not beat native, that is
  the result and it is reported as the headline. It is **not** proof that native
  dependence is adequate. A null result is consistent with several explanations,
  and the write up must distinguish them rather than pick the flattering one.
  Either the native dependence is genuinely good, or the historical template is
  itself a poor estimate of the true conditional dependence, since it is drawn
  from unconditional historical behaviour rather than conditioned on the weather
  situation, or portfolio CRPS at this lead time and this portfolio size is not
  sensitive enough to resolve the difference, which the interval width will
  indicate. The variogram score and the aggregation curve are what separate these.

---

## 8. Deliverables

1. Public repository, MIT licensed, one command reproducible, Docker, CI with
   leakage tests and the marginal invariance assertion above as a unit test.
2. Saved forecasts and couplings, so the numbers are checkable.
3. A write up of roughly 1500 words. Question, why marginals are not enough,
   design, result, what failed.
4. Three figures. Portfolio CRPS by coupling with intervals, the same by
   aggregation level from single farm to full portfolio, and a portfolio rank
   histogram under native against historical coupling.
5. **One explorable results page.** This is a core deliverable, not decoration. It
   shows the farms on a map, the portfolio forecast distribution, the observed
   output, and what visibly changes when the pairing changes. One carefully chosen
   case illustrates the mechanism, with the full held out evaluation behind it.
   The purpose is that someone can open it and understand the finding in a minute,
   which is what turns the work into something discussable rather than something
   claimed.

---

## 9. Extensions, in order, none required

| # | Extension |
| --- | --- |
| X1 | WeatherNext 3 as a third arm, once full 64 member Zarr access is granted. The free BigQuery and Earth Engine statistics give precomputed quantiles only, which destroy the cross farm dependence being measured, so members are mandatory |
| X2 | Cross model coupling, IFS ranks onto AIFS marginals and the reverse |
| X3 | One small dependence model conditioned on the weather situation, in place of the unconditional historical template. This is the extension that adds a trained component, and it directly addresses the most likely reason for a null result, namely that dependence varies by regime and an unconditional template averages over it. It is small on purpose. Adding a foundation model to make the project sound larger is not the aim |
| X4 | Temporal dependence and ramp distributions, if a premise for it is found |
| X5 | National aggregation. Settlement at the System Price is in section 7, not an extension |
| X6 | Cold start at unseen farms, the previous design, as a separate study |
| X7 | The next study: what the farm network reveals about which weather evolution is happening. Question: when do metered outputs at one group of farms carry information about the atmosphere that transfers to farms not yet observed? Decisive experiment, on archived ensembles first: withhold an entire geographic group of farms; weight the 51 AIFS ENS members by the likelihood of the observed power sequence at the remaining farms over the early leads, using a probabilistic forward model (weather and operating state to power, curtailment as a mixture component, so rated output is evidence of a wind range rather than none); score the withheld group at later leads. Baselines: the raw ensemble, plain reweighting as the method's floor, the Schaake shuffle, and a statistical model given the same weather and power history. Scored on per-farm CRPS, joint ramps and portfolio shortfall. Learned generation of updated scenarios, including ones absent from the ensemble, is attempted only if reweighting transfers; a claim about the atmosphere, rather than power, needs independent wind observations. A later step tests which farm's next measurement would be most informative, by hiding and revealing measurements. Constraints: use metering only from when it was actually published; the archive is 6 hourly, so no half-hourly ramp claim is made by interpolation. It depends on the core result: transfer through the ensemble works only if the ensemble's cross-farm dependence is right, which is H1. Prior work: SCADA and lidar assimilated into WRF (Wind Energy Science 2025, Belgian offshore farms, gain about an hour ahead), turbine anemometers (Renewable Energy 2017), and Li, Guan and Wu (Energy Conversion and Management 2015), who already estimate atmospheric states from wind power measurements with a particle filter and forecast aggregate generation, so power as an atmospheric observation is prior art. Generative posterior sampling of weather states is active (WIND, arXiv:2602.03924), so a generative model alone is not a contribution. Any contribution would lie in the observation model, the treatment of ambiguity, or demonstrated transfer, and none is claimed until searched properly. Runs after the core comparison: the per-farm power curves and the live forward chain come first |

---

## 10. Schedule

| Dates | Work | State at the end |
| --- | --- | --- |
| 26 to 30 Sep | Elexon ingestion, curtailment flagging, portfolio selection, AIFS ENS and IFS ENS archive opened and members verified at farm locations | Data layer done, repo public, CI green |
| 1 to 5 Oct | Power conversion, the four couplings, marginal invariance test passing | The machinery works |
| 6 to 12 Oct | Development period results, diagnostics, bootstrap | The answer is known internally |
| 13 to 17 Oct | Sealed period opened once, write up, figures | Complete |
| Later | X1 onward, one at a time | Additive |

---

## 11. What will not be claimed

- No claim that aggregate wind power evaluation of ML ensembles is new. GenCast
  did it. Section 2 says so.
- No claim that marginal CRPS training causes weak dependence. It leaves
  dependence unconstrained, which is a different statement.
- No claim that the effect grows with farm count, or that IFS ENS dependence is
  correct.
- No claim to decompose portfolio CRPS into marginal and dependence components,
  and no claim that a coupling which improves the score has recovered the true
  dependence, since it may instead compensate for imperfect marginals.
- No claim that independent shuffling achieves exact independence, or that the
  variogram score is universally more sensitive than the energy score.
- No claim about national wind. The pound figure is the settlement of one
  prespecified nomination at the System Price. It is not a trading result.
- No claim that these models are defective through squared error training.
  WeatherNext 3 trains on CRPS and AIFS ENS uses a related probabilistic
  objective.
- No claim that fine-tuning a weather foundation model for energy assets is new.
  Silurian with Hydro-Québec and Aardvark already do per-site versions. The
  sentence that is claimed, and only to my knowledge, is the portfolio CRPS
  objective together with a test that separates a dependence change from a
  marginal change.
- No claim that H1 must show up on the full portfolio, or that its sign is
  underdispersion. The one-hop pair is Whitelee with Griffin, nodes 3402 and
  3238. The two-hop pair is Beatrice with Hornsea 1, nodes 2923 and 3573.
  Dudgeon with Race Bank, both node 3744, is a sanity check.
- No claim that a D versus C difference is detectable until the development-set
  calculation in section 7 has been done.

---

## 12. Fine-tuning does not inherit fixed marginals

Logged 26 September 2026, before any WeatherNext forecast is scored. This does
not move the locked question in section 1.

The coupling experiment holds every farm's forecast values fixed and changes
only their pairing. That is why a portfolio-score difference can be attributed
to the coupling.

A later experiment that fine-tunes weather-model parameters does not have that
property. Decoder parameters and stochastic norm-conditioning parameters can
change the marginal distribution at each farm as well as the dependence across
farms. Updating the norm-conditioning path is not a dependence-only
intervention. A portfolio gain after fine-tuning therefore cannot, by itself,
be distinguished from a gain that comes from better marginals.

Any fine-tuned comparison must report, on the same initialisations, portfolio
skill, per-farm skill, and a weather-skill panel. The coupling permutation
remains the experiment that isolates dependence. Fine-tuning is a different
experiment.

---

## 13. What the two completed jobs measured

Logged 25 October 2026 from the job logs, before any AIFS memory result.

Job 39193411 asked for two GPUs. The scheduler gave two NVIDIA A100-SXM4-80GB devices, and JAX saw both.

Job 39194625 asked for one GPU and was given one A100-SXM4-80GB. On that device, two members for seven steps (42 hours) with rematerialisation used 67,511 MiB at peak and 12.0 seconds a step in steady state. Four members, same rollout and rematerialisation, did not fit. The run died asking for 154.65 GiB.

Job 39248120, 26 October, AIFS ENS v2 on one A100-SXM4-80GB: one member, one 6 hour step, inference only, initialised from ECMWF open data `enfo` perturbed member 1 at 2026-10-26 06 UTC. Peak 21,883 MiB allocated by PyTorch, 24,381 MiB seen by the GPU. The output has 120 fields on the 542,080 point N320 grid, including 10 m and 100 m wind. Without gradients this is a lower bound and does not settle the model choice. The deciding number is a forward and backward pass with the CRPS loss over at least two members. The checkpoint's training config sets `num_gpus_per_ensemble` to 2.

Training memory, jobs 39256505, 39258338 and 39259276, 26 October, one A100-SXM4-80GB each. One AdamW step: a rollout with gradients, 100 m wind at the 17 farm nodes through a smooth power curve, CRPS of the portfolio total, backward. Activation checkpointing per processor block (16 chunks), bf16 autocast. Inputs are random in normalised space with the real shapes, so these are memory and time, not skill. Seconds per step are the steady-state iteration after warm-up.

| Run | Members on the GPU | 6 h steps | Lead | Peak allocated | Seconds per step |
| --- | --- | --- | --- | --- | --- |
| a | 2 | 1 | 6 h | 55,197 MiB | 7.5 |
| b | 2 | 2 | 12 h | 65,813 MiB | 15.1 |
| c | 2 | 4 | 24 h | out of memory at 80 GB | |
| e | 1 | 1 | 6 h | 29,673 MiB | 3.7 |
| f | 1 | 4 | 24 h | 43,191 MiB | 15.0 |
| g | 1 | 7 | 42 h | 55,505 MiB | 26.3 |

Each extra step costs about 4.3 GB per member, and time is linear in steps. Two members on one device cannot reach 24 hours; one member reaches 42 hours with 24 GB to spare.

**Decision: AIFS ENS, one member per GPU.** An ensemble of M members trains on M GPUs, the layout the checkpoint itself was trained with (`num_gpus_per_ensemble` 2, `ensemble_size_per_device` 1). The CRPS needs every member's portfolio total, so the one cross-device exchange is an all-gather of 17 farm values per member per step, negligible beside the model. That exchange and the two-GPU step time are not yet measured; a two-GPU run of configuration g is the next job. At 26.3 seconds per step per GPU, one two-member optimiser step at 42 hours costs about 53 GPU-seconds, so the 288 GPU hours reserved for arms C and D buy about 19,500 such steps before any data-loading overhead. WeatherNext Mini stays the fallback if the two-GPU run fails.

---

## Amendments

- 25 Sep 2026. Initial design, national physics structured forecasting.
- 25 Sep 2026. Narrowed to offshore cold start.
- 25 Sep 2026, benchmark corrections, before any forecast is scored.
  - *Capacity.* Registered BM unit capacity disagreed with REPD by up to 90 per
    cent at some farms, and a wrongly scaled power forecast is exactly what a
    coupling could appear to fix by compensation. The power curve is now scaled
    by the 99.5th percentile of each farm's admissible metered output before 12
    May 2026, and portfolio CRPS is normalised by the same capacity. Registered
    and REPD capacity are kept alongside for comparison, and the percentile is
    also reported per year so drift is visible.
  - *Curtailment threshold.* The any-acceptance rule was too broad. A BM unit is
    now curtailed in a settlement period only when an accepted instruction takes
    it below 85 per cent of its physical notification, that is a reduction of
    more than 15 per cent. An acceptance with no notification is still flagged.
    The any-acceptance flag is kept for sensitivity.
  - *Negative price exclusion.* Self-curtailment at negative prices is invisible
    to acceptances, so settlement periods whose Elexon market index price is
    negative in any provider with traded volume are excluded at every farm, from
    both the template and evaluation. The market index is a short term price and
    stands in for the day ahead price, which is not freely available.
  - *Kilgallioch.* Its second BM unit, an extension, is metered only from October
    2025. The farm is defined as the original unit alone, so that it is the same
    farm across the template history.
- 25 Sep 2026, execution note, before any result. Narrows execution only.
  - *Verify on first open, resolved.* `wind_u_100m` and `wind_v_100m` are present
    in both `ecmwf-aifs-ens-forecast` and
    `ecmwf-ifs-ens-forecast-15-day-0-25-degree`, so every arm uses 100 m wind,
    with 10 m wind used only to estimate shear for the extrapolation to hub height.
  - *Lead steps.* AIFS ENS is 6 hourly and IFS ENS is 3 hourly in the medium
    range, so the steps shared by both inside the fixed 24 to 47 hour window are
    24, 30, 36 and 42 hours. These are the primary steps for both systems. IFS ENS
    at its additional 3 hourly steps is a sensitivity check only.
  - *Matching an instantaneous forecast to settlement periods.* The observation for
    a forecast valid at time *t* is the mean output over the two settlement periods
    either side of *t*, from *t* minus 30 minutes to *t* plus 30 minutes. The
    template is built on exactly the same centred hourly quantity, so a
    timestamp is admissible only if both of its settlement periods are admissible
    at every farm.
  - *Hub height.* REPD records tip height rather than hub height and leaves it
    blank for many large farms. Hub height is taken from the portfolio file where
    stated, and otherwise approximated as 0.62 times the REPD tip height, or 100 m
    offshore and 80 m onshore where tip height is missing. The approximation is
    fixed now and not tuned against results.
  - *Power curve.* One idealised curve, cut in 3 m/s, rated 12.5 m/s, cut out
    25 m/s, cubic between cut in and rated, scaled by BM unit registered capacity
    (superseded by the capacity correction above).
    Nothing is fitted to the forecast archive, consistent with section 6.
  - *Curtailment flag.* A farm's settlement period is flagged curtailed if any
    accepted bid-offer acceptance for any of its BM units overlaps it
    (superseded by the threshold and negative price rules above).
  - *Scores.* CRPS is the empirical-distribution ensemble CRPS. The energy score and
    variogram score (order 0.5, uniform weights) are computed on capacity factors,
    so that large farms do not dominate the dependence diagnostics.
  - *Bootstrap.* Circular block bootstrap over issue days, block length 5 days,
    2000 replicates, with an independent draw of the historical and independent
    coupling realisation per day inside each replicate. Block length 3 and 10 days
    are reported as sensitivities.
  - *Illustrative case for the results page.* Chosen by a rule fixed now: among
    cases whose native portfolio ensemble spread is above the period median, the
    one with the lowest observed portfolio output. That is an underproduce
    together event the forecast was genuinely uncertain about. The rule uses no
    coupling score, so it cannot select for a flattering result. Other cases are
    selectable.
- 25 Sep 2026, final plus two. Added the strongest evidence against the
  preregistered hypothesis to section 3, namely that the WeatherNext 3 paper
  reports largely realistic joint structure from the FGN architecture despite
  marginal training, and demonstrates it through spatially pooled precipitation.
  Added the adjacent applied literature on copula based scenario generation and
  probabilistic offshore wind forecasting, inside which this project sits. Labelled
  the coastal sea level work as a preprint. Made explicit that the first
  experiment's findings concern IFS ENS and AIFS ENS only.
- 25 Sep 2026, final plus one. Withdrew the claim that the experiment decomposes
  portfolio CRPS into marginal and dependence parts. It isolates a score
  difference under fixed marginals, which is weaker, and an improvement may
  compensate for imperfect marginals rather than recover true dependence. Independent
  coupling restated as an independence reference with residual finite sample
  correlation. Variogram score restated as complementary rather than superior to
  the energy score. Historical template construction fully specified, covering
  pool admissibility, the publication lag that makes "before issue time"
  insufficient, seasonal windowing, tie handling, and a pool size feasibility
  check with the remedy chosen in advance.
- 25 Sep 2026, final. Question locked. Plain language framing adopted as the
  title. Independent coupling reclassified from a predicted loser to a zero
  dependence reference point. Null result interpretation corrected, since a
  historical shuffle failing to help does not establish that native dependence is
  adequate. Explorable results page promoted to a core deliverable. A small
  conditional dependence model added as the training bearing extension.
- 25 Sep 2026. Replaced with the dependence decomposition above. Novelty claim
  corrected against the GenCast aggregate wind power evaluation. Experiment
  changed from a cross system comparison, which confounds marginals with
  dependence, to a within system coupling permutation, which does not. Ramp
  experiment dropped for lack of a premise. Ensemble copula coupling replaced by
  a historical template and a cross model variant, since plain ECC would preserve
  the dependence under test. Archive route identified, closing the retention gate.
- 25 Oct 2026, before any operational ensemble is scored on metered generation.
  Signed prediction H1. If member deviations at nearby farms are less correlated
  than the real forecast errors, a farm can look calibrated while the portfolio
  total is underdispersed. Because AIFS ENS uses spatially uncorrelated
  perturbations, the prediction is that AIFS ENS shows a larger gap between
  farm-level and portfolio-level calibration than IFS ENS. The test is the
  portfolio rank histogram and interval coverage, set against the same
  diagnostics per farm, and the rank-shuffle intervention. The pair choice in
  this bullet was superseded later the same day: separation is measured in O96
  node spacings, and the calibration error is two-sided. The Milestone 1
  weather-loss measurements are not evidence about this prediction. This does
  not move the locked question in section 1.
- 26 Sep 2026, before any WeatherNext result. Section 12. Fine-tuning weather-model
  parameters, including decoder and stochastic norm-conditioning parameters, can
  change farm-level marginals as well as cross-farm dependence. It does not inherit
  the fixed-marginals property of the coupling experiment. A portfolio gain on a
  fine-tuned arm is reported together with per-farm skill and a weather-skill panel
  on the same initialisations, or it is not interpretable as a dependence result.
  This does not move the locked question in section 1.
- 25 Oct 2026, prior-work audit, before any operational ensemble is scored and
  before arms C or D are trained. Three corrections. First, fine-tuning a
  weather model on energy assets is prior art: Silurian with Hydro-Québec
  (per-site hub-height wind and icing, no ensemble and no portfolio) and
  Aardvark (joint fine-tune on station 10 m wind or temperature, deterministic
  RMSE, no aggregation). The contribution is narrowed to a portfolio CRPS on
  physically converted power, plus a test of whether that moves dependence
  rather than marginals, stated as "to my knowledge". Aurora 1.5, released 9
  July 2026, is a probabilistic ensemble and is related work; its member
  mechanism and fine-tuning procedure are unchecked, and it is not the training
  target because the live system measures AIFS ENS. Second, H1 is also scored
  on East Anglia One with Greater Gabbard (30 km) and on Dudgeon with Race Bank
  (2.6 km), because a full-portfolio null can miss a small-scale effect.
  Those two distances were the wrong scale; the next amendment replaces them.
  Third, the spread of paired CRPS differences is estimated from arms A and B
  on development dates before any GPU hours go to C or D. This does not move
  the locked question in section 1.
- 25 Oct 2026, later the same day, before scoring. Pairs are chosen in O96 node
  spacings. The design spacing is 40,075 / (4×96) = 104.36 km. Whitelee and
  Griffin are the one-node pair: 105.6 km, 1.01 spacings. Clyde and Dorenell
  are the two-node pair: 211.5 km, 2.03 spacings. Dudgeon and Race Bank, 2.6 km
  and 0.03 spacings, stay as a sanity check and should be nearly perfectly
  coupled; East Anglia One and Greater Gabbard, 29.6 km and 0.28 spacings, are
  inside one noise node and are not a test pair. H1 is two-sided: aggregate
  calibration error may be under- or over-dispersion. The underpowered-window
  line is a minimum detectable relative effect above 1.8 percent, three times
  the 0.6 percent STIPP shift from 9.52 to 9.46. Beside arm B, the Schaake
  shuffle is the baseline that trains nothing. Each arm is also settled at the
  Elexon System Price under one prespecified nomination, the ensemble mean of
  member power. This does not move the locked question in section 1.
- 25 Oct 2026, later still, before scoring. The 104.36 km spacing is replaced
  by node ids and hop counts from the AIFS ENS v2 checkpoint. Hidden nodes are
  O96, index in `grid-o96.npz`. Hops are shortest paths through the checkpoint's
  250 km data-to-hidden edges. Dudgeon and Race Bank are node 3744, hop 0.
  Whitelee (3402) and Griffin (3238) are hop 1. Beatrice (2923) and Hornsea 1
  (3573) are hop 2. No roster pair reaches hop 3. Whitelee and Clyde share node
  3402, so the earlier "two spacing" pair was the wrong scale. H1 cites Adamov,
  Fuhrer, Knutti and Schemm, arXiv:2609.08412, for a spatial-mean error that can
  be overdispersion; that paper is not a measurement of AIFS ENS. The 1.8 percent
  gate is three times a station wind energy score, not a portfolio CRPS. Beside
  the ensemble-mean settlement, which is invariant to coupling, the portfolio
  total is scored by pinball loss at 10 and 90 percent, and a second nomination
  is the 10 percent quantile. Jobs 39193411 and 39194625, same evening: the
  scheduler gave two A100-80GB devices and JAX saw both; one device runs two
  members for 42 hours, rematerialised, at 67,511 MiB and 12.0 seconds a step
  in steady state; four members on that device asked for 154.65 GiB and died.
  The choice between AIFS ENS and WeatherNext Mini waits on the one-step AIFS
  memory measurement. This does not move the locked question in section 1.
- 26 Oct 2026, data provenance, before any initial state is built and before
  scoring. Three changes.
  - *Initial states.* Both candidate checkpoints are built to start from
    ECMWF's operational analysis, not a reanalysis. The WeatherNext README
    loads HRES initial conditions and describes WeatherNext 2 as designed to be
    initialised from operational HRES rather than ERA5. The Mini checkpoint
    carries only the description `WeatherNext2_Mini_<2024` and its demo sample
    is `source-hres_forecast`. The AIFS ENS v2 checkpoint metadata names its
    training data as the operational analysis `aifs-od-an-oper`, N320, 2016 to
    2024. The primary source is therefore ECMWF open data step 0 of the IFS
    deterministic run, 0.25 degree, regridded to N320 for AIFS ENS and to 1
    degree for Mini (for AIFS ENS, superseded by the next amendment). The two frames six hours apart come from the 00/12 UTC
    main runs and the 06/18 UTC short cutoff runs, whose step 0 field lists
    were checked identical on 2 Jul 2025 and 26 Oct 2026. The Beyond Weather
    IFS analysis archive is the cross-check: on a stated sample of development
    dates, the frozen model is run from both and the difference in forecast
    skill is reported. ERA5 is not used as an initial state. MARS is not
    needed for 2025 to 2026.
  - *Field check, 26 Oct 2026, against the step 0 index files.* Mini's
    inputs are temperature, geopotential (from `gh` times g), u, v, vertical
    velocity (`w`) and specific humidity on 50, 100, 150, 200, 250, 300, 400,
    500, 600, 700, 850, 925 and 1000 hPa, all present; 2 m temperature, mean
    sea level pressure, 10 m wind, surface geopotential and land-sea mask,
    present. Sea surface temperature is not published in open data. Mini takes
    `skt` over sea points as its SST, and the cross-check dates report the
    difference against the Beyond Weather field. AIFS ENS's surface, soil and
    pressure-level inputs were fetched from the same source by the one-step
    job; its runner warns that `lsm`, `sdor`, `slor`, `wmb` and `z` are passed
    as constant rather than coupled forcings, which is checked before the
    first rollout is scored.
  - *Wind to power.* Live and archive scoring already use 100 m wind from
    both dynamical.org archives; 10 m wind is used only for the shear exponent
    from 100 m to hub height. The idealised curve is replaced by an empirical
    curve per farm: ERA5 hourly 100 m wind interpolated to the farm and
    extrapolated to hub height, against Elexon metered output, uncurtailed
    periods only, fitted on dates before 2 July 2025 so no scored window is
    touched. The hub-height step was dropped on 27 Oct. The fitted form is
    monotone and smooth, so it stays in the differentiable path, and every arm
    uses the same curve. A farm with too little uncurtailed history before that
    date keeps the idealised curve and is listed. The curve absorbs ERA5's
    local wind biases as well as the farm's real behaviour, so it is a mapping
    from ERA5-like wind to power, not a turbine curve. On development dates,
    ERA5 wind through the fitted curve is scored against metered output as the
    conversion floor. Describing that as the error left by perfect reanalysis
    weather was wrong: ERA5's error is in the number. Corrected on 27 Oct.
    Turbine SCADA and commercial power data are not used; that is a stated
    limitation.
  This does not move the locked question in section 1.
- 26 Oct 2026, later the same day, before any initial state is built. AIFS ENS
  starts from perturbed ensemble states. ECMWF's implementation page says
  operational AIFS ENS uses the IFS ensemble initial conditions regridded to
  N320, member n from IFS ENS member n. Starting every training member from one
  analysis would leave model noise as the only source of spread, while the
  MeteoSwiss study (arXiv:2609.08412) attributes about 90 percent of ensemble
  variance at 6 hours, and 50 to 70 percent at 24 hours, to initial
  perturbations. Arms C and D would then learn coupling without the spread
  that the scored operational system has. Each AIFS training member m
  therefore starts from open data `enfo` step 0 perturbed member m, both
  frames, valid at t minus 6 hours and t, as ECMWF's own AIFS notebook does.
  This departs from how the checkpoint was trained: its dataloader reads only
  `aifs-od-an-oper-...-for-single` and two parallel-suite operational analyses,
  2016 to 2024, one analysis per sample. It matches how the checkpoint runs,
  and the scored product is the operational one, so matching operation is the
  choice. WeatherNext Mini runs operationally from one analysis and keeps the
  deterministic step 0. Checked on 26 Oct against the archived index files:
  - The deterministic run has 13 pressure levels, 50 to 1000 hPa, on every date
    checked. AIFS ENS needs 10 hPa as well, so the deterministic run cannot
    initialise it at all. `enfo` carries 10 hPa from 00 UTC on 13 May 2026, and
    not at 00 UTC on 12 May or any earlier date checked. Development starts at
    the first cycle with 10 hPa in both frames.
  - For perturbed member 1 at 00 UTC on 20 May 2026, `enfo` holds every
    surface, soil (`sot` and `vsw`, levels 1 and 2) and pressure-level input,
    and `waef` holds all eleven wave inputs. `z`, `lsm`, `sdor` and `slor` are
    not in `enfo` and are taken once from the deterministic run. The 18 UTC
    `enfo` file has the 50 perturbed members and no control.
  - Volume, from the index byte lengths: 99 `enfo` fields of 73.4 MB plus 11
    `waef` fields of 9.0 MB, about 82 MB per member per frame, fetched by byte
    range. The frame at t minus 6 hours is step 0 of the previous cycle, so it is
    shared with the previous case. M members over 200 six-hourly cases is about
    M times 16 GB.
  - Unchecked: whether perturbed member n at consecutive cycles is a coherent
    pair. The EDA part of the perturbation cycles continuously, but the singular
    vector part is redrawn each cycle. This follows ECMWF's notebook, and the
    frozen arm A on these starts is compared with the operational AIFS ENS
    archive on the same dates before any training.
  Beyond Weather remains the cross-check for the analysis. MARS, if Cambridge
  has member-state access, would add SST and full resolution; nothing waits on
  it. This does not move the locked question in section 1.
- 26 Oct 2026, extensions only. X7 is rewritten as the next study, on what the
  farm network reveals about the weather evolution, tested by transfer to a
  withheld group of farms. It is sequenced after the core comparison, not
  substituted for it, and cites Li, Guan and Wu (2015) as prior art for power
  as an atmospheric observation. This does not move the locked question in
  section 1.
- 26 Oct 2026, power curves fitted, before scoring. ERA5 hourly 100 m and 10 m
  wind for each of the 17 farms, September 2020 to June 2026, from the
  Copernicus time-series dataset. It returns the nearest 0.25 degree grid
  point, not an interpolation. Each curve is the smooth logistic already used
  on the training path, with cut-in, rated speed and sharpness fitted by least
  squares to uncurtailed metered output before 2 July 2025. Every farm has
  between 30,548 and 41,457 such hours, so none falls back to the idealised
  curve. Hours above 20 m/s are scarce everywhere, and the onshore farms have
  few hours above 16 m/s, so the top of those curves is weakly constrained.
  On the development dates the median absolute error is 10.1 per cent of
  capacity. Hornsea 1 is the outlier, at 24.0 per cent. Parameters, bin counts
  and one figure per farm are in `results/power-curves/` on RDS. Superseded on
  27 Oct 2026 by the bilinear refit below; these two percentages are the
  nearest-point fit. This does not move the locked question in section 1.
- 27 Oct 2026, power curves refit, before any score. Job 39311089. Corrects the
  nearest-point fit and the floor wording above.
  - *Input.* ERA5 100 m wind at the four corners of the surrounding 0.25 degree
    cell, September 2020 through 30 June 2026, combined with bilinear weights.
    That is what a linear interpolation does on this cell. No hub-height step.
    In `gbwind.power` the same rule applies to a forecast: 100 m wind goes into
    the curve as it is, and the one-seventh extrapolation from 10 m wind is only
    there for a field that has no 100 m wind.
  - *Rated speed and cut-out.* The fitted curve is a logistic, so it only
    identifies a midpoint and a width. Cut-in and rated speed trade off. Left
    alone, rated speed came out between 11.4 and 15.4 m/s, including at the
    offshore farms, which are not short of hours above 16 m/s (Hornsea 1 has
    2,464 fitting hours in the 16 to 20 m/s bin). That spread is the tradeoff,
    not a measurement of the turbine. A Gaussian prior, mean 12.5 m/s and
    standard deviation 2 m/s, weighted as 100 hours, holds every fitted rated
    speed at 12.5 m/s. The unpenalised value is stored next to it. Cut-out is
    not fitted and not applied. A hard zero would block the training gradient,
    and hours at or above 25 m/s are scarce: 23 at Hornsea 1 and 4 at Whitelee
    in the whole fit. The idealised reference curve still cuts out at 25 m/s.
  - *Floor.* Development dates, 12 May 2026 through 30 June 2026. Median
    absolute error 10.2 per cent of effective capacity. Whitelee is 6.0 per
    cent; most farms sit between 8 and 13. This is the error with ERA5 100 m
    wind as the input. It includes ERA5 being wrong.
  - *Hornsea 1 stays.* Units T_HOWAO-1, T_HOWAO-2 and T_HOWAO-3, 400 MW each,
    present in every settlement period from September 2020. Maximum metered
    output 1198.5 MW. The 99.5th percentile before 12 May 2026 is 1197.7 MW.
    Not a missing unit, and not a capacity error. Metered fraction 0.9995,
    against the roster rule of 0.95, so the farm stays. On the development
    dates the absolute error is 291 MW, 24.3 per cent of capacity, mean error
    +231 MW, and 75 per cent of hours are over-predicted. 428 of the 1,165
    admissible hours are 3 to 21 June 2026, when all three units meter about
    zero and neither the acceptance flag nor the negative-price flag is set.
    Those hours are about 70 per cent of the absolute error. The other 737
    hours are 139 MW, 11.6 per cent of capacity. February to May 2024 is a
    smaller version of the same gap: monthly maxima of 781 to 826 MW against a
    farm that otherwise reaches about 1190, and the acceptance flag does not
    mark it. Those hours stay in the fit. The farm is not removed for scoring
    badly.
  - *What the code loads.* `config/power_curves.json`, written 27 Oct 2026.
    `capacity_mw` in `data/interim/portfolio.csv` is now the same 99.5th
    percentile the curves were fit to. Registered capacity is still in
    `registered_capacity_mw`. All 17 farms are selected. The live score task
    has not written a power score, so there is no earlier number to mix with
    this curve. The next image build copies the parameter file, and scores
    from that deploy onward use it.
  This does not move the locked question in section 1.
