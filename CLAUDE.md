# NDBC Wave & Wind Imputation/Forecasting Project

## What this project is
Spatiotemporal deep learning for NDBC buoy data. Two-part contribution:
imputation of missing sensor observations using graph-aware methods, and
downstream probabilistic forecasting of wave height and wind speed for
marine renewable energy applications. Collaborators in electrical
engineering consume the forecasts.

## Targets
- Significant wave height (WVHT)
- Wind speed (WSPD)

## Scope decisions (locked)
- Region: Gulf of Mexico
- Station class: offshore NDBC 42xxx buoys ONLY (no C-MAN towers, no
  NOS tide gauges, no Atlantic 41xxx stations). Homogeneous offshore
  network is the contribution; coastal/heterogeneous is a follow-up.
- Filter: has_stdmet == True (must measure both WVHT and WSPD)
- Observational window: 2010-01-01 through 2023-12-31
- Cadence: hourly (NDBC stdmet)
- Initial candidate count: 34 (from Diagnostic 1, 2026-05-25);
  final cluster size set by Diagnostic 2 (completeness audit).
  Target final size: 15-25 stations.
- Methods family: deep learning (graph-based imputation, probabilistic
  imputation via CSDI or equivalent, spatiotemporal forecasting)

## Open decisions
- Final station list (pending Diagnostic 2 completeness audit)
- Adjacency definition: distance-based vs learned vs physics-informed
- Probabilistic forecaster choice

## Repo layout
- data/raw/        NDBC files as pulled (gitignored)
- data/processed/  Cleaned/aligned tables (gitignored)
- src/             Python modules
- notebooks/       Exploratory and diagnostic notebooks
- figures/         Generated figures (gitignored by default)

## Conventions
- Python 3.12, pandas, numpy, matplotlib
- Use ndbc-api for data access; cross-check with ERDDAP if discrepancies
- Commit every meaningful step with descriptive messages
- Bounding-box-style "find everything in a region" filters are NOT
  sufficient on their own — always combine with station-class filters
  (e.g., station_id prefix) to avoid pulling in stations of the wrong
  type or wrong basin. Diagnostic 1 found 263 stations in the original
  Gulf bounding box; only 39 were actual offshore Gulf buoys.

## Diagnostic 1 result (2026-05-25)
1355 NDBC active stations -> 263 in (lat 18-31N, lon -98 to -80W)
bounding box -> 39 stations matching ^42\d{3}$ -> 34 after has_stdmet
filter -> these are the candidates we proceed with.

## Diagnostic 2 result (2026-05-26)

Attempted full 2010-2023 hourly stdmet download for all 34 candidate
stations. 27 succeeded, 7 returned empty dicts from ndbc-api despite
has_stdmet=True in NDBC's inventory (likely TABS / partner-network
buoys whose archived data lives outside NDBC's primary stdmet
endpoint). The 7 dropped: 42027, 42028, 42031, 42066, 42354, 42357,
42358. They appear as zero-rows in the completeness heatmap and are
to be excluded from the working set going forward.

Aggregate statistics across the 27 surviving stations:
- Total raw stdmet rows: 5,339,638
- Mean station-month completeness: 47.4%
- Station-months >= 75% complete: 46.0%
- Working candidate count after Diagnostic 2: 27

The mean completeness is pulled down by partial-lifetime stations
(42084, 42091, 42095, 42097, 42098 came online 2018-2021;
42067 has a single 2016-2017 deployment window). Whole-window
workhorses include 42001, 42035, 42036, 42039, 42040, 42055, 42056.

Storm-period outage signatures visible in 2017 (Harvey), 2020
(Laura/Sally/Delta), 2021 (Ida) — supports the storm-MNAR
stress-test framing in the research plan.

## Diagnostic 3 result (2026-05-26)

Computed pairwise Pearson correlations of deseasoned hourly WVHT and
WSPD anomalies across the 27 surviving stations. 351 possible pairs;
229 retained (pair excluded only if BOTH variables had <2 years of
overlapping observations). Per-variable inclusion: 165 pairs for WSPD,
130 pairs for WVHT.

Empirical decorrelation length scales (LOWESS-smoothed correlation
crossing 0.5):
- WSPD: ~400 km (consistent with synoptic-scale system widths)
- WVHT: ~650 km (consistent with basin-scale swell propagation)

Wave height correlations are systematically higher than wind speed
correlations by 0.1-0.2 across all observed distances (50-1400 km),
with the gap peaking near 400 km where wind has decorrelated but
waves remain coherent.

Implication for downstream imputation: the graph adjacency used for
WSPD imputation should have a tighter spatial kernel than the
adjacency used for WVHT imputation. A single shared adjacency for
both variables would mis-specify one of them. This is a candidate
methodological contribution for the paper or a methods spinout.

Methodology: hourly climatology (month-of-year x hour-of-day, 288
bins) removed per station per variable. Pairwise-complete overlap
indexing. Pearson and Spearman both computed and stored;
visualization currently shows Pearson only. LOWESS smoother
(frac=0.4) with 1000-iteration pair-level bootstrap 95% CI bands.
Lag-0 only; lag-aware correlation flagged as future appendix work.
Haversine distance; landmass-routing flagged as future refinement.

## Diagnostic 4 result (2026-05-26)

Constructed candidate adjacency matrices for the 27-station network
using Gaussian kernels with variable-specific sigma values calibrated
to match the empirical 0.5-decorrelation distances from Diagnostic 3:

  sigma_WSPD   = 400 / sqrt(2 ln 2) ≈ 340 km
  sigma_WVHT   = 650 / sqrt(2 ln 2) ≈ 552 km
  sigma_shared = sqrt(sigma_WSPD * sigma_WVHT) ≈ 433 km

Four matrices written to data/processed/adjacency_*.npy:
A_WSPD, A_WVHT, A_shared, A_uniform (1/26 off-diagonal baseline).
Effective edges (> 0.01): WSPD 504, WVHT 702, shared 638, uniform 702.

The kernel calibration figure (figures/kernel_calibration.png)
exposed a systematic misfit between the Gaussian kernel and the
empirical LOWESS curve: the Gaussian overshoots short-range
correlations and undershoots the long-range tail on both variables.
This misfit motivated the Diagnostic 5 kernel-family comparison
below; the Gaussian adjacencies remain on disk as baseline-rejection
evidence and as comparison points for Phase 3 imputation ablation.

## Diagnostic 5 result (2026-05-26)

Fitted three single-parameter kernel families (Gaussian, Exponential,
Matern-1.5) to the empirical decorrelation curves for WSPD and WVHT,
scored by RMSE against the per-pair scatter.

Per-variable fits and RMSE rankings:

  WSPD (165 pairs):
    Gaussian:    ell = 332 km, RMSE = 0.1271, half_corr = 391 km
    Exponential: ell = 459 km, RMSE = 0.0558, half_corr = 318 km
    Matern-1.5:  ell = 377 km, RMSE = 0.0899, half_corr = 365 km

  WVHT (130 pairs):
    Gaussian:    ell = 495 km, RMSE = 0.1380, half_corr = 583 km
    Exponential: ell = 715 km, RMSE = 0.1021, half_corr = 495 km
    Matern-1.5:  ell = 560 km, RMSE = 0.1095, half_corr = 542 km

DECISION: adopt the exponential kernel
  K(d, ell) = exp(-d / ell)
with variable-specific length scales for the imputation pipeline.
The Gaussian kernel committed to in Diagnostic 4 is retained as a
baseline for ablation experiments. Justification:

  1. Exponential wins both panels by RMSE.
  2. RMSE gaps are substantial (2.3x lower than Gaussian on WSPD,
     1.35x lower on WVHT) and not within fit noise.
  3. The same kernel family wins both variables, simplifying the
     methods description: variable-specific length scales, single
     kernel family.
  4. The exponential kernel corresponds to a Matern-0.5 covariance
     (first-order Markov spatial structure), which is the physically
     correct expectation for non-differentiable atmospheric and
     oceanographic fields. Gaussian kernels imply infinitely
     differentiable underlying fields, which is unphysical for these
     variables.

The Gaussian-based adjacency matrices from Diagnostic 4 stay on
disk (data/processed/adjacency_*.npy) as the baseline-rejection
evidence and as comparison points for Phase 3 imputation ablation.

Diagnostic 6 will recompute the four adjacency matrices using the
exponential kernel.

## Diagnostic 6 result (2026-05-26)

Recomputed the four candidate adjacency matrices using the exponential
kernel adopted in Diagnostic 5, written under the _exp filename suffix
alongside the D4 Gaussian matrices.

Fitted exponential length scales (from D5 kernel_fit_results.csv,
verified at startup):
  ELL_WSPD   = 459.1 km   (half_corr ≈ 318 km)
  ELL_WVHT   = 714.8 km   (half_corr ≈ 495 km)
  ELL_SHARED = sqrt(ELL_WSPD * ELL_WVHT) ≈ 572.9 km

Per-matrix statistics (27-station network, 702 off-diagonal entries):

  A_WSPD_exp   sum=205.0  min=0.0346  max=0.9039  effective_edges=702
  A_WVHT_exp   sum=295.9  min=0.1152  max=0.9372  effective_edges=702
  A_shared_exp sum=249.1  min=0.0675  max=0.9222  effective_edges=702
  A_uniform    sum= 27.0  min=0.0385  max=0.0385  effective_edges=702
               (bit-for-bit identical to D4 uniform; saved under both
               adjacency_uniform.npy and adjacency_uniform_exp.npy for
               naming consistency)

Key structural observation: under the exponential kernel, every off-
diagonal entry across all three distance-based matrices exceeds the
0.01 effective-edges threshold. The Gaussian-WSPD matrix from D4 was
effectively sparse (504/702 edges); the exponential-WSPD matrix is
effectively dense (702/702). This is the heavy-tailed property of the
exponential kernel made numerical: distant station pairs receive small
but non-negligible adjacency weight, where the Gaussian kernel
effectively zeroed them out. Downstream graph imputation models will
see substantially more cross-station signal under the exponential
adjacency.

The Gaussian adjacencies from D4 remain on disk for Phase 3 ablation:
each imputation experiment in Phase 3 will be run under both kernel
families to quantify the impact of the kernel choice on imputation
accuracy.

End of Phase 2 (graph adjacency construction). Phase 3 (imputation
model training) is the next methodological step.

## Diagnostic 7 result (2026-05-26)

Assembled the 27 per-station Parquet files into a single
(122712, 27, 2) float64 observation tensor for downstream Phase 3
imputation training. Variable axis order locked as [WVHT, WSPD]
(slot 0 = WVHT, slot 1 = WSPD). Station order verified against
adjacency_station_order.csv from Phase 2 at script startup; mismatch
would have aborted with exit 1 since the tensor's N axis must align
with the adjacency matrices.

Outputs:
  observations.npy       (122712, 27, 2) float64, NaN for missing
  observation_mask.npy   (122712, 27, 2) bool,    True iff observed
  tensor_metadata.json   shape, dimensions, time range, station
                         and variable order, per-variable and
                         per-station missingness rates

Tensor-level statistics:
  Total cells:          6,626,448
  Observed cells:       3,166,016
  Overall missingness:  0.5222
  WVHT missingness:     0.5434
  WSPD missingness:     0.5010

WSPD missingness is lower than WVHT missingness, consistent with the
D3 pair-inclusion asymmetry (165 WSPD pairs cleared the two-year
overlap threshold versus 130 WVHT pairs). Wind-sensor reliability
exceeds wave-sensor reliability on the offshore Gulf fleet, and the
tensor representation preserves this asymmetry. Tensor footprint
50.6 MB; mask footprint 6.3 MB.

D7's 52.2% overall missingness is slightly higher than D2's 47.4%
mean completeness (so 52.6% missingness) because D2 counted any-of-
WVHT-or-WSPD as observed at a given hour while D7 counts each
variable separately. The 5-point gap is the fraction of cells where
one variable is reported and the other is not.

End of Phase 3 data foundation. Diagnostic 8 builds the evaluation
masks that will score every Phase 3 imputation model against the
same held-out cells.

## Diagnostic 8 result (2026-05-26)

Designed the temporal train/val/test split and pre-computed 15
evaluation masks for the Phase 3 imputation pipeline. Training-time
masks are generated on the fly by the training script and are NOT
materialized here; only the evaluation masks (used to score all
Phase 3 models against identical held-out cells) are produced.

Temporal split (locked):
  train:  2010-01-01 00:00 .. 2019-12-31 23:00   (87,648 hours)
  val:    2020-01-01 00:00 .. 2021-12-31 23:00   (17,544 hours)
  test:   2022-01-01 00:00 .. 2023-12-31 23:00   (17,520 hours)
  Sum:    122,712 hours (matches full time axis)

Each split contains at least one major Gulf hurricane: Harvey
(2017, train), Laura/Sally/Delta (2020) and Ida (2021, val),
Ian (2022, test). Storm-period generalization is therefore tested
at every stage of model development.

Mask types and parameters:
  MCAR:        rates 10%, 25%, 50%; 3 seeds each (9 masks)
  Block MCAR:  rate 25%, 72-hr block length; 3 seeds (3 masks)
  Empirical MNAR: rate 25%, 80% of hidden cells in storm windows;
                  3 seeds (3 masks)
  Total: 15 masks. All target the test split for this iteration;
  cells outside test are always False so the mask shape stays
  (T, N, V) bool for broadcasting convenience.

DECISION (storm-window detection): a "storm hour" is one in which
at least 25% of currently-deployed stations are simultaneously
missing (transient missingness within each station's deployment
window, NOT pre-deployment or post-retirement absence). The
fractional threshold replaces the original absolute threshold
because the latter biased early years (when only 21 stations were
deployed) against later years (27 stations deployed). The
deployment-window filter was added after the initial implementation
flagged every hour from 2010-2017 as a storm hour due to six
late-deployment stations being trivially "missing" before they
existed.

Storm-window summary statistics (after +/- 12 hr buffer):
  WVHT: 33,759 storm hours; 455 windows; 55,074 hrs (0.449 of time)
  WSPD: 46,715 storm hours; 493 windows; 77,218 hrs (0.629 of time)

WSPD coverage exceeds WVHT coverage; this is acceptable because the
80/20 MNAR sampling still concentrates hidden cells in storm-
correlated periods even with broader windows. The figure
(figures/missingness_masks.png) shows MNAR masks visually
concentrated in Jan-Sep 2022 (the storm-rich first half of the
test period, including Hurricane Ian in September), with sparse
banding in 2023.

Achieved hidden fractions: all masks land within 0.0001 of target
rate (floor() rounding error on cell-count math).

Numbering note: this is the eighth diagnostic but the ninth script
file (src/09_missingness_masks.py). The file/diagnostic offset
accumulated earlier in the project and is preserved.

Phase 3 data foundation (tensor + masks + split) is now complete.
Diagnostic 9 is the PyTorch + PyTorch Geometric + GRIN reference
implementation environment setup.
