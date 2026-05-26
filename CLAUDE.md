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
