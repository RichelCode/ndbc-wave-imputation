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
