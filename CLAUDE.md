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
