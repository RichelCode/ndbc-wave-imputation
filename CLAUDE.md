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
- Region: Gulf of Mexico, central/north-central sub-region
- Observational window: 2010-01-01 through 2023-12-31
- Cadence: hourly (NDBC stdmet)
- Cluster size: 15-20 stations, selected by empirical diagnostics
  (completeness + correlation-vs-distance), NOT by radius
- Methods family: deep learning (graph-based imputation, probabilistic
  imputation via CSDI or equivalent, spatiotemporal forecasting)

## Open decisions
- Final station list (pending diagnostics)
- Adjacency definition: distance-based vs learned vs physics-informed
- Probabilistic forecaster choice

## Repo layout
- data/raw/        NDBC files as pulled (gitignored)
- data/processed/  Cleaned/aligned tables (gitignored)
- src/             Python modules
- notebooks/       Exploratory and diagnostic notebooks
- figures/         Generated figures (gitignored by default)

## Conventions
- Python 3.11+, pandas, numpy, matplotlib
- Use ndbc-api for data access; cross-check with ERDDAP if discrepancies
- Commit every meaningful step with descriptive messages
