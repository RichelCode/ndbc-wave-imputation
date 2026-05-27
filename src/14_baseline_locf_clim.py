"""
Diagnostic 14 — Baselines: LOCF + Climatology (WSPD)
====================================================

Two trivial baselines in one script, evaluated against the 15 D8 masks.

Method 1 — LOCF (Last Observation Carried Forward):
    For each (station, WSPD) test series, fill NaN with the most recent
    prior observed value via pandas.Series.ffill(), then .bfill() to
    handle leading NaN at the start of the test span.

Method 2 — Climatology:
    For each station, compute mean WSPD by (month-of-year, hour-of-day)
    from training rows ONLY (no leakage). 12 x 24 = 288 bins per
    station. For each test cell, predict the climatological value for
    its (month, hour). Fallback: station-level training mean if the
    bin is empty; NaN if the station has no training observations.

    Timestamps are derived from the canonical project anchors:
    train starts 2010-01-01 00:00 UTC, test starts 2022-01-01 00:00 UTC.

Usage:  .venv/bin/python src/14_baseline_locf_clim.py
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
framework = import_module("12_baselines_framework")
evaluate_method = framework.evaluate_method

TRAIN_START = "2010-01-01 00:00"
TEST_START = "2022-01-01 00:00"


def locf_method_fn(
    train_observations,
    train_mask,
    test_input,
    test_observation_mask,
    test_hidden_mask,
    station_metadata,
):
    """Per-station LOCF on the variable_slot column: ffill then bfill."""
    variable_slot = station_metadata["variable_slot"]
    predictions = np.full_like(test_input, np.nan)
    n_stations = test_input.shape[1]
    for n in range(n_stations):
        series_vals = test_input[:, n, variable_slot]
        if np.isnan(series_vals).all():
            continue
        s = pd.Series(series_vals)
        predictions[:, n, variable_slot] = s.ffill().bfill().to_numpy()
    return predictions


def climatology_method_fn(
    train_observations,
    train_mask,
    test_input,
    test_observation_mask,
    test_hidden_mask,
    station_metadata,
):
    """Per-station (month, hour) climatology fitted on training rows only."""
    variable_slot = station_metadata["variable_slot"]
    predictions = np.full_like(test_input, np.nan)
    n_stations = test_input.shape[1]
    T_train = train_observations.shape[0]
    T_test = test_input.shape[0]

    train_ts = pd.date_range(TRAIN_START, periods=T_train, freq="h")
    train_months = train_ts.month.to_numpy()
    train_hours = train_ts.hour.to_numpy()
    test_ts = pd.date_range(TEST_START, periods=T_test, freq="h")
    test_lookup = pd.DataFrame(
        {"month": test_ts.month.to_numpy(), "hour": test_ts.hour.to_numpy()}
    )

    for n in range(n_stations):
        train_vals = train_observations[:, n, variable_slot]
        train_obs_n = train_mask[:, n, variable_slot]
        if not train_obs_n.any():
            continue
        df = pd.DataFrame({
            "val": train_vals[train_obs_n],
            "month": train_months[train_obs_n],
            "hour": train_hours[train_obs_n],
        })
        clim = df.groupby(["month", "hour"])["val"].mean()
        station_mean = float(df["val"].mean())
        merged = test_lookup.merge(
            clim.reset_index().rename(columns={"val": "clim_val"}),
            on=["month", "hour"],
            how="left",
        )
        pred_series = merged["clim_val"].fillna(station_mean).to_numpy()
        predictions[:, n, variable_slot] = pred_series
    return predictions


if __name__ == "__main__":
    evaluate_method(
        method_fn=locf_method_fn,
        method_name="baseline_locf",
        variable_slot=1,
        method_config={
            "method": "last_observation_carried_forward",
            "library": "pandas",
            "fill_strategy": "ffill_then_bfill",
        },
    )
    evaluate_method(
        method_fn=climatology_method_fn,
        method_name="baseline_climatology",
        variable_slot=1,
        method_config={
            "method": "climatology_by_month_hour",
            "train_period": "2010-2019",
            "n_bins_per_station": 288,
            "bin_fallback": "station_training_mean",
            "station_fallback": "nan",
        },
    )
