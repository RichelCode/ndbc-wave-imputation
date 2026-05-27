"""
Diagnostic 13 — Baseline: Linear Interpolation (WSPD)
=====================================================

Per-station per-variable temporal linear interpolation, evaluated against
the 15 D8 evaluation masks on the test split (2022-2023).

Method: for each (station, WSPD) time series in test_input, fill NaN cells
via pandas.Series.interpolate(method='linear', limit_direction='both').
Surrounding observed cells in the same split provide anchors. Boundary
cells (start/end of test span with no neighbor on one side) take the
nearest observed value via limit_direction='both'. Training data is
intentionally NOT included to avoid edge-leakage.

Stations with 100% WSPD missingness across the test split produce all-NaN
predictions for that station-variable; the framework's score_mask skips
them since they have no observed cells to score against.

Usage:  .venv/bin/python src/13_baseline_linear.py
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


def linear_interpolation_method_fn(
    train_observations,
    train_mask,
    test_input,
    test_observation_mask,
    test_hidden_mask,
    station_metadata,
):
    """Per-station linear temporal interpolation on the variable_slot column."""
    variable_slot = station_metadata["variable_slot"]
    predictions = np.full_like(test_input, np.nan)
    n_stations = test_input.shape[1]
    for n in range(n_stations):
        series_vals = test_input[:, n, variable_slot]
        if np.isnan(series_vals).all():
            continue
        s = pd.Series(series_vals)
        interpolated = s.interpolate(method="linear", limit_direction="both")
        predictions[:, n, variable_slot] = interpolated.to_numpy()
    return predictions


if __name__ == "__main__":
    evaluate_method(
        method_fn=linear_interpolation_method_fn,
        method_name="baseline_linear",
        variable_slot=1,
        method_config={
            "method": "linear_interpolation",
            "library": "pandas",
            "limit_direction": "both",
        },
    )
