"""
Diagnostic 15 — Baseline: k-Nearest Neighbors with Inverse-Distance Weighting (WSPD)
=====================================================================================

Spatial k=5 NN imputation. For each missing cell at (time t, station s):
    1. Take the 5 stations geographically nearest to s (precomputed
       from station_metadata["distance_matrix"]).
    2. Among them, take those observed in test_input at time t
       (i.e., non-NaN — NaN means D8-hidden or naturally missing).
    3. Inverse-distance-weighted mean: pred = sum(w_i * x_i) / sum(w_i),
       w_i = 1 / d_i.

Fallback chain when fewer than 2 of the 5 nearest neighbors are
observed at time t (storm periods, dense gaps):
    1. (station, month, hour) training climatology
    2. station-level training mean
    3. NaN

Climatology is computed inline (not imported from D14) so this script
is self-contained for paper-reproducibility purposes. The fitting logic
matches D14: per-station, per (month, hour) bin from observed training
cells only, with the bin -> station-mean -> NaN escalation built in.

Usage:  .venv/bin/python src/15_baseline_knn.py
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
K = 5
MIN_OBSERVED_NEIGHBORS = 2


def compute_climatology_predictions(
    train_observations: np.ndarray,
    train_mask: np.ndarray,
    T_test: int,
    variable_slot: int,
) -> np.ndarray:
    """Per-station (month, hour) climatology fitted on training rows only.

    Returns (T_test, n_stations) float64. Cells where the station has no
    training observations are NaN; cells whose (month, hour) bin is empty
    fall back to the station's overall training mean.
    """
    T_train = train_observations.shape[0]
    n_stations = train_observations.shape[1]
    train_ts = pd.date_range(TRAIN_START, periods=T_train, freq="h")
    train_months = train_ts.month.to_numpy()
    train_hours = train_ts.hour.to_numpy()
    test_ts = pd.date_range(TEST_START, periods=T_test, freq="h")
    test_lookup = pd.DataFrame(
        {"month": test_ts.month.to_numpy(), "hour": test_ts.hour.to_numpy()}
    )

    out = np.full((T_test, n_stations), np.nan, dtype=np.float64)
    for s in range(n_stations):
        train_vals = train_observations[:, s, variable_slot]
        train_obs_s = train_mask[:, s, variable_slot]
        if not train_obs_s.any():
            continue
        df = pd.DataFrame({
            "val": train_vals[train_obs_s],
            "month": train_months[train_obs_s],
            "hour": train_hours[train_obs_s],
        })
        clim = df.groupby(["month", "hour"])["val"].mean()
        station_mean = float(df["val"].mean())
        merged = test_lookup.merge(
            clim.reset_index().rename(columns={"val": "clim_val"}),
            on=["month", "hour"],
            how="left",
        )
        out[:, s] = merged["clim_val"].fillna(station_mean).to_numpy()
    return out


def knn_method_fn(
    train_observations,
    train_mask,
    test_input,
    test_observation_mask,
    test_hidden_mask,
    station_metadata,
):
    """Inverse-distance k-NN with (month, hour) climatology fallback."""
    variable_slot = station_metadata["variable_slot"]
    distance_matrix = station_metadata["distance_matrix"]
    n_stations = test_input.shape[1]
    T_test = test_input.shape[0]

    # Precompute k nearest non-self neighbors per station.
    nearest = np.argsort(distance_matrix, axis=1)              # (N, N)
    neighbor_idx = nearest[:, 1:K + 1]                         # (N, K) — skip self at col 0
    neighbor_dist = np.take_along_axis(distance_matrix, neighbor_idx, axis=1)
    weights_per_station = 1.0 / neighbor_dist                  # (N, K)

    # Climatology fallback array, computed once from training data.
    clim_pred = compute_climatology_predictions(
        train_observations, train_mask, T_test, variable_slot
    )  # (T_test, N)

    predictions = np.full_like(test_input, np.nan)
    for s in range(n_stations):
        n_idx = neighbor_idx[s]                                # (K,)
        w = weights_per_station[s]                             # (K,)
        neighbor_vals = test_input[:, n_idx, variable_slot]    # (T_test, K)
        obs_per_t = ~np.isnan(neighbor_vals)                   # (T_test, K)
        n_observed = obs_per_t.sum(axis=1)                     # (T_test,)

        masked_vals = np.where(obs_per_t, neighbor_vals, 0.0)
        masked_w = np.where(obs_per_t, np.broadcast_to(w, neighbor_vals.shape), 0.0)
        numerator = (masked_vals * masked_w).sum(axis=1)
        denominator = masked_w.sum(axis=1)
        safe_denom = np.where(denominator > 0, denominator, 1.0)
        knn_pred = np.where(denominator > 0, numerator / safe_denom, np.nan)

        sufficient = n_observed >= MIN_OBSERVED_NEIGHBORS
        predictions[:, s, variable_slot] = np.where(
            sufficient, knn_pred, clim_pred[:, s]
        )
    return predictions


if __name__ == "__main__":
    evaluate_method(
        method_fn=knn_method_fn,
        method_name="baseline_knn",
        variable_slot=1,
        method_config={
            "method": "knn_inverse_distance",
            "k": K,
            "weighting": "inverse_distance",
            "fallback_chain": "month_hour_climatology, station_mean, nan",
        },
    )
