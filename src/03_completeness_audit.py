"""
Diagnostic 3: Download stdmet history for all candidate offshore Gulf
buoys (2010-2023) and produce a per-station-per-month completeness
matrix plus heatmap figure.

Inputs
------
- data/processed/candidate_stations.csv (produced by
  src/02_filter_candidate_stations.py).

Per-station downloads
---------------------
- data/raw/stdmet/<station_id>.parquet : full 2010-2023 stdmet history,
  all columns ndbc-api returns, timestamp index, snappy-compressed.
- data/raw/stdmet/<station_id>.FAILED  : marker file containing the
  exception message if a station's download fails.

Downloads are resumable: a station whose .parquet already exists and
is >= 10 kB is skipped on subsequent runs. A .FAILED marker triggers
a re-attempt; a successful re-attempt deletes the marker. Writes are
atomic (write to .tmp then POSIX-replace) so an interrupted run never
leaves a partially-written .parquet that would pass the size check.

Derived outputs
---------------
- data/processed/completeness_matrix.csv : 34 stations x 168 months,
  values are percent of hours in each month with WVHT or WSPD present
  (0-100). Stations whose download failed are rows of zeros.
- figures/completeness_heatmap.png : viridis heatmap of the matrix.

Completeness metric: for each hour in [2010-01-01 00:00, 2023-12-31
23:00], the station counts as "observed" if any obs in that hour has
non-null WVHT or non-null WSPD. The metric is the fraction of
observed hours per calendar month, in percent.

Stations that came online after 2010 or retired before 2023 will show
0% completeness outside their lifetime; from the imputation problem's
perspective those hours genuinely are missing observations, and the
heatmap surfaces these "partial-life" stations visually.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save-to-file only; never opens a display
import matplotlib.pyplot as plt
import pandas as pd
from ndbc_api import NdbcApi
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "candidate_stations.csv"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "stdmet"
MATRIX_PATH = PROJECT_ROOT / "data" / "processed" / "completeness_matrix.csv"
FIGURE_PATH = PROJECT_ROOT / "figures" / "completeness_heatmap.png"

WINDOW_START = datetime(2010, 1, 1, 0, 0, 0)
WINDOW_END = datetime(2023, 12, 31, 23, 59, 59)
MIN_PARQUET_BYTES = 10 * 1024
REQUEST_SLEEP_SECONDS = 0.5
TARGET_COLS = ("WVHT", "WSPD")
USABLE_THRESHOLD_PCT = 75.0


def parquet_path(station_id: str) -> Path:
    return RAW_DIR / f"{station_id}.parquet"


def failed_path(station_id: str) -> Path:
    return RAW_DIR / f"{station_id}.FAILED"


def is_cached(station_id: str) -> bool:
    p = parquet_path(station_id)
    return p.exists() and p.stat().st_size >= MIN_PARQUET_BYTES


def fetch_station(api: NdbcApi, station_id: str) -> pd.DataFrame:
    """Pull full-window stdmet for one station. Raises on any error."""
    df = api.get_data(
        station_id=station_id,
        mode="stdmet",
        start_time=WINDOW_START,
        end_time=WINDOW_END,
        as_df=True,
        use_timestamp=True,
    )
    if not isinstance(df, pd.DataFrame):
        raise RuntimeError(f"unexpected return type {type(df).__name__}")
    return df


def download_all(
    api: NdbcApi, station_ids: list[str]
) -> tuple[list[str], list[str], list[str]]:
    """Returns (newly_downloaded, cached, failed)."""
    newly: list[str] = []
    cached: list[str] = []
    failed: list[str] = []
    for sid in tqdm(station_ids, desc="downloading stdmet", unit="stn"):
        if is_cached(sid):
            cached.append(sid)
            failed_path(sid).unlink(missing_ok=True)
            continue
        try:
            df = fetch_station(api, sid)
            final = parquet_path(sid)
            tmp = final.with_name(final.name + ".tmp")
            df.to_parquet(tmp, engine="pyarrow", compression="snappy")
            tmp.replace(final)  # atomic on POSIX
            failed_path(sid).unlink(missing_ok=True)
            newly.append(sid)
            print(f"  [{sid}] {len(df):>7} rows -> {final.name}")
        except Exception as exc:  # noqa: BLE001 - keep going on any failure
            msg = f"{type(exc).__name__}: {exc}"
            failed_path(sid).write_text(msg + "\n")
            failed.append(sid)
            print(f"  [{sid}] FAILED: {msg}")
        time.sleep(REQUEST_SLEEP_SECONDS)
    return newly, cached, failed


def hourly_observed_mask(df: pd.DataFrame) -> pd.Series:
    """Per-hour bool: True if any obs in that hour has WVHT or WSPD non-null."""
    if df.empty:
        return pd.Series(dtype=bool)
    have_cols = [c for c in TARGET_COLS if c in df.columns]
    if not have_cols:
        return pd.Series(dtype=bool)
    numeric = df[have_cols].apply(pd.to_numeric, errors="coerce")
    any_target = numeric.notna().any(axis=1)
    ts = df.index.get_level_values("timestamp") if isinstance(df.index, pd.MultiIndex) else df.index
    return any_target.groupby(ts.floor("h")).any()


def station_monthly_completeness(
    df: pd.DataFrame,
    month_index: pd.PeriodIndex,
    full_hours: pd.DatetimeIndex,
) -> pd.Series:
    observed = hourly_observed_mask(df).reindex(full_hours, fill_value=False)
    monthly = observed.groupby(observed.index.to_period("M")).mean() * 100.0
    return monthly.reindex(month_index, fill_value=0.0)


def build_matrix(station_ids: list[str]) -> tuple[pd.DataFrame, int]:
    month_index = pd.period_range(WINDOW_START, WINDOW_END, freq="M")
    full_hours = pd.date_range(
        WINDOW_START.replace(minute=0, second=0),
        WINDOW_END.replace(minute=0, second=0),
        freq="h",
    )
    matrix = pd.DataFrame(
        0.0, index=station_ids, columns=[str(p) for p in month_index]
    )
    total_rows = 0
    for sid in tqdm(station_ids, desc="computing completeness", unit="stn"):
        p = parquet_path(sid)
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p, engine="pyarrow")
        except Exception as exc:  # noqa: BLE001
            print(f"  [{sid}] could not read parquet: {exc}; row stays at zero")
            continue
        total_rows += len(df)
        pct = station_monthly_completeness(df, month_index, full_hours)
        matrix.loc[sid] = pct.values
    matrix.index.name = "station_id"
    return matrix, total_rows


def plot_heatmap(matrix: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
    im = ax.imshow(
        matrix.values,
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=100,
        interpolation="nearest",
    )
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=8)
    n_months = matrix.shape[1]
    year_tick_pos = list(range(0, n_months, 12))
    year_tick_labels = [matrix.columns[i].split("-")[0] for i in year_tick_pos]
    ax.set_xticks(year_tick_pos)
    ax.set_xticklabels(year_tick_labels)
    ax.set_xlabel("Year")
    ax.set_ylabel("Station ID")
    ax.set_title("Completeness of NDBC Gulf of Mexico Buoys, 2010-2023")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("Hourly observations present (%)")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> int:
    if not INPUT_PATH.exists():
        print(
            f"ERROR: candidate stations not found at {INPUT_PATH}. "
            f"Run src/02_filter_candidate_stations.py first.",
            file=sys.stderr,
        )
        return 1

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(INPUT_PATH, dtype={"station_id": str})
    station_ids = sorted(candidates["station_id"].tolist())
    print(
        f"Loaded {len(station_ids)} candidate stations from "
        f"{INPUT_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(f"Window: {WINDOW_START:%Y-%m-%d} to {WINDOW_END:%Y-%m-%d}")
    print()

    api = NdbcApi()
    newly, cached, failed = download_all(api, station_ids)
    print()
    print(f"Downloads: {len(newly)} new, {len(cached)} cached, {len(failed)} failed")

    succeeded = [s for s in station_ids if is_cached(s)]
    if not succeeded:
        print(
            "ERROR: no stations have usable parquet files. Aborting.",
            file=sys.stderr,
        )
        return 1

    print(f"Building completeness matrix ({len(station_ids)} rows x 168 months)...")
    matrix, total_rows = build_matrix(station_ids)
    matrix.to_csv(MATRIX_PATH)
    print(f"Wrote {MATRIX_PATH.relative_to(PROJECT_ROOT)}")

    print(f"Rendering heatmap to {FIGURE_PATH.relative_to(PROJECT_ROOT)}...")
    plot_heatmap(matrix, FIGURE_PATH)

    mean_completeness = float(matrix.values.mean())
    usable_share = float((matrix.values >= USABLE_THRESHOLD_PCT).mean()) * 100.0

    print()
    print("=" * 72)
    print(f"Stations attempted:                      {len(station_ids)}")
    print(f"Stations with usable parquet:            {len(succeeded)}")
    print(f"Stations failed:                         {len(failed)}")
    print(f"Total raw stdmet rows downloaded:        {total_rows:,}")
    print(f"Mean station-month completeness:         {mean_completeness:.1f}%")
    print(
        f"Station-months >= {USABLE_THRESHOLD_PCT:.0f}% complete:          "
        f"{usable_share:.1f}%"
    )
    print(
        f"Matrix:                                  "
        f"{MATRIX_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"Heatmap:                                 "
        f"{FIGURE_PATH.relative_to(PROJECT_ROOT)}"
    )
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
