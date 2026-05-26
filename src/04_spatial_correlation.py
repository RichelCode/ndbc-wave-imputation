"""
Diagnostic 4: Pairwise correlation of deseasoned WVHT and WSPD anomalies
as a function of inter-station distance, for the 27 surviving offshore
Gulf of Mexico buoys (2010-2023).

Inputs
------
- data/raw/stdmet/*.parquet      (per-station hourly stdmet, written by
  src/03_completeness_audit.py).
- data/processed/candidate_stations.csv (station_id, latitude, longitude).

Outputs
-------
- data/processed/pairwise_correlations.csv : one row per included pair
  (station_a < station_b lexicographically). Columns:
    station_a, station_b, distance_km,
    n_overlap_hours_wvht, n_overlap_hours_wspd,
    pearson_wvht, pearson_wspd, spearman_wvht, spearman_wspd.
  Per-variable correlation cells are NaN when that variable's overlap
  is < 17,520 hours (2 years). A pair is excluded from the file
  entirely if BOTH variables fail the overlap threshold.
- figures/decorrelation_curves.png : 1x3 panel figure (WSPD, WVHT,
  WVHT-WSPD difference) with LOWESS smoothers and 95% bootstrap CI
  bands on the first two panels.

Methodology
-----------
1. Discover stations by glob of data/raw/stdmet/*.parquet so the 7
   stations that failed download in Diagnostic 2 are excluded
   automatically. Lat/lon comes from candidate_stations.csv.
2. Per station: read parquet, take WVHT and WSPD, resample to hourly
   mean, reindex to the full 122,712-hour grid [2010-01-01 00:00,
   2023-12-31 23:00], subtract a (month-of-year, hour-of-day)
   climatology to get hourly anomalies.
3. For every unordered pair, compute haversine distance, per-variable
   overlap counts, and Pearson + Spearman on the pairwise-complete
   overlapping hours.
4. Plot decorrelation curves with LOWESS (frac=0.4) and 95% bootstrap
   CI bands (1000 pair-level resamples, fixed seed 42).

Lag-aware correlation is intentionally out of scope here. Downstream
graph imputers (GRIN, SPIN) operate at lag 0, so the lag-0 metric is
what governs adjacency-weight design. Lag sensitivity is a planned
appendix.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "stdmet"
STATIONS_PATH = PROJECT_ROOT / "data" / "processed" / "candidate_stations.csv"
PAIRS_PATH = PROJECT_ROOT / "data" / "processed" / "pairwise_correlations.csv"
FIGURE_PATH = PROJECT_ROOT / "figures" / "decorrelation_curves.png"

WINDOW_START = pd.Timestamp("2010-01-01 00:00")
WINDOW_END = pd.Timestamp("2023-12-31 23:00")
TARGETS = ("WVHT", "WSPD")
MIN_OVERLAP_HOURS = 17_520  # 2 years of hourly samples
N_BOOTSTRAP = 1000
LOWESS_FRAC = 0.4
BOOT_GRID_POINTS = 100
EARTH_RADIUS_KM = 6371.0
RNG_SEED = 42


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def load_anomalies(parquet_path: Path, full_hours: pd.DatetimeIndex) -> pd.DataFrame:
    """Read one station and return (WVHT, WSPD) hourly anomalies on full_hours.

    DataFrame is indexed by full_hours, columns ['WVHT', 'WSPD'], dtype
    float64. Hours with no underlying obs are NaN; NaN propagates through
    climatology subtraction so the downstream overlap mask is meaningful.
    """
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    ts = (
        df.index.get_level_values("timestamp")
        if isinstance(df.index, pd.MultiIndex)
        else df.index
    )
    keep = [c for c in TARGETS if c in df.columns]
    raw = df[keep].copy()
    raw.index = ts
    raw = raw.apply(pd.to_numeric, errors="coerce")
    for col in TARGETS:
        if col not in raw.columns:
            raw[col] = np.nan
    raw = raw[list(TARGETS)]
    hourly = raw.resample("h").mean().reindex(full_hours)
    anomalies = pd.DataFrame(
        index=full_hours, columns=list(TARGETS), dtype="float64"
    )
    for col in TARGETS:
        s = hourly[col]
        clim = s.groupby([s.index.month, s.index.hour]).transform("mean")
        anomalies[col] = s - clim
    return anomalies


def compute_pair_correlations(
    a_id: str,
    b_id: str,
    a_anom: pd.DataFrame,
    b_anom: pd.DataFrame,
    distance_km: float,
) -> dict | None:
    """One pair's row of the output table, or None if both variables fail."""
    row: dict = {
        "station_a": a_id,
        "station_b": b_id,
        "distance_km": distance_km,
    }
    kept_any = False
    for col in TARGETS:
        mask = (a_anom[col].notna() & b_anom[col].notna()).to_numpy()
        n = int(mask.sum())
        row[f"n_overlap_hours_{col.lower()}"] = n
        if n >= MIN_OVERLAP_HOURS:
            a_vals = a_anom[col].to_numpy()[mask]
            b_vals = b_anom[col].to_numpy()[mask]
            row[f"pearson_{col.lower()}"] = float(pearsonr(a_vals, b_vals).statistic)
            row[f"spearman_{col.lower()}"] = float(spearmanr(a_vals, b_vals).statistic)
            kept_any = True
        else:
            row[f"pearson_{col.lower()}"] = np.nan
            row[f"spearman_{col.lower()}"] = np.nan
    return row if kept_any else None


def bootstrap_lowess_ci(
    x: np.ndarray, y: np.ndarray, grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pair-level resampling: 95% CI band for LOWESS evaluated on `grid`."""
    n = len(x)
    boot = np.full((N_BOOTSTRAP, len(grid)), np.nan)
    for i in range(N_BOOTSTRAP):
        idx = np.random.randint(0, n, size=n)
        try:
            boot[i] = lowess(y[idx], x[idx], frac=LOWESS_FRAC, xvals=grid)
        except Exception:  # noqa: BLE001 - drop bad resample, keep going
            pass
    lower = np.nanpercentile(boot, 2.5, axis=0)
    upper = np.nanpercentile(boot, 97.5, axis=0)
    return lower, upper


def plot_decorrelation(pairs: pd.DataFrame, output_path: Path) -> None:
    np.random.seed(RNG_SEED)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)

    for corr_col, title, ax, color in [
        ("pearson_wspd", "Wind speed (WSPD)", axes[0], "C0"),
        ("pearson_wvht", "Wave height (WVHT)", axes[1], "C0"),
    ]:
        sub = pairs.dropna(subset=[corr_col])
        x = sub["distance_km"].to_numpy()
        y = sub[corr_col].to_numpy()
        if len(x) >= 2:
            grid = np.linspace(x.min(), x.max(), BOOT_GRID_POINTS)
            smooth = lowess(y, x, frac=LOWESS_FRAC, xvals=grid)
            lo, hi = bootstrap_lowess_ci(x, y, grid)
            ax.fill_between(grid, lo, hi, color=color, alpha=0.2,
                            label="95% bootstrap CI")
            ax.plot(grid, smooth, color=color, linewidth=2,
                    label=f"LOWESS (frac={LOWESS_FRAC})")
        ax.scatter(x, y, s=14, alpha=0.5, color=color)
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Pearson correlation")
        ax.set_ylim(-0.2, 1.0)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    ax = axes[2]
    sub = pairs.dropna(subset=["pearson_wvht", "pearson_wspd"])
    x = sub["distance_km"].to_numpy()
    y = (sub["pearson_wvht"] - sub["pearson_wspd"]).to_numpy()
    ax.scatter(x, y, s=14, alpha=0.5, color="C3")
    if len(x) >= 2:
        grid = np.linspace(x.min(), x.max(), BOOT_GRID_POINTS)
        smooth = lowess(y, x, frac=LOWESS_FRAC, xvals=grid)
        ax.plot(grid, smooth, color="C3", linewidth=2,
                label=f"LOWESS (frac={LOWESS_FRAC})")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Correlation difference")
    ax.set_ylim(-0.5, 0.5)
    ax.set_title("WVHT - WSPD correlation difference")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Decorrelation of NDBC Gulf buoys, 2010-2023 "
        "(deseasoned hourly anomalies)",
        fontsize=13,
    )
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def summarize(pairs: pd.DataFrame, n_stations: int, n_excluded: int) -> None:
    total_pairs = n_stations * (n_stations - 1) // 2
    print()
    print("=" * 72)
    print(f"Stations included:                       {n_stations}")
    print(f"Pairs computed (n*(n-1)/2):              {total_pairs}")
    print(f"Pairs excluded (both vars low overlap):  {n_excluded}")
    print(f"Pairs in output:                         {len(pairs)}")
    for var in ("wspd", "wvht"):
        col = f"pearson_{var}"
        sub = pairs.dropna(subset=[col])
        print()
        print(f"  {var.upper()}:")
        print(f"    included pairs:                      {len(sub)}")
        if len(sub):
            print(
                f"    correlation min/median/max:          "
                f"{sub[col].min():.3f} / {sub[col].median():.3f} / "
                f"{sub[col].max():.3f}"
            )
            strong = sub[sub[col] > 0.5]
            if len(strong):
                print(
                    f"    distance range with corr > 0.5:      "
                    f"{strong['distance_km'].min():.0f} - "
                    f"{strong['distance_km'].max():.0f} km "
                    f"({len(strong)} pairs)"
                )
            else:
                print("    distance range with corr > 0.5:      none")
    print()
    print(
        f"Pairs CSV:                               "
        f"{PAIRS_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"Figure:                                  "
        f"{FIGURE_PATH.relative_to(PROJECT_ROOT)}"
    )
    print("=" * 72)


def main() -> int:
    parquets = sorted(RAW_DIR.glob("*.parquet"))
    if len(parquets) < 2:
        print(
            f"ERROR: need >= 2 station parquets in {RAW_DIR}, "
            f"found {len(parquets)}.",
            file=sys.stderr,
        )
        return 1
    if not STATIONS_PATH.exists():
        print(
            f"ERROR: candidate stations not found at {STATIONS_PATH}.",
            file=sys.stderr,
        )
        return 1

    PAIRS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    stations = pd.read_csv(STATIONS_PATH, dtype={"station_id": str}).set_index(
        "station_id"
    )
    station_ids = sorted(p.stem for p in parquets)
    missing = [sid for sid in station_ids if sid not in stations.index]
    if missing:
        print(
            f"ERROR: parquet stations missing lat/lon: {missing}",
            file=sys.stderr,
        )
        return 1

    full_hours = pd.date_range(WINDOW_START, WINDOW_END, freq="h")
    print(
        f"Window: {WINDOW_START} .. {WINDOW_END} "
        f"({len(full_hours):,} hours)"
    )
    print(f"Loading anomalies for {len(station_ids)} stations...")

    anomalies: dict[str, pd.DataFrame] = {}
    for sid in tqdm(station_ids, desc="deseasoning", unit="stn"):
        anomalies[sid] = load_anomalies(RAW_DIR / f"{sid}.parquet", full_hours)

    pairs_total = len(station_ids) * (len(station_ids) - 1) // 2
    print(f"Computing {pairs_total} pairwise correlations...")
    rows: list[dict] = []
    n_excluded = 0
    for a, b in tqdm(
        list(combinations(station_ids, 2)), desc="pairs", unit="pair"
    ):
        d = haversine_km(
            stations.loc[a, "latitude"], stations.loc[a, "longitude"],
            stations.loc[b, "latitude"], stations.loc[b, "longitude"],
        )
        row = compute_pair_correlations(a, b, anomalies[a], anomalies[b], d)
        if row is None:
            n_excluded += 1
        else:
            rows.append(row)

    pairs = pd.DataFrame(rows)
    pairs.to_csv(PAIRS_PATH, index=False)
    print(f"Wrote {PAIRS_PATH.relative_to(PROJECT_ROOT)} ({len(pairs)} rows)")

    print(
        f"Rendering decorrelation figure ({N_BOOTSTRAP} bootstrap resamples)..."
    )
    plot_decorrelation(pairs, FIGURE_PATH)
    print(f"Wrote {FIGURE_PATH.relative_to(PROJECT_ROOT)}")

    summarize(pairs, len(station_ids), n_excluded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
