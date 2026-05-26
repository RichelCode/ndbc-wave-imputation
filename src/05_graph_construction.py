"""
Diagnostic 4 (Phase 2 setup): Build candidate adjacency matrices for
the 27-station offshore Gulf of Mexico buoy network, using the
empirical decorrelation length scales from Diagnostic 3.

Inputs
------
- data/raw/stdmet/*.parquet      (station discovery only; not read).
- data/processed/candidate_stations.csv (station_id, latitude, longitude).
- data/processed/pairwise_correlations.csv (D3 output; re-derives the
  LOWESS smoothers shown in the kernel-calibration figure).

Outputs
-------
- data/processed/distance_matrix.npy    : 27 x 27 haversine distances (km).
- data/processed/adjacency_wspd.npy     : Gaussian, sigma = 340 km.
- data/processed/adjacency_wvht.npy     : Gaussian, sigma = 552 km.
- data/processed/adjacency_shared.npy   : Gaussian, sigma = 433 km.
- data/processed/adjacency_uniform.npy  : 1/(n-1) off-diagonal baseline.
- data/processed/adjacency_station_order.csv : station_id row/column order
  for the matrices above. Downstream code MUST join on this.
- figures/kernel_calibration.png        : LOWESS vs Gaussian, per variable.
- figures/adjacency_heatmaps.png        : 2x2 of the four adjacencies.

Methodology
-----------
1. Station set: glob data/raw/stdmet/*.parquet, sort lex, take stems.
2. Vectorized haversine on broadcast lat/lon vectors yields the 27x27
   distance matrix (R = 6371 km, same formula as D3).
3. Gaussian kernel K(d, sigma) = exp(-d^2 / (2 sigma^2)) with sigma
   calibrated so K crosses 0.5 at the empirical D3 decorrelation
   scale: sigma = scale / sqrt(2 ln 2). Diagonal forced to 0 to
   remove self-loops. No thresholding, no sparsification, no row
   normalization -- those are downstream choices for the imputer.
4. The uniform baseline is 1/(n-1) off-diagonal, 0 on the diagonal;
   it is the maximum-entropy reference any graph-aware imputer
   must beat to justify its existence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "stdmet"
STATIONS_PATH = PROJECT_ROOT / "data" / "processed" / "candidate_stations.csv"
PAIRS_PATH = PROJECT_ROOT / "data" / "processed" / "pairwise_correlations.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"

DISTANCE_PATH = PROCESSED_DIR / "distance_matrix.npy"
ADJ_WSPD_PATH = PROCESSED_DIR / "adjacency_wspd.npy"
ADJ_WVHT_PATH = PROCESSED_DIR / "adjacency_wvht.npy"
ADJ_SHARED_PATH = PROCESSED_DIR / "adjacency_shared.npy"
ADJ_UNIFORM_PATH = PROCESSED_DIR / "adjacency_uniform.npy"
ORDER_PATH = PROCESSED_DIR / "adjacency_station_order.csv"
KERNEL_FIG_PATH = FIGURES_DIR / "kernel_calibration.png"
ADJ_FIG_PATH = FIGURES_DIR / "adjacency_heatmaps.png"

EARTH_RADIUS_KM = 6371.0
LOWESS_FRAC = 0.4
LOWESS_GRID_POINTS = 100

# Empirical decorrelation scales from Diagnostic 3 (distance at which the
# LOWESS-smoothed Pearson correlation crosses 0.5). Convert to Gaussian
# sigma via the half-width-at-half-max factor: a Gaussian with sigma =
# scale / sqrt(2 ln 2) has K(scale) = 0.5 exactly.
WSPD_DECORR_KM = 400.0
WVHT_DECORR_KM = 650.0
HWHM_FACTOR = np.sqrt(2.0 * np.log(2.0))
SIGMA_WSPD = WSPD_DECORR_KM / HWHM_FACTOR
SIGMA_WVHT = WVHT_DECORR_KM / HWHM_FACTOR
SIGMA_SHARED = np.sqrt(SIGMA_WSPD * SIGMA_WVHT)


def haversine_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Pairwise great-circle distance matrix in km; symmetric, zero diag."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    phi = np.radians(lat)
    lam = np.radians(lon)
    dphi = phi[:, None] - phi[None, :]
    dlam = lam[:, None] - lam[None, :]
    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi[:, None]) * np.cos(phi[None, :]) * np.sin(dlam / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def gaussian_adjacency(D: np.ndarray, sigma: float) -> np.ndarray:
    """K(d, sigma) = exp(-d^2 / (2 sigma^2)), diagonal forced to 0."""
    A = np.exp(-(D ** 2) / (2.0 * sigma ** 2))
    np.fill_diagonal(A, 0.0)
    return A


def uniform_adjacency(n: int) -> np.ndarray:
    A = np.full((n, n), 1.0 / (n - 1))
    np.fill_diagonal(A, 0.0)
    return A


def plot_kernel_calibration(pairs: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    panels = [
        ("pearson_wspd", "Wind speed (WSPD)", SIGMA_WSPD, axes[0]),
        ("pearson_wvht", "Wave height (WVHT)", SIGMA_WVHT, axes[1]),
    ]
    for corr_col, var_title, sigma, ax in panels:
        sub = pairs.dropna(subset=[corr_col])
        x = sub["distance_km"].to_numpy()
        y = sub[corr_col].to_numpy()
        grid = np.linspace(x.min(), x.max(), LOWESS_GRID_POINTS)
        smooth = lowess(y, x, frac=LOWESS_FRAC, xvals=grid)
        kernel_curve = np.exp(-(grid ** 2) / (2.0 * sigma ** 2))
        ax.scatter(x, y, s=14, alpha=0.3, color="C0", label="Pair observations")
        ax.plot(grid, smooth, color="C0", linewidth=2, label="Empirical LOWESS")
        ax.plot(
            grid,
            kernel_curve,
            color="C3",
            linewidth=2,
            linestyle="--",
            label=f"Gaussian kernel, sigma = {sigma:.0f} km",
        )
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Correlation / kernel weight")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"{var_title}: kernel calibration")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "Variable-specific Gaussian kernel calibration to empirical "
        "decorrelation curves (Diagnostic 3)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(output_path)
    plt.close(fig)


def plot_adjacency_heatmaps(
    adjacencies: dict[str, tuple[str, np.ndarray]],
    station_ids: list[str],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=150)
    panel_keys = ["wspd", "wvht", "shared", "uniform"]
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    n = len(station_ids)
    tick_positions = list(range(0, n, 4))
    tick_labels = [station_ids[i] for i in tick_positions]
    im = None
    for key, pos in zip(panel_keys, positions):
        title, matrix = adjacencies[key]
        ax = axes[pos]
        im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_title(title, fontsize=11)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=7, rotation=45, ha="right")
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels, fontsize=7)
    fig.suptitle(
        "Candidate adjacency matrices for the 27-station offshore "
        "Gulf network (Diagnostic 4)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.02, 0.93, 0.95])
    cax = fig.add_axes([0.94, 0.10, 0.02, 0.78])
    fig.colorbar(im, cax=cax, label="Adjacency weight")
    fig.savefig(output_path)
    plt.close(fig)


def summarize(
    D: np.ndarray,
    adjacencies: dict[str, tuple[str, np.ndarray]],
    station_ids: list[str],
) -> None:
    n = len(station_ids)
    off_diag_mask = ~np.eye(n, dtype=bool)
    d_off = D[off_diag_mask]
    label_w = 41
    sub_w = label_w - 4
    print()
    print("=" * 72)
    print(f"{'Stations included:':<{label_w}}{n}")
    print(
        f"{'Off-diagonal distance min/median/max:':<{label_w}}"
        f"{d_off.min():.0f} / {np.median(d_off):.0f} / {d_off.max():.0f} km"
    )
    for key in ["wspd", "wvht", "shared", "uniform"]:
        title, A = adjacencies[key]
        a_off = A[off_diag_mask]
        n_effective = int((a_off > 0.01).sum())
        print()
        print(f"  {title}:")
        print(f"    {'matrix sum:':<{sub_w}}{A.sum():.3f}")
        print(
            f"    {'off-diag min/max:':<{sub_w}}"
            f"{a_off.min():.4f} / {a_off.max():.4f}"
        )
        print(f"    {'effective edges (> 0.01):':<{sub_w}}{n_effective}")
    print()
    label = "Distance matrix:"
    print(f"{label:<{label_w}}{DISTANCE_PATH.relative_to(PROJECT_ROOT)}")
    for key, path in [
        ("wspd", ADJ_WSPD_PATH),
        ("wvht", ADJ_WVHT_PATH),
        ("shared", ADJ_SHARED_PATH),
        ("uniform", ADJ_UNIFORM_PATH),
    ]:
        label = f"Adjacency ({key}):"
        print(f"{label:<{label_w}}{path.relative_to(PROJECT_ROOT)}")
    for label, path in [
        ("Station order CSV:", ORDER_PATH),
        ("Kernel calibration figure:", KERNEL_FIG_PATH),
        ("Adjacency heatmaps figure:", ADJ_FIG_PATH),
    ]:
        print(f"{label:<{label_w}}{path.relative_to(PROJECT_ROOT)}")
    print("=" * 72)


def main() -> int:
    parquets = sorted(RAW_DIR.glob("*.parquet"))
    if not parquets:
        print(f"ERROR: no parquets in {RAW_DIR}.", file=sys.stderr)
        return 1
    if not STATIONS_PATH.exists():
        print(f"ERROR: {STATIONS_PATH} missing.", file=sys.stderr)
        return 1
    if not PAIRS_PATH.exists():
        print(
            f"ERROR: {PAIRS_PATH} missing (run Diagnostic 3 first).",
            file=sys.stderr,
        )
        return 1

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    stations = pd.read_csv(
        STATIONS_PATH, dtype={"station_id": str}
    ).set_index("station_id")
    station_ids = sorted(p.stem for p in parquets)
    missing = [sid for sid in station_ids if sid not in stations.index]
    if missing:
        print(
            f"ERROR: parquet stations missing lat/lon: {missing}",
            file=sys.stderr,
        )
        return 1

    print(
        f"Stations: {len(station_ids)} "
        f"(from {RAW_DIR.relative_to(PROJECT_ROOT)})"
    )
    lats = stations.loc[station_ids, "latitude"].to_numpy()
    lons = stations.loc[station_ids, "longitude"].to_numpy()

    print("Building distance matrix (vectorized haversine)...")
    D = haversine_matrix(lats, lons)

    print(
        f"Kernel sigmas: WSPD={SIGMA_WSPD:.1f} km, "
        f"WVHT={SIGMA_WVHT:.1f} km, shared={SIGMA_SHARED:.1f} km"
    )
    A_wspd = gaussian_adjacency(D, SIGMA_WSPD)
    A_wvht = gaussian_adjacency(D, SIGMA_WVHT)
    A_shared = gaussian_adjacency(D, SIGMA_SHARED)
    A_uniform = uniform_adjacency(len(station_ids))

    np.save(DISTANCE_PATH, D)
    np.save(ADJ_WSPD_PATH, A_wspd)
    np.save(ADJ_WVHT_PATH, A_wvht)
    np.save(ADJ_SHARED_PATH, A_shared)
    np.save(ADJ_UNIFORM_PATH, A_uniform)
    pd.DataFrame({"station_id": station_ids}).to_csv(ORDER_PATH, index=False)
    print(
        f"Saved 4 adjacency matrices + distance + order to "
        f"{PROCESSED_DIR.relative_to(PROJECT_ROOT)}"
    )

    pairs = pd.read_csv(
        PAIRS_PATH, dtype={"station_a": str, "station_b": str}
    )
    print("Rendering kernel calibration figure...")
    plot_kernel_calibration(pairs, KERNEL_FIG_PATH)
    print(f"Wrote {KERNEL_FIG_PATH.relative_to(PROJECT_ROOT)}")

    adjacencies = {
        "wspd": (f"WSPD-specific (sigma = {SIGMA_WSPD:.0f} km)", A_wspd),
        "wvht": (f"WVHT-specific (sigma = {SIGMA_WVHT:.0f} km)", A_wvht),
        "shared": (f"Shared (sigma = {SIGMA_SHARED:.0f} km)", A_shared),
        "uniform": (f"Uniform baseline (1/{len(station_ids) - 1})", A_uniform),
    }
    print("Rendering adjacency heatmaps figure...")
    plot_adjacency_heatmaps(adjacencies, station_ids, ADJ_FIG_PATH)
    print(f"Wrote {ADJ_FIG_PATH.relative_to(PROJECT_ROOT)}")

    summarize(D, adjacencies, station_ids)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
