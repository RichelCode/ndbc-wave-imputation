"""
Diagnostic 6: Recompute the 27-station adjacency matrices using the
exponential kernel selected by Diagnostic 5, and render a side-by-side
comparison against the Gaussian adjacencies from Diagnostic 4.

Inputs
------
- data/processed/distance_matrix.npy        (canonical 27x27 haversine
  distances from D4; loaded, never recomputed).
- data/processed/adjacency_station_order.csv (row/column order).
- data/processed/adjacency_{wspd,wvht,shared,uniform}.npy (Gaussian
  adjacencies from D4; loaded for the comparison figure only).
- data/processed/kernel_fit_results.csv     (D5 output; verified at
  startup to confirm the hardcoded ell constants match what D5 fitted).

Outputs
-------
- data/processed/adjacency_wspd_exp.npy    : Exponential, ell = 459.1 km.
- data/processed/adjacency_wvht_exp.npy    : Exponential, ell = 714.8 km.
- data/processed/adjacency_shared_exp.npy  : Exponential, ell ~ 572.7 km.
- data/processed/adjacency_uniform_exp.npy : 1/26 off-diagonal baseline,
  bit-for-bit identical to adjacency_uniform.npy. Duplicated under the
  _exp name so downstream code can append "_exp" to any matrix name
  to load the winning-kernel version.
- figures/adjacency_kernel_comparison.png  : 2x4 heatmap grid (top row
  Gaussian from D4, bottom row exponential, columns WSPD/WVHT/shared/
  uniform), shared color scale, single colorbar.

Methodology
-----------
1. Kernel: K(d, ell) = exp(-d / ell). Diagonal forced to 0.
2. Length scales fitted in D5:
     ELL_WSPD   = 459.1 km
     ELL_WVHT   = 714.8 km
     ELL_SHARED = sqrt(ELL_WSPD * ELL_WVHT) ~ 572.7 km
3. Distance matrix and station order are loaded from disk, never
   recomputed. Any drift in those inputs would break cell-by-cell
   comparison with the Gaussian matrices.
4. The uniform baseline is rebuilt rather than copied so the math
   is auditable, but it is bit-for-bit identical to D4's version.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "stdmet"
FIGURES_DIR = PROJECT_ROOT / "figures"

DISTANCE_PATH = PROCESSED_DIR / "distance_matrix.npy"
ORDER_PATH = PROCESSED_DIR / "adjacency_station_order.csv"
KERNEL_FIT_PATH = PROCESSED_DIR / "kernel_fit_results.csv"
STATIONS_PATH = PROCESSED_DIR / "candidate_stations.csv"

ADJ_WSPD_GAUSS_PATH = PROCESSED_DIR / "adjacency_wspd.npy"
ADJ_WVHT_GAUSS_PATH = PROCESSED_DIR / "adjacency_wvht.npy"
ADJ_SHARED_GAUSS_PATH = PROCESSED_DIR / "adjacency_shared.npy"
ADJ_UNIFORM_GAUSS_PATH = PROCESSED_DIR / "adjacency_uniform.npy"

ADJ_WSPD_EXP_PATH = PROCESSED_DIR / "adjacency_wspd_exp.npy"
ADJ_WVHT_EXP_PATH = PROCESSED_DIR / "adjacency_wvht_exp.npy"
ADJ_SHARED_EXP_PATH = PROCESSED_DIR / "adjacency_shared_exp.npy"
ADJ_UNIFORM_EXP_PATH = PROCESSED_DIR / "adjacency_uniform_exp.npy"

COMPARISON_FIG_PATH = FIGURES_DIR / "adjacency_kernel_comparison.png"

# Fitted exponential length scales from Diagnostic 5
# (verified at startup against kernel_fit_results.csv).
ELL_WSPD = 459.1
ELL_WVHT = 714.8
ELL_SHARED = float(np.sqrt(ELL_WSPD * ELL_WVHT))

# D4 Gaussian sigmas (used only as figure-title annotations).
HWHM_FACTOR = float(np.sqrt(2.0 * np.log(2.0)))
SIGMA_WSPD = 400.0 / HWHM_FACTOR
SIGMA_WVHT = 650.0 / HWHM_FACTOR
SIGMA_SHARED = float(np.sqrt(SIGMA_WSPD * SIGMA_WVHT))

ELL_TOLERANCE_KM = 0.5


def exp_adjacency(D: np.ndarray, ell: float) -> np.ndarray:
    """K(d, ell) = exp(-d / ell), diagonal forced to 0."""
    A = np.exp(-D / ell)
    np.fill_diagonal(A, 0.0)
    return A


def uniform_adjacency(n: int) -> np.ndarray:
    A = np.full((n, n), 1.0 / (n - 1))
    np.fill_diagonal(A, 0.0)
    return A


def verify_ell_against_d5(expected: dict[str, float]) -> bool:
    """Confirm hardcoded ell constants still match D5 fits within tolerance.

    `expected` maps variable name ("WSPD" / "WVHT") to expected ell_km.
    Returns True if all checks pass; prints diagnostic info either way.
    """
    fits = pd.read_csv(KERNEL_FIT_PATH)
    ok = True
    for var, expected_ell in expected.items():
        mask = (fits["kernel"] == "Exponential") & (fits["variable"] == var)
        rows = fits[mask]
        if len(rows) != 1:
            print(
                f"ERROR: expected exactly 1 Exponential/{var} row in "
                f"{KERNEL_FIT_PATH}, got {len(rows)}.",
                file=sys.stderr,
            )
            ok = False
            continue
        ell_csv = float(rows["ell_km"].iloc[0])
        if not np.isfinite(ell_csv):
            print(
                f"ERROR: D5 Exponential/{var} ell is non-finite ({ell_csv}); "
                f"re-run Diagnostic 5 before this script.",
                file=sys.stderr,
            )
            ok = False
            continue
        if abs(ell_csv - expected_ell) > ELL_TOLERANCE_KM:
            print(
                f"ERROR: D5 Exponential/{var} ell drift detected. "
                f"This script hardcodes {expected_ell:.3f} km, but "
                f"kernel_fit_results.csv shows {ell_csv:.3f} km "
                f"(diff {abs(ell_csv - expected_ell):.3f} km, "
                f"tolerance {ELL_TOLERANCE_KM}). Update the constants "
                f"or re-run D5 to match.",
                file=sys.stderr,
            )
            ok = False
        else:
            print(
                f"  Exponential/{var}: D5 says ell={ell_csv:.3f} km, "
                f"script uses {expected_ell:.3f} km (within tolerance)."
            )
    return ok


def plot_comparison(
    gauss: dict[str, np.ndarray],
    exp: dict[str, np.ndarray],
    station_ids: list[str],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(20, 11), dpi=150)
    n = len(station_ids)
    tick_positions = list(range(0, n, 4))
    tick_labels = [station_ids[i] for i in tick_positions]

    panel_keys = ["wspd", "wvht", "shared", "uniform"]
    gauss_titles = {
        "wspd": f"Gaussian WSPD (sigma={SIGMA_WSPD:.0f})",
        "wvht": f"Gaussian WVHT (sigma={SIGMA_WVHT:.0f})",
        "shared": f"Gaussian shared (sigma={SIGMA_SHARED:.0f})",
        "uniform": f"Uniform baseline (1/{n - 1})",
    }
    exp_titles = {
        "wspd": f"Exp WSPD (ell={ELL_WSPD:.0f})",
        "wvht": f"Exp WVHT (ell={ELL_WVHT:.0f})",
        "shared": f"Exp shared (ell={ELL_SHARED:.0f})",
        "uniform": f"Uniform baseline (1/{n - 1})",
    }

    im = None
    for col_idx, key in enumerate(panel_keys):
        for row_idx, (mat_dict, title_dict) in enumerate(
            [(gauss, gauss_titles), (exp, exp_titles)]
        ):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(
                mat_dict[key], cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto"
            )
            ax.set_title(title_dict[key], fontsize=10)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, fontsize=7, rotation=45, ha="right")
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels, fontsize=7)

    fig.suptitle(
        "Adjacency matrices: Gaussian (top, rejected) vs. Exponential "
        "(bottom, adopted)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.02, 0.93, 0.93])
    cax = fig.add_axes([0.94, 0.08, 0.015, 0.80])
    fig.colorbar(im, cax=cax, label="Adjacency weight")
    fig.savefig(output_path)
    plt.close(fig)


def summarize(
    exp_matrices: dict[str, tuple[str, np.ndarray]], n: int
) -> None:
    label_w = 41
    sub_w = label_w - 4
    off_diag_mask = ~np.eye(n, dtype=bool)
    print()
    print("=" * 72)
    print(f"{'Stations included:':<{label_w}}{n}")
    print(
        f"{'Exponential ell (WSPD / WVHT / shared):':<{label_w}}"
        f"{ELL_WSPD:.1f} / {ELL_WVHT:.1f} / {ELL_SHARED:.1f} km"
    )
    for key in ["wspd", "wvht", "shared", "uniform"]:
        title, A = exp_matrices[key]
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
    for label, path in [
        ("Adjacency WSPD (exp):", ADJ_WSPD_EXP_PATH),
        ("Adjacency WVHT (exp):", ADJ_WVHT_EXP_PATH),
        ("Adjacency shared (exp):", ADJ_SHARED_EXP_PATH),
        ("Adjacency uniform (exp):", ADJ_UNIFORM_EXP_PATH),
        ("Comparison figure:", COMPARISON_FIG_PATH),
    ]:
        print(f"{label:<{label_w}}{path.relative_to(PROJECT_ROOT)}")
    print("=" * 72)


def main() -> int:
    parquets = list(RAW_DIR.glob("*.parquet"))
    if not parquets or not STATIONS_PATH.exists():
        print(
            f"ERROR: upstream pipeline state appears incomplete; "
            f"expected at least one parquet in {RAW_DIR} and "
            f"{STATIONS_PATH} to exist.",
            file=sys.stderr,
        )
        return 1

    required = [
        DISTANCE_PATH,
        ORDER_PATH,
        KERNEL_FIT_PATH,
        ADJ_WSPD_GAUSS_PATH,
        ADJ_WVHT_GAUSS_PATH,
        ADJ_SHARED_GAUSS_PATH,
        ADJ_UNIFORM_GAUSS_PATH,
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        for p in missing:
            print(f"ERROR: required input missing: {p}", file=sys.stderr)
        return 1

    print("Verifying D5 ell constants against kernel_fit_results.csv...")
    if not verify_ell_against_d5({"WSPD": ELL_WSPD, "WVHT": ELL_WVHT}):
        return 1

    D = np.load(DISTANCE_PATH)
    order_df = pd.read_csv(ORDER_PATH, dtype={"station_id": str})
    station_ids = order_df["station_id"].tolist()
    if len(station_ids) != D.shape[0] or D.shape[0] != D.shape[1]:
        print(
            f"ERROR: station_order count {len(station_ids)} disagrees with "
            f"distance matrix shape {D.shape}.",
            file=sys.stderr,
        )
        return 1
    n = len(station_ids)
    print(f"Loaded distance matrix {D.shape} and {n} station IDs.")

    print(
        f"Building exponential adjacencies: "
        f"WSPD ell={ELL_WSPD:.1f}, WVHT ell={ELL_WVHT:.1f}, "
        f"shared ell={ELL_SHARED:.1f} km."
    )
    A_wspd_exp = exp_adjacency(D, ELL_WSPD)
    A_wvht_exp = exp_adjacency(D, ELL_WVHT)
    A_shared_exp = exp_adjacency(D, ELL_SHARED)
    A_uniform_exp = uniform_adjacency(n)

    np.save(ADJ_WSPD_EXP_PATH, A_wspd_exp)
    np.save(ADJ_WVHT_EXP_PATH, A_wvht_exp)
    np.save(ADJ_SHARED_EXP_PATH, A_shared_exp)
    np.save(ADJ_UNIFORM_EXP_PATH, A_uniform_exp)
    print(
        f"Saved 4 exponential adjacency matrices to "
        f"{PROCESSED_DIR.relative_to(PROJECT_ROOT)}"
    )

    gauss = {
        "wspd": np.load(ADJ_WSPD_GAUSS_PATH),
        "wvht": np.load(ADJ_WVHT_GAUSS_PATH),
        "shared": np.load(ADJ_SHARED_GAUSS_PATH),
        "uniform": np.load(ADJ_UNIFORM_GAUSS_PATH),
    }
    exp_mats = {
        "wspd": A_wspd_exp,
        "wvht": A_wvht_exp,
        "shared": A_shared_exp,
        "uniform": A_uniform_exp,
    }
    print("Rendering Gaussian-vs-exponential comparison figure...")
    plot_comparison(gauss, exp_mats, station_ids, COMPARISON_FIG_PATH)
    print(f"Wrote {COMPARISON_FIG_PATH.relative_to(PROJECT_ROOT)}")

    exp_with_titles = {
        "wspd": (f"Exp WSPD (ell = {ELL_WSPD:.0f} km)", A_wspd_exp),
        "wvht": (f"Exp WVHT (ell = {ELL_WVHT:.0f} km)", A_wvht_exp),
        "shared": (f"Exp shared (ell = {ELL_SHARED:.0f} km)", A_shared_exp),
        "uniform": (f"Uniform baseline (1/{n - 1})", A_uniform_exp),
    }
    summarize(exp_with_titles, n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
