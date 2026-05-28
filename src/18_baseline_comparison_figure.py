"""
Diagnostic 18 — Baseline Comparison Figure
===========================================

Publication-grade grouped bar chart comparing per-method WSPD MAE across
the five mask family groupings (MCAR-10%, MCAR-25%, MCAR-50%, Block MCAR,
MNAR). This is the headline visual for Stage 1.

Methods plotted: GRIN (D10 reference) + the 5 baselines from D13-D16.
Any baseline whose results JSON is missing is silently dropped from the
figure (e.g. CSDI before training completes); the included set is logged
to stdout. Re-run after CSDI lands to regenerate with the full set.

Bar height = mean MAE across the 3 mask seeds in each family group.
Error bar = std across those 3 seeds. For single-run baselines this is
pure mask-seed sampling variance; for CSDI (seed-aggregated schema) it
is variance of the per-mask mae_mean across the 3 mask seeds.

Y-axis log scale (climatology ~5x GRIN; linear scale compresses signal).
Reference line at MAE = 5.54 m/s (the zeros-baseline floor from D12).

Outputs:
    figures/baseline_comparison.png            (300 dpi)
    figures/baseline_comparison.pdf            (vector)
    figures/baseline_comparison_caption.txt    (single-paragraph caption)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"

METHODS = [
    ("GRIN", DATA_DIR / "d10_results.json"),
    ("linear", DATA_DIR / "baseline_linear_results.json"),
    ("LOCF", DATA_DIR / "baseline_locf_results.json"),
    ("climatology", DATA_DIR / "baseline_climatology_results.json"),
    ("k-NN", DATA_DIR / "baseline_knn_results.json"),
    ("CSDI", DATA_DIR / "baseline_csdi_results.json"),
]

FAMILY_GROUPS = [
    ("MCAR-10%", "mcar_r10"),
    ("MCAR-25%", "mcar_r25"),
    ("MCAR-50%", "mcar_r50"),
    ("Block MCAR", "blockmcar_r25"),
    ("MNAR", "mnar_r25"),
]

ZEROS_FLOOR = 5.54  # mean|truth| from D12's zeros baseline smoke test

METHOD_COLORS = {
    "GRIN":        "#1f77b4",            # saturated blue, the hero method
    "linear":      plt.cm.viridis(0.15),  # dark purple
    "LOCF":        plt.cm.viridis(0.35),  # blue-purple
    "climatology": plt.cm.viridis(0.55),  # teal-green
    "k-NN":        plt.cm.viridis(0.75),  # yellow-green
    "CSDI":        plt.cm.viridis(0.95),  # yellow
}

PNG_PATH = FIGURES_DIR / "baseline_comparison.png"
PDF_PATH = FIGURES_DIR / "baseline_comparison.pdf"
CAPTION_PATH = FIGURES_DIR / "baseline_comparison_caption.txt"


def get_mae(entry: dict) -> float:
    """Extract MAE from a per-mask entry; handles both schemas."""
    if "mae" in entry:
        return float(entry["mae"])
    if "mae_mean" in entry:
        return float(entry["mae_mean"])
    raise KeyError(f"no mae or mae_mean in entry; keys = {list(entry.keys())}")


def collect_family_stats(per_mask: dict, family_prefix: str) -> tuple[float, float]:
    """Mean and std of MAE for mask_ids starting with prefix + '_'."""
    maes = [
        get_mae(entry)
        for mid, entry in per_mask.items()
        if mid.startswith(family_prefix + "_")
    ]
    if not maes:
        return float("nan"), float("nan")
    mean = float(np.mean(maes))
    std = float(np.std(maes, ddof=1)) if len(maes) > 1 else 0.0
    return mean, std


def main() -> int:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    loaded = []
    skipped = []
    for name, path in METHODS:
        if not path.exists():
            print(f"  [{name}] SKIPPED: {path.name} not found")
            skipped.append(name)
            continue
        with open(path) as f:
            loaded.append((name, json.load(f)["per_mask_results"]))

    if not loaded:
        print("ERROR: no results JSONs found", file=sys.stderr)
        return 1
    print(f"Included methods: {[n for n, _ in loaded]}")
    if skipped:
        print(f"Skipped (missing): {skipped}")

    n_methods = len(loaded)
    n_groups = len(FAMILY_GROUPS)
    means = np.full((n_methods, n_groups), np.nan)
    stds = np.full((n_methods, n_groups), np.nan)
    for i, (_, per_mask) in enumerate(loaded):
        for j, (_, prefix) in enumerate(FAMILY_GROUPS):
            means[i, j], stds[i, j] = collect_family_stats(per_mask, prefix)

    fig, ax = plt.subplots(figsize=(10.5, 5), dpi=300)
    x = np.arange(n_groups)
    bar_width = 0.8 / n_methods

    for i, (name, _) in enumerate(loaded):
        offset = (i - (n_methods - 1) / 2) * bar_width
        ax.bar(
            x + offset, means[i], bar_width,
            yerr=stds[i], capsize=2,
            label=name, color=METHOD_COLORS[name],
            edgecolor="white", linewidth=0.5,
            error_kw={"linewidth": 0.8},
        )

    ax.axhline(
        ZEROS_FLOOR, color="gray", linestyle="--", linewidth=1.0, alpha=0.7,
        label=f"naive zero-prediction floor ({ZEROS_FLOOR:.2f} m/s)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in FAMILY_GROUPS])
    ax.set_ylabel("MAE (m/s, log scale)")
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.3, top=7)
    ax.set_yticks([0.4, 0.6, 1.0, 2.0, 3.0, 5.0])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_title(
        "Single-variable WSPD imputation: MAE comparison across mask "
        "families and methods\n(NDBC Gulf 2010-2023)",
        fontsize=10,
    )
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {PNG_PATH}")
    print(f"Wrote {PDF_PATH}")

    method_list = ", ".join(n for n, _ in loaded)
    caption = (
        f"Mean absolute error (m/s, log scale) of single-variable WSPD "
        f"imputation on the 2022-2023 test split of 27 offshore NDBC Gulf of "
        f"Mexico buoys. Bars show mean and ±1 standard deviation across the "
        f"three seeds within each mask family. Methods compared: {method_list}. "
        f"Mask families: MCAR at three rates (10%, 25%, 50%), Block MCAR "
        f"(72-hour blocks, 25% rate), and Empirical MNAR (25% rate, 80% of "
        f"hidden cells concentrated in storm windows). The dashed line at "
        f"MAE = {ZEROS_FLOOR:.2f} m/s marks the naive zero-prediction floor "
        f"(mean|truth| on the test split). Key empirical observation: the "
        f"optimal method depends on the missingness scope. Temporal "
        f"interpolation methods (linear, LOCF) dominate when hidden cells "
        f"have nearby observed neighbors in time (MCAR); the graph-aware GRIN "
        f"dominates on Block MCAR (72-hour gaps) where temporal neighbors are "
        f"unavailable but spatial ones remain informative. No single method "
        f"wins across all families; all methods sit between the climatology "
        f"baseline and the zeros floor."
    )
    with open(CAPTION_PATH, "w") as f:
        f.write(caption + "\n")
    print(f"Wrote {CAPTION_PATH}")
    print()
    print("Caption:")
    print(caption)
    return 0


if __name__ == "__main__":
    sys.exit(main())
