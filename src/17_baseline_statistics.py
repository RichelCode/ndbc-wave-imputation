"""
Diagnostic 17 — Baseline Statistics: Significance Testing vs GRIN
==================================================================

Paired t-tests of each baseline's per-mask MAE against GRIN (D10) across
the 15 D8 evaluation masks. Two separate Bonferroni thresholds:

    method-level:  0.05 / 5  = 0.01     (5 overall comparisons)
    family-level:  0.05 / 15 = 0.00333  (5 methods × 3 families)

Per baseline:
    overall            : paired t-test across all 15 masks
    by_family/mcar     : 9 masks (rates 10%, 25%, 50% × 3 seeds)
    by_family/blockmcar: 3 masks (rate 25% × 3 seeds)
    by_family/mnar     : 3 masks (rate 25% × 3 seeds)

For each test we report mean_diff (baseline − GRIN, so positive = baseline
worse), t_stat, p_value, 95% paired bootstrap CI on mean_diff (1000
resamples, seed 42). Overall entries get a significant_method_bonferroni
flag (threshold 0.01); per-family entries get significant_family_bonferroni
(threshold 0.00333). Per-family entries with n_masks=3 get a power_note
field warning that "not significant" should not be read as "no difference".

Handles schema difference between baseline JSONs:
    single-seed (D10, D13-D15): per_mask_results[mid]["mae"]
    aggregated  (D16):          per_mask_results[mid]["mae_mean"]

Gracefully skips baselines whose results JSON is missing (e.g. CSDI
training still in progress); skipped names appear in the output JSON's
skipped_methods field.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
GRIN_PATH = DATA_DIR / "d10_results.json"
OUTPUT_PATH = DATA_DIR / "baseline_significance.json"

BASELINES = [
    ("linear", DATA_DIR / "baseline_linear_results.json"),
    ("locf", DATA_DIR / "baseline_locf_results.json"),
    ("climatology", DATA_DIR / "baseline_climatology_results.json"),
    ("knn", DATA_DIR / "baseline_knn_results.json"),
    ("csdi", DATA_DIR / "baseline_csdi_results.json"),
]

N_METHODS = len(BASELINES)
N_FAMILIES = 3
BONFERRONI_METHOD = 0.05 / N_METHODS            # 0.01
BONFERRONI_FAMILY = 0.05 / (N_METHODS * N_FAMILIES)  # 0.00333

FAMILIES = ("mcar", "blockmcar", "mnar")
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42


def get_mae(entry: dict) -> float:
    """Extract MAE from a per-mask entry; handles both schemas."""
    if "mae" in entry:
        return float(entry["mae"])
    if "mae_mean" in entry:
        return float(entry["mae_mean"])
    raise KeyError(f"no mae or mae_mean in entry; keys = {list(entry.keys())}")


def paired_bootstrap_ci(
    diffs: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float]:
    """95% bootstrap CI on the mean of paired differences."""
    n = len(diffs)
    if n < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = diffs[idx].mean()
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def run_paired_test(
    baseline_maes: np.ndarray, grin_maes: np.ndarray
) -> dict:
    """Paired t-test + bootstrap CI; caller adds significance flags."""
    n = len(baseline_maes)
    if n < 2:
        return {
            "n_masks": n,
            "mean_diff": float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    diffs = baseline_maes - grin_maes
    result = stats.ttest_rel(baseline_maes, grin_maes)
    p_value = float(result.pvalue)
    ci_low, ci_high = paired_bootstrap_ci(diffs)
    return {
        "n_masks": n,
        "mean_diff": float(diffs.mean()),
        "t_stat": float(result.statistic),
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def filter_masks_by_family(mask_ids: list[str], family: str) -> list[str]:
    return [mid for mid in mask_ids if mid.startswith(family + "_")]


def build_baseline_entry(
    baseline_per_mask: dict, grin_per_mask: dict
) -> dict:
    """Run overall + per-family paired tests for one baseline."""
    common_ids = sorted(set(baseline_per_mask) & set(grin_per_mask))
    baseline_maes = np.array([get_mae(baseline_per_mask[mid]) for mid in common_ids])
    grin_maes = np.array([get_mae(grin_per_mask[mid]) for mid in common_ids])

    overall = run_paired_test(baseline_maes, grin_maes)
    overall["significant_method_bonferroni"] = bool(
        overall["p_value"] < BONFERRONI_METHOD
    )

    by_family: dict = {}
    for family in FAMILIES:
        fam_ids = filter_masks_by_family(common_ids, family)
        fam_b = np.array([get_mae(baseline_per_mask[mid]) for mid in fam_ids])
        fam_g = np.array([get_mae(grin_per_mask[mid]) for mid in fam_ids])
        fam_entry = run_paired_test(fam_b, fam_g)
        fam_entry["significant_family_bonferroni"] = bool(
            fam_entry["p_value"] < BONFERRONI_FAMILY
        )
        if fam_entry["n_masks"] == 3:
            fam_entry["power_note"] = "n=3 observations; limited statistical power"
        by_family[family] = fam_entry

    out = {
        "n_masks_grin": len(grin_per_mask),
        "n_masks_baseline": len(baseline_per_mask),
        "n_masks_used": len(common_ids),
        "overall": overall,
        "by_family": by_family,
    }
    if len(grin_per_mask) != len(baseline_per_mask) or len(common_ids) != len(grin_per_mask):
        out["mask_mismatch"] = True
    return out


def main() -> int:
    if not GRIN_PATH.exists():
        print(f"ERROR: GRIN reference missing: {GRIN_PATH}", file=sys.stderr)
        return 1
    with open(GRIN_PATH) as f:
        grin = json.load(f)
    grin_per_mask = grin["per_mask_results"]

    by_method: dict = {}
    skipped: list = []
    for name, path in BASELINES:
        if not path.exists():
            print(f"  [{name}] SKIPPED: {path.name} not found")
            skipped.append(name)
            continue
        with open(path) as f:
            baseline = json.load(f)
        by_method[name] = build_baseline_entry(baseline["per_mask_results"], grin_per_mask)

    mismatch_baselines = [n for n, e in by_method.items() if e.get("mask_mismatch")]
    notes_parts = [
        "bonferroni_method (0.01) corresponds to the 5 overall method comparisons.",
        "bonferroni_family (0.00333) corresponds to the 5 methods x 3 families = 15 family tests.",
        "Overall entries carry significant_method_bonferroni; per-family entries carry significant_family_bonferroni.",
        "Per-family entries with n_masks=3 carry a power_note flagging limited statistical power.",
    ]
    if mismatch_baselines:
        notes_parts.append(
            f"WARNING: mask count mismatch in {mismatch_baselines}; "
            f"see per-baseline n_masks_* fields."
        )

    output = {
        "comparison_basis": grin.get("run_id", "d10_grin_wspd_exp"),
        "n_methods_compared": len(by_method),
        "n_methods_skipped": len(skipped),
        "skipped_methods": skipped,
        "bonferroni_method": BONFERRONI_METHOD,
        "bonferroni_family": BONFERRONI_FAMILY,
        "n_masks_overall": len(grin_per_mask),
        "bootstrap_n": N_BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "notes": " ".join(notes_parts),
        "by_method": by_method,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {OUTPUT_PATH}")

    print()
    print("=" * 86)
    print(
        f"{'Method':<14} {'Scope':<12} {'Δ MAE':>9} {'CI low':>9} {'CI high':>9} "
        f"{'t':>7} {'p':>10} {'sig?':>6}"
    )
    print("=" * 86)
    for name, entry in by_method.items():
        rows = [("overall", entry["overall"], "significant_method_bonferroni")] + [
            (f"  {f}", entry["by_family"][f], "significant_family_bonferroni")
            for f in FAMILIES
        ]
        for scope, s, flag_key in rows:
            sig = "***" if s.get(flag_key) else ""
            print(
                f"{name:<14} {scope:<12} {s['mean_diff']:>9.4f} "
                f"{s['ci_low']:>9.4f} {s['ci_high']:>9.4f} "
                f"{s['t_stat']:>7.2f} {s['p_value']:>10.4e} "
                f"{sig:>6}"
            )
    print("=" * 86)
    print(f"Bonferroni thresholds: method={BONFERRONI_METHOD:.4f}, family={BONFERRONI_FAMILY:.5f}")
    if skipped:
        print(f"Skipped (no results JSON): {', '.join(skipped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
