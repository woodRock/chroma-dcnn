"""
Statistical comparison of classifier conditions.

Protocol (matches Wood 2025 / DepthSim / Sign-RP papers):
  - Mann-Whitney U test (two-sided, non-parametric — appropriate for small k-fold samples)
  - Bonferroni correction across all pairwise comparisons
  - Rank-biserial correlation as effect size r = 1 - 2U/(n1·n2)
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


def rank_biserial_r(u_stat: float, n1: int, n2: int) -> float:
    """Effect size: r = 1 - 2U/(n1·n2).  Range [-1, 1]."""
    return 1.0 - (2.0 * u_stat) / (n1 * n2)


def compare_conditions(
    results: dict[str, list[float]],
    metric: str = "balanced_accuracy",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Pairwise Mann-Whitney U tests with Bonferroni correction.

    Parameters
    ----------
    results : dict mapping condition_name → list of fold scores
    metric  : key within each result dict (unused here — pass pre-extracted lists)
    alpha   : family-wise error rate

    Returns a DataFrame with columns:
      condition_a, condition_b, mean_a, mean_b, U, p_raw, p_corrected, effect_r, significant
    """
    condition_names = list(results.keys())
    pairs = list(combinations(condition_names, 2))
    n_comparisons = len(pairs)

    rows = []
    for a, b in pairs:
        vals_a = np.array(results[a])
        vals_b = np.array(results[b])
        u_stat, p_raw = mannwhitneyu(vals_a, vals_b, alternative="two-sided")
        p_corr = min(p_raw * n_comparisons, 1.0)
        r = rank_biserial_r(u_stat, len(vals_a), len(vals_b))
        rows.append({
            "condition_a": a,
            "condition_b": b,
            "mean_a": vals_a.mean(),
            "std_a": vals_a.std(),
            "mean_b": vals_b.mean(),
            "std_b": vals_b.std(),
            "U": u_stat,
            "p_raw": p_raw,
            "p_corrected": p_corr,
            "effect_r": r,
            "significant": p_corr < alpha,
        })

    return pd.DataFrame(rows).sort_values("p_corrected")


def bonferroni_correct(p_values: list[float]) -> list[float]:
    n = len(p_values)
    return [min(p * n, 1.0) for p in p_values]


def results_table(
    all_results: dict[str, dict[str, list[float]]],
    metric: str = "balanced_accuracy",
) -> pd.DataFrame:
    """
    Build a summary table: condition × task with mean ± std.

    all_results : {condition_name: {task_name: [fold_scores]}}
    """
    rows = []
    for cond, task_results in all_results.items():
        row = {"condition": cond}
        for task, scores in task_results.items():
            arr = np.array(scores)
            row[f"{task}_mean"] = arr.mean()
            row[f"{task}_std"] = arr.std()
            row[f"{task}_n"] = len(arr)
        rows.append(row)
    return pd.DataFrame(rows).set_index("condition")


def print_results_table(
    all_results: dict[str, dict[str, list[float]]],
    metric: str = "balanced_accuracy",
) -> None:
    df = results_table(all_results, metric)
    print(f"\n=== Results table ({metric}) ===")
    for col in df.columns:
        if col.endswith("_mean"):
            task = col.removesuffix("_mean")
            std_col = f"{task}_std"
            n_col = f"{task}_n"
            df[task] = df.apply(
                lambda r: f"{r[col]:.3f} ± {r[std_col]:.3f} (n={int(r[n_col])})", axis=1
            )
    display_cols = [c for c in df.columns if not c.endswith("_std") and not c.endswith("_n")]
    print(df[display_cols].to_string())
