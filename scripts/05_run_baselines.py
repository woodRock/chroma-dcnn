"""
Step 5: Run published baselines (PLS-DA, RF, SVM) on both downstream tasks.

Olive oil:
  python scripts/05_run_baselines.py --task olive_oil --config configs/finetune_oliveoil.yaml

Hemp seed:
  python scripts/05_run_baselines.py --task hemp_seed --config configs/finetune_hemp.yaml

Results are written to results/{task}/baseline_results.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from msformer.evaluation.baselines import baseline_cv
from msformer.evaluation.stats import compare_conditions


def main(task: str, config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["output"]["results_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = cfg["task"].get("cv_seeds", list(range(10)))
    cv_strategy = cfg["task"].get("cv_strategy", "kfold")
    cv_folds = cfg["task"].get("cv_folds", 5)

    if task == "olive_oil":
        from msformer.downstream.olive_oil import (
            load_olive_oil_data,
            reproduce_gcims_plsda_baseline,
        )
        data_dir = cfg["data"]["data_dir"]
        reductions = ["sum", "apex"]

        for reduction in reductions:
            print(f"\n=== Olive oil ({reduction}) ===")
            X, y, _ = load_olive_oil_data(data_dir, reduction=reduction)

            baseline_results = {}
            for model_name in ["plsda", "rf", "svm"]:
                print(f"  Running {model_name} …")
                baseline_results[model_name] = baseline_cv(
                    X, y, model_name, seeds=seeds,
                    cv_strategy=cv_strategy, cv_folds=cv_folds,
                )
                ba = baseline_results[model_name]["balanced_accuracy"]
                print(f"    BA: {np.mean(ba):.3f} ± {np.std(ba):.3f}")

            # Also run the gc_ims_tools-specific PLS-DA reproduction
            try:
                gcims_plsda = reproduce_gcims_plsda_baseline(Path(data_dir))
                baseline_results["plsda_gcims_tools"] = gcims_plsda
            except ImportError as e:
                print(f"  gc_ims_tools not installed; skipping: {e}")

            _save_and_compare(baseline_results, output_dir, f"baseline_{reduction}")

    elif task == "hemp_seed":
        from msformer.downstream.hemp_seed import load_hemp_seed_data
        data_dir = cfg["data"]["data_dir"]
        X, y, _, meta = load_hemp_seed_data(data_dir)
        if meta["preliminary"]:
            cv_strategy = "loso"
            print("⚠  n < 30: using LOSO-CV for baselines")

        baseline_results = {}
        for model_name in ["plsda", "rf", "svm"]:
            print(f"Running {model_name} …")
            baseline_results[model_name] = baseline_cv(
                X, y, model_name, seeds=seeds,
                cv_strategy=cv_strategy, cv_folds=cv_folds,
            )
            ba = baseline_results[model_name]["balanced_accuracy"]
            print(f"  BA: {np.mean(ba):.3f} ± {np.std(ba):.3f}")

        _save_and_compare(baseline_results, output_dir, "baseline_gcms")

    else:
        raise ValueError(f"Unknown task: {task}")


def _save_and_compare(baseline_results: dict, output_dir: Path, name: str) -> None:
    ba_results = {k: v["balanced_accuracy"] for k, v in baseline_results.items()}
    comparison_df = compare_conditions(ba_results)
    print(f"\n--- Baseline comparisons ({name}) ---")
    print(comparison_df.to_string(index=False))

    comparison_df.to_csv(output_dir / f"{name}_comparisons.csv", index=False)
    with open(output_dir / f"{name}_results.json", "w") as f:
        serialisable = {
            k: {m: [float(v) for v in vals] for m, vals in metrics.items()}
            for k, metrics in baseline_results.items()
        }
        json.dump(serialisable, f, indent=2)
    print(f"Saved → {output_dir}/{name}_results.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["olive_oil", "hemp_seed"])
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.task, args.config)
