"""
Step 4: Fine-tune and evaluate all conditions on both downstream tasks.

Olive oil:
  python scripts/04_finetune_evaluate.py --task olive_oil --config configs/finetune_oliveoil.yaml

Hemp seed:
  python scripts/04_finetune_evaluate.py --task hemp_seed --config configs/finetune_hemp.yaml

Results are written to results/{task}/transformer_results.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from msformer.evaluation.stats import compare_conditions, print_results_table
from msformer.training.finetune import Finetuner


def load_data(task: str, cfg: dict) -> tuple:
    if task == "olive_oil":
        from msformer.downstream.olive_oil import load_olive_oil_data
        data_dir = cfg["data"]["data_dir"]
        reduction = cfg["task"].get("reduction", "sum")
        target_dim = cfg["task"].get("ims_target_dim", 1000)

        if reduction == "both":
            Xs, ys, ids_list, groups_map = {}, {}, {}, {}
            for red in ("sum", "apex"):
                X, y, ids = load_olive_oil_data(data_dir, reduction=red, target_dim=target_dim)
                Xs[red], ys[red], ids_list[red], groups_map[red] = X, y, ids, None
            return Xs, ys, ids_list, groups_map
        else:
            X, y, ids = load_olive_oil_data(data_dir, reduction=reduction, target_dim=target_dim)
            return {reduction: X}, {reduction: y}, {reduction: ids}, {reduction: None}

    elif task == "hemp_seed":
        from msformer.downstream.hemp_seed import load_hemp_seed_data
        data_dir = cfg["data"]["data_dir"]
        X, y, ids, meta = load_hemp_seed_data(data_dir)
        if meta["preliminary"]:
            print("\n⚠  PRELIMINARY: n < 30 samples.  Using group LOSO-CV.  "
                  "Results are indicative only.\n")
            cfg["task"]["cv_strategy"] = "loso"
        bio_groups = meta.get("bio_groups")
        return {"gcms": X}, {"gcms": y}, {"gcms": ids}, {"gcms": bio_groups}

    else:
        raise ValueError(f"Unknown task: {task}")


def main(task: str, config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["output"]["results_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    Xs, ys, ids, groups_map = load_data(task, cfg)
    seeds = cfg["task"].get("cv_seeds", list(range(10)))

    all_results = {}  # condition → {reduction → {metric → scores}}

    for reduction_key, X in Xs.items():
        y = ys[reduction_key]
        print(f"\n{'='*60}")
        print(f"Task: {task}  |  Reduction/variant: {reduction_key}")
        print(f"{'='*60}")

        groups = groups_map.get(reduction_key)
        finetuner = Finetuner(cfg, X, y, groups=groups)
        results = finetuner.run_all_conditions(seeds=seeds)

        for cond, metrics in results.items():
            if cond not in all_results:
                all_results[cond] = {}
            all_results[cond][reduction_key] = metrics

        # Statistical comparisons for balanced_accuracy
        ba_results = {cond: metrics["balanced_accuracy"] for cond, metrics in results.items()}
        comparison_df = compare_conditions(ba_results)

        print(f"\n--- Pairwise comparisons ({reduction_key}, balanced_accuracy) ---")
        print(comparison_df.to_string(index=False))

        comparison_df.to_csv(
            output_dir / f"comparisons_{reduction_key}.csv", index=False
        )

    # Summary table
    summary = {
        cond: {
            rkey: np.mean(all_results[cond][rkey]["balanced_accuracy"])
            for rkey in all_results[cond]
        }
        for cond in all_results
    }
    print_results_table(
        {cond: {rk: all_results[cond][rk]["balanced_accuracy"] for rk in all_results[cond]}
         for cond in all_results}
    )

    with open(output_dir / "transformer_results.json", "w") as f:
        # Convert numpy floats to plain Python for JSON serialisation
        serialisable = {
            cond: {
                rkey: {metric: [float(v) for v in vals]
                       for metric, vals in metrics.items()}
                for rkey, metrics in rdict.items()
            }
            for cond, rdict in all_results.items()
        }
        json.dump(serialisable, f, indent=2)
    print(f"\nResults saved → {output_dir}/transformer_results.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["olive_oil", "hemp_seed"])
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.task, args.config)
