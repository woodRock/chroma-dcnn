"""
Step 4: Fine-tune and evaluate downstream tasks.

Fish oil — ChromatogramCNN (1D CNN on uniform RT bins):
  python scripts/04_finetune_evaluate.py --task fish_oil_chroma --config configs/finetune_fish_oil_chroma.yaml

Results are written to results/fish_oil/chroma_results.json

NOTE: transformer-based conditions (olive_oil, hemp_seed, fish_oil, fish_oil_scans)
have been temporarily removed while the transformer architecture is being improved.
They will be restored once the updated transformer is ready.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from msformer.evaluation.stats import compare_conditions, print_results_table


def main(task: str, config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["output"]["results_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = cfg["task"].get("cv_seeds", list(range(10)))

    if task == "fish_oil_chroma":
        from msformer.downstream.fish_oil import load_fish_oil_chroma_paths
        from msformer.training.finetune_chroma import ChromaFinetuner
        npz_paths, y = load_fish_oil_chroma_paths(cfg["data"]["data_dir"])
        finetuner = ChromaFinetuner(cfg, npz_paths, y)
        results = finetuner.run_all_conditions(seeds=seeds)
        ba_results = {cond: metrics["balanced_accuracy"] for cond, metrics in results.items()}
        comparison_df = compare_conditions(ba_results)
        print("\n--- Pairwise comparisons (balanced_accuracy) ---")
        print(comparison_df.to_string(index=False))
        comparison_df.to_csv(output_dir / "comparisons_chroma.csv", index=False)
        print_results_table({cond: {"chroma": r["balanced_accuracy"]} for cond, r in results.items()})
        with open(output_dir / "chroma_results.json", "w") as f:
            json.dump({
                cond: {m: [float(v) for v in vals] for m, vals in r.items()}
                for cond, r in results.items()
            }, f, indent=2)
        print(f"\nResults saved → {output_dir}/chroma_results.json")
        return

    raise ValueError(f"Unknown task: {task!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task",   required=True, choices=["fish_oil_chroma"])
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.task, args.config)
