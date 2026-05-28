"""
Fine-tune and evaluate ChromatogramCNN on the fish oil dataset.

Usage:
  python scripts/04_finetune_evaluate.py
  python scripts/04_finetune_evaluate.py --device cuda:0
  python scripts/04_finetune_evaluate.py --config configs/finetune.yaml --device cuda:0

Results are written to results/fish_oil/chroma_results.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from msformer.evaluation.stats import compare_conditions, print_results_table


def main(config_path: str, device: str | None = None) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["output"]["results_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = cfg["task"].get("cv_seeds", list(range(10)))

    from msformer.downstream.fish_oil import load_fish_oil_chroma_paths
    from msformer.training.finetune_chroma import ChromaFinetuner

    npz_paths, y = load_fish_oil_chroma_paths(cfg["data"]["data_dir"])
    finetuner = ChromaFinetuner(cfg, npz_paths, y, device=device)
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate ChromatogramCNN on fish oil GC-MS")
    ap.add_argument("--config", default="configs/finetune.yaml")
    ap.add_argument("--device", type=str, default=None, help="Force device: cuda, cuda:0, cpu")
    args = ap.parse_args()
    main(args.config, args.device)
