"""
Fine-tune and evaluate ChromatogramCNN on the MTBLS71 urine dataset.

Usage:
  python scripts/09_finetune_evaluate_urine.py
  python scripts/09_finetune_evaluate_urine.py --device cuda:0
  python scripts/09_finetune_evaluate_urine.py --config configs/finetune_urine.yaml

Results are written to results/mtbls71/chroma_results.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from chroma_dcnn.downstream.urine import load_urine_chroma_paths
from chroma_dcnn.evaluation.stats import compare_conditions, print_results_table
from chroma_dcnn.training.finetune_chroma import ChromaFinetuner


def main(config_path: str, device: str | None = None) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["output"]["results_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = cfg["task"].get("cv_seeds", list(range(10)))

    data_dir = Path(cfg["data"]["data_dir"])
    npz_paths, y = load_urine_chroma_paths(data_dir)

    groups_path = data_dir / "groups.txt"
    groups = np.array(groups_path.read_text().splitlines()) if groups_path.exists() else None

    finetuner = ChromaFinetuner(cfg, npz_paths, y, device=device, groups=groups)
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
    ap = argparse.ArgumentParser(description="Evaluate ChromatogramCNN on MTBLS71 urine GC-MS")
    ap.add_argument("--config", default="configs/finetune_urine.yaml")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()
    main(args.config, args.device)
