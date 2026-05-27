"""
Step 6: Generate attention rollout visualisations for representative samples.

python scripts/06_visualize_attention.py \
    --task olive_oil \
    --checkpoint checkpoints/msm/best.pt \
    --config configs/finetune_oliveoil.yaml \
    --output-dir results/olive_oil/attention_maps
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from msformer.evaluation.visualize import batch_visualise
from msformer.models.encoder import SpectrumClassifier, SpectrumConfig


def main(task: str, checkpoint: str, config_path: str, output_dir: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg_d = cfg["model"]
    model_config = SpectrumConfig(
        input_type=model_cfg_d["input_type"],
        mz_max=model_cfg_d["mz_max"],
        patch_size=model_cfg_d["patch_size"],
        hidden_dim=model_cfg_d["hidden_dim"],
        num_layers=model_cfg_d["num_layers"],
        num_heads=model_cfg_d["num_heads"],
        ffn_dim=model_cfg_d.get("ffn_dim", model_cfg_d["hidden_dim"] * 4),
        dropout=0.0,  # eval mode
        num_classes=cfg["task"]["num_classes"],
    )

    clf = SpectrumClassifier(model_config)
    clf.load_pretrained_encoder(checkpoint)
    clf.eval()

    if task == "olive_oil":
        from msformer.downstream.olive_oil import load_olive_oil_data
        X, y, sample_ids = load_olive_oil_data(cfg["data"]["data_dir"])
    elif task == "hemp_seed":
        from msformer.downstream.hemp_seed import load_hemp_seed_data
        X, y, sample_ids, _ = load_hemp_seed_data(cfg["data"]["data_dir"])
    else:
        raise ValueError(f"Unknown task: {task}")

    batch_visualise(
        encoder=clf.encoder,
        X=X,
        y=y,
        sample_ids=sample_ids,
        config=model_config,
        output_dir=output_dir,
        n_per_class=2,
    )
    print(f"Attention maps saved to {output_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["olive_oil", "hemp_seed"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", default="results/attention_maps")
    args = ap.parse_args()
    main(args.task, args.checkpoint, args.config, args.output_dir)
