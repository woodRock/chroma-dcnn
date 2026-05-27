"""
Step 3: Pretrain the transformer encoder.

Run MSM:
  python scripts/03_pretrain.py --config configs/pretrain_msm.yaml

Run contrastive:
  python scripts/03_pretrain.py --config configs/pretrain_contrastive.yaml

To compare both input_types (dense vs sparse), pass --input-type sparse:
  python scripts/03_pretrain.py --config configs/pretrain_msm.yaml --input-type sparse
"""

import argparse
import yaml
from msformer.training.pretrain import PretrainTrainer


def main(config_path: str, overrides: dict) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides
    for key, val in overrides.items():
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = val

    trainer = PretrainTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--input-type", choices=["dense", "sparse"], default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    overrides = {}
    if args.input_type:
        overrides["model.input_type"] = args.input_type
    if args.epochs:
        overrides["training.epochs"] = args.epochs
    if args.batch_size:
        overrides["training.batch_size"] = args.batch_size

    main(args.config, overrides)
