"""
Smoke test: one seed × one fold for ChromatogramCNN conditions on fish oil.

Usage:
  python scripts/07_smoke_test.py
  python scripts/07_smoke_test.py --config configs/finetune_fish_oil_chroma.yaml
  python scripts/07_smoke_test.py --seed 42 --fold 2 --epochs 50
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import StratifiedKFold

from msformer.downstream.fish_oil import load_fish_oil_chroma_paths
from msformer.models.chroma_cnn import ChromaCNNConfig
from msformer.training.finetune_chroma import (
    ConditionName,
    _build_model,
    _train_fold,
)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(config_path: str, seed: int, fold_idx: int, epochs: int) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg["finetuning"]["epochs"] = epochs

    npz_paths, y = load_fish_oil_chroma_paths(cfg["data"]["data_dir"])
    device = _device()

    print(f"\nDevice  : {device}")
    print(f"Seed    : {seed}  |  Fold : {fold_idx}  |  Epochs : {epochs}")
    print(f"N       : {len(npz_paths)} samples  |  Classes : {cfg['task']['num_classes']}")
    print(f"CNN ch  : {cfg['model'].get('cnn_channels', 128)}\n")

    m = cfg["model"]
    model_config = ChromaCNNConfig(
        mz_max       = m["mz_max"],
        cnn_channels = m.get("cnn_channels", 128),
        kernel_size  = m.get("kernel_size", 7),
        num_classes  = cfg["task"]["num_classes"],
        dropout      = m.get("dropout", 0.3),
    )
    chroma_pretrain_ckpt = cfg.get("pretrained_checkpoints", {}).get("chroma_pretrain")

    torch.manual_seed(seed)
    np.random.seed(seed)
    skf = StratifiedKFold(
        n_splits=cfg["task"].get("cv_folds", 5), shuffle=True, random_state=seed
    )
    splits = list(skf.split(npz_paths, y))
    if fold_idx >= len(splits):
        raise ValueError(f"fold_idx {fold_idx} out of range (n_splits={len(splits)})")
    train_idx, val_idx = splits[fold_idx]

    print(f"Train : {len(train_idx)} samples  |  Val : {len(val_idx)} samples")
    print("\n" + "=" * 60)
    print(f"SMOKE TEST  fish_oil_chroma  (seed={seed}, fold={fold_idx}, epochs={epochs})")
    print("=" * 60)

    conditions: list[ConditionName] = ["from_scratch"]
    if chroma_pretrain_ckpt and Path(chroma_pretrain_ckpt).exists():
        conditions.append("chroma_pretrain")

    for cond in conditions:
        torch.manual_seed(seed)
        np.random.seed(seed)

        t0    = time.perf_counter()
        model = _build_model(model_config, cond, chroma_pretrain_ckpt).to(device)
        metrics = _train_fold(
            model,
            [npz_paths[i] for i in train_idx],
            [npz_paths[i] for i in val_idx],
            y[train_idx], y[val_idx],
            cfg, device,
        )
        elapsed = time.perf_counter() - t0
        print(f"\n{cond}:")
        print(f"  balanced_accuracy : {metrics['balanced_accuracy']:.3f}")
        print(f"  macro_f1          : {metrics['macro_f1']:.3f}")
        print(f"  time              : {elapsed:.1f}s")

    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Smoke test: ChromatogramCNN (1 seed × 1 fold)")
    ap.add_argument("--config", default="configs/finetune.yaml")
    ap.add_argument("--seed",   type=int, default=0)
    ap.add_argument("--fold",   type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()
    main(args.config, args.seed, args.fold, args.epochs)
