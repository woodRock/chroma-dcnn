"""
Pretrain ChromaNextFramePredictor via next-frame prediction on synthetic GC-MS data.

Synthetic chromatograms are generated on-the-fly from MoNA/MassBank EI-MS spectra
so no fish oil data is used — no leakage into fine-tuning CV folds.

Usage:
  python scripts/08_pretrain_chroma.py
  python scripts/08_pretrain_chroma.py --device cuda:0
  python scripts/08_pretrain_chroma.py --iterations 200   # quick smoke test
"""

from __future__ import annotations

import argparse

import h5py
import numpy as np
import torch
import yaml

from msformer.training.pretrain_chroma import pretrain_chroma


def _device(device_str: str | None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(config_path: str, iterations_override: int | None, device_str: str | None) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if iterations_override is not None:
        cfg["pretraining"]["n_iterations"] = iterations_override

    print(f"Loading spectra from {cfg['data']['h5_path']} ...")
    with h5py.File(cfg["data"]["h5_path"], "r") as f:
        spectra = f["spectra"][:].astype(np.float32)
    print(f"Loaded {len(spectra)} EI-MS spectra")

    device = _device(device_str)
    print(f"Device: {device}")
    pretrain_chroma(cfg, spectra, device)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pretrain ChromaNextFramePredictor")
    ap.add_argument("--config",     default="configs/pretrain.yaml")
    ap.add_argument("--iterations", type=int, default=None, help="Override iteration count")
    ap.add_argument("--device",     type=str, default=None, help="Force device: cuda, cuda:0, cpu")
    args = ap.parse_args()
    main(args.config, args.iterations, args.device)
