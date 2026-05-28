"""
Step 8: Self-supervised pretraining of ChromaNextFramePredictor.

Next-frame prediction along the RT axis — given bins [0..t], predict spectrum at t+1.
Synthetic GC-MS chromatograms are generated on-the-fly from MoNA/MassBank EI-MS
spectra (data/pretraining/spectra.h5), so no fish oil data is used and there is
no leakage into fine-tuning CV folds.

Training is iteration-based: each step draws a fresh random batch, giving
unlimited data variety without epoch bookkeeping.

Usage:
  python scripts/08_pretrain_chroma.py
  python scripts/08_pretrain_chroma.py --config configs/pretrain_chroma.yaml
  python scripts/08_pretrain_chroma.py --iterations 500   # quick smoke test
"""

from __future__ import annotations

import argparse

import h5py
import numpy as np
import torch
import yaml

from msformer.training.pretrain_chroma import pretrain_chroma


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(config_path: str, iterations_override: int | None) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if iterations_override is not None:
        cfg["pretraining"]["n_iterations"] = iterations_override

    with h5py.File(cfg["data"]["h5_path"], "r") as f:
        spectra = f["spectra"][:].astype(np.float32)
    print(f"Loaded {len(spectra)} EI-MS spectra from {cfg['data']['h5_path']}")

    pretrain_chroma(cfg, spectra, _device())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pretrain ChromaNextFramePredictor")
    ap.add_argument("--config",     default="configs/pretrain_chroma.yaml")
    ap.add_argument("--iterations", type=int, default=None, help="Override iteration count")
    args = ap.parse_args()
    main(args.config, args.iterations)
