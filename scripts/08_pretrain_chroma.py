"""
Step 8: Self-supervised pretraining of ChromaNextFramePredictor.

Next-frame prediction along the RT axis — given bins [0..t], predict spectrum at t+1.
Pretraining uses synthetic GC-MS chromatograms generated from MoNA/MassBank EI-MS
spectra (data/pretraining/spectra.h5), NOT the fish oil data.  This avoids data
leakage: CV test samples are never seen during pretraining.

Usage:
  python scripts/08_pretrain_chroma.py
  python scripts/08_pretrain_chroma.py --config configs/pretrain_chroma.yaml
  python scripts/08_pretrain_chroma.py --epochs 500   # quick smoke test
"""

from __future__ import annotations

import argparse

import torch
import yaml

from msformer.data.datasets import SyntheticChromaDataset
from msformer.training.pretrain_chroma import pretrain_chroma


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(config_path: str, epochs_override: int | None) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if epochs_override is not None:
        cfg["pretraining"]["epochs"] = epochs_override

    dcfg = cfg["data"]
    dataset = SyntheticChromaDataset(
        h5_path           = dcfg["h5_path"],
        n_samples         = dcfg.get("n_samples", 50_000),
        n_bins            = cfg["model"]["n_bins"],
        n_compounds_range = tuple(dcfg.get("n_compounds_range", [5, 30])),
        sigma_range       = tuple(dcfg.get("sigma_range", [1.0, 15.0])),
    )
    device = _device()
    pretrain_chroma(cfg, dataset, device)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pretrain ChromaNextFramePredictor")
    ap.add_argument("--config", default="configs/pretrain_chroma.yaml")
    ap.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    args = ap.parse_args()
    main(args.config, args.epochs)
