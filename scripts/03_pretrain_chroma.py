"""
Pretrain ChromaNextFramePredictor via next-frame prediction on synthetic GC-MS data.

Synthetic chromatograms are generated on-the-fly from MoNA/MassBank EI-MS spectra
so no fish oil data is used — no leakage into fine-tuning CV folds.

Usage:
  python scripts/03_pretrain_chroma.py
  python scripts/03_pretrain_chroma.py --device cuda:0
  python scripts/03_pretrain_chroma.py --iterations 200   # quick smoke test

Outputs: checkpoints/chroma_pretrain/best.pt and results/pretrain_summary.json (wall-clock time,
final and best MSE, iterations, device) for the paper's description of pretraining.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from chroma_dcnn.training.pretrain_chroma import pretrain_chroma

SUMMARY = Path("results/pretrain_summary.json")


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
    start = time.perf_counter()
    ckpt_path = pretrain_chroma(cfg, spectra, device)
    seconds = time.perf_counter() - start

    # What the paper reports about pretraining (wall-clock time, final MSE), for the ledger.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    summary = {
        "wall_seconds": seconds,
        "wall_minutes": seconds / 60,
        "final_mse": float(ckpt["last_loss"]),
        "best_mse": float(ckpt["final_loss"]),
        "n_iterations": int(ckpt["n_iterations"]),
        "n_spectra": int(len(spectra)),
        "seed": ckpt.get("seed"),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "checkpoint": str(ckpt_path),
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    minutes, mse = summary["wall_minutes"], summary["final_mse"]
    print(f"Pretraining took {minutes:.1f} min; final MSE {mse:.3e} → {SUMMARY}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pretrain ChromaNextFramePredictor")
    ap.add_argument("--config",     default="configs/pretrain.yaml")
    ap.add_argument("--iterations", type=int, default=None, help="Override iteration count")
    ap.add_argument("--device",     type=str, default=None, help="Force device: cuda, cuda:0, cpu")
    args = ap.parse_args()
    main(args.config, args.iterations, args.device)
