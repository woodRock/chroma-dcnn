"""
Self-supervised pretraining of ChromaNextFramePredictor on unlabeled GC-MS data.

Trains with next-frame prediction along the RT axis:
  at each bin t, predict the spectrum at bin t+1.

Uses iteration-based training: each step draws a fresh random batch of synthetic
GC-MS chromatograms from the MoNA/MassBank spectra array, giving effectively
unlimited data variety without epoch bookkeeping.

Saves a checkpoint containing:
  spec_proj_state  — weights for ChromatogramCNN.spec_proj
  cnn_state        — weights for ChromatogramCNN.cnn
  config           — ChromaPretrainConfig
  n_iterations     — total gradient steps taken
  final_loss       — best loss seen
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from chroma_dcnn.models.chroma_pretrain import ChromaNextFramePredictor, ChromaPretrainConfig


def _make_batch(
    spectra: np.ndarray,
    batch_size: int,
    n_bins: int,
    n_compounds_range: tuple[int, int],
    sigma_range: tuple[float, float],
    rng: np.random.Generator,
) -> torch.Tensor:
    """
    Generate a batch of synthetic GC-MS chromatograms on-the-fly.

    Each sample: K randomly chosen EI-MS spectra placed at random RT positions
    with Gaussian elution profiles, summed and sqrt+L2 normalised per bin.

    Returns [batch_size, n_bins, mz_max] float32 tensor.
    """
    n_spec, mz_max = spectra.shape
    lo, hi = n_compounds_range
    K = int(rng.integers(lo, hi + 1))

    spec_idx     = rng.integers(0, n_spec, size=(batch_size, K))
    batch_specs  = spectra[spec_idx]                                      # [B, K, mz_max]

    t   = np.arange(n_bins, dtype=np.float32)
    mu  = rng.uniform(0, n_bins, size=(batch_size, K)).astype(np.float32)
    sig = rng.uniform(*sigma_range, size=(batch_size, K)).astype(np.float32)

    # Gaussian elution profiles: [B, K, n_bins]
    profiles = np.exp(-0.5 * ((t[None, None] - mu[:, :, None]) / sig[:, :, None]) ** 2)

    # Sum spectral contributions: [B, n_bins, mz_max]
    chroma = np.einsum("bkn,bkm->bnm", profiles, batch_specs)

    # Sqrt + L2 normalise per bin
    chroma = np.sqrt(np.maximum(chroma, 0.0))
    norms  = np.linalg.norm(chroma, axis=-1, keepdims=True)
    chroma = chroma / np.maximum(norms, 1e-8)

    return torch.from_numpy(chroma.astype(np.float32))


def pretrain_chroma(
    cfg: dict,
    spectra: np.ndarray,
    device: torch.device,
) -> str:
    """
    Train ChromaNextFramePredictor with iteration-based sampling and save checkpoint.

    Parameters
    ----------
    cfg     : pretraining YAML config dict
    spectra : [N, mz_max] float32 array of EI-MS spectra (e.g. from spectra.h5)
    device  : compute device

    Returns
    -------
    Path to saved checkpoint.
    """
    pcfg = cfg["pretraining"]
    mcfg = cfg["model"]

    config = ChromaPretrainConfig(
        mz_max       = mcfg["mz_max"],
        n_bins       = mcfg["n_bins"],
        cnn_channels = mcfg["cnn_channels"],
        kernel_size  = mcfg.get("kernel_size", 7),
        dropout      = mcfg.get("dropout", 0.1),
    )

    model = ChromaNextFramePredictor(config).to(device)

    n_iterations = pcfg["n_iterations"]
    batch_size   = pcfg.get("batch_size", 16)
    lr           = pcfg.get("lr", 1e-3)
    wd           = pcfg.get("weight_decay", 1e-4)
    log_every    = pcfg.get("log_every", 500)

    n_compounds_range = tuple(cfg["data"].get("n_compounds_range", [5, 30]))
    sigma_range       = tuple(cfg["data"].get("sigma_range", [1.0, 15.0]))

    opt   = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = CosineAnnealingLR(opt, T_max=n_iterations, eta_min=lr * 0.01)
    rng   = np.random.default_rng()

    print(f"\nPretraining ChromaNextFramePredictor (iteration-based)")
    print(f"  EI-MS spectra : {len(spectra)}")
    print(f"  N bins        : {config.n_bins}")
    print(f"  CNN channels  : {config.cnn_channels}")
    print(f"  Iterations    : {n_iterations}")
    print(f"  Batch size    : {batch_size}")
    print(f"  Device        : {device}\n")

    best_loss   = float("inf")
    running_loss = 0.0

    model.train()
    for it in range(1, n_iterations + 1):
        chroma = _make_batch(
            spectra, batch_size, config.n_bins,
            n_compounds_range, sigma_range, rng,
        ).to(device)

        pred, target = model(chroma)
        loss = nn.functional.mse_loss(pred, target)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), pcfg.get("grad_clip", 1.0))
        opt.step()
        sched.step()

        running_loss += loss.item()

        if it % log_every == 0 or it == n_iterations:
            avg = running_loss / log_every
            if avg < best_loss:
                best_loss = avg
            print(f"  iter {it:>6d}/{n_iterations}  loss={avg:.6f}  best={best_loss:.6f}")
            running_loss = 0.0

    ckpt_dir  = Path(cfg["output"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best.pt"

    torch.save({
        "spec_proj_state": model.spec_proj.state_dict(),
        "cnn_state":       model.cnn.state_dict(),
        "config":          config,
        "n_iterations":    n_iterations,
        "final_loss":      best_loss,
    }, ckpt_path)

    print(f"\nCheckpoint saved → {ckpt_path}  (best loss={best_loss:.6f})")
    return str(ckpt_path)
