"""
Self-supervised pretraining of ChromaNextFramePredictor on unlabeled GC-MS data.

Trains with next-frame prediction along the RT axis:
  at each bin t, predict the spectrum at bin t+1.

Saves a checkpoint containing:
  spec_proj_state — weights for ChromatogramCNN.spec_proj
  cnn_state       — weights for ChromatogramCNN.cnn  (Conv1d shapes are compatible
                    between causal pretraining and non-causal fine-tuning ResBlocks)
  config          — ChromaPretrainConfig

Usage:
  from msformer.training.pretrain_chroma import pretrain_chroma
  pretrain_chroma(cfg, npz_paths, device)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from msformer.data.datasets import SyntheticChromaDataset
from msformer.models.chroma_pretrain import ChromaNextFramePredictor, ChromaPretrainConfig


def pretrain_chroma(
    cfg: dict,
    dataset: "torch.utils.data.Dataset",
    device: torch.device,
) -> str:
    """
    Train ChromaNextFramePredictor and save checkpoint.

    Parameters
    ----------
    cfg     : pretraining YAML config dict
    dataset : Dataset yielding (chroma [n_bins, mz_max], label) pairs.
              Pass a SyntheticChromaDataset built from spectra.h5 to avoid
              leaking fish oil test samples into the pretraining phase.
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

    loader = DataLoader(
        dataset,
        batch_size = pcfg.get("batch_size", 16),
        shuffle    = True,
        drop_last  = False,
    )

    epochs = pcfg["epochs"]
    lr     = pcfg.get("lr", 1e-3)
    wd     = pcfg.get("weight_decay", 1e-4)
    opt    = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched  = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

    log_every = pcfg.get("log_every", 100)
    best_loss = float("inf")

    print(f"\nPretraining ChromaNextFramePredictor")
    print(f"  Samples  : {len(dataset)}")
    print(f"  N bins   : {config.n_bins}")
    print(f"  CNN ch   : {config.cnn_channels}")
    print(f"  Epochs   : {epochs}")
    print(f"  Device   : {device}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for chroma, _ in loader:
            chroma = chroma.to(device)
            pred, target = model(chroma)
            loss = nn.functional.mse_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), pcfg.get("grad_clip", 1.0))
            opt.step()
            epoch_loss += loss.item()
            n_batches  += 1

        sched.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss

        if epoch % log_every == 0 or epoch == epochs:
            print(f"  epoch {epoch:>5d}/{epochs}  loss={avg_loss:.6f}  best={best_loss:.6f}")

    # Save checkpoint
    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best.pt"

    torch.save({
        "spec_proj_state": model.spec_proj.state_dict(),
        "cnn_state":       model.cnn.state_dict(),
        "config":          config,
        "final_loss":      best_loss,
    }, ckpt_path)

    print(f"\nCheckpoint saved → {ckpt_path}  (best loss={best_loss:.6f})")
    return str(ckpt_path)
