"""
Per-scan encoding model for 2D GC-MS chromatograms.

Architecture:
  1. Apply the pretrained SpectrumEncoder independently to each RT scan
     (all scans from all samples are batched together into one encoder call)
  2. TIC-weighted pool: average scan embeddings weighted by total ion current,
     so chromatographic peaks dominate and baseline scans contribute near zero
  3. Linear classification head

This directly leverages MSM pretraining — each RT scan is an EI mass spectrum
in the same m/z domain as the MoNA/MassBank reference spectra.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from msformer.models.encoder import SpectrumConfig, SpectrumEncoder


class ScanPoolClassifier(nn.Module):
    """
    Encoder + TIC-weighted pooling + classification head.

    Parameters
    ----------
    config         : SpectrumConfig with num_classes > 0
    freeze_encoder : if True, encoder weights are frozen (linear probe mode)
    """

    def __init__(self, config: SpectrumConfig, freeze_encoder: bool = False) -> None:
        super().__init__()
        assert config.num_classes > 0, "Set config.num_classes for classification"
        self.encoder = SpectrumEncoder(config)
        self.head = nn.Linear(config.hidden_dim, config.num_classes)
        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, scans: Tensor, tic_weights: Tensor) -> Tensor:
        """
        Parameters
        ----------
        scans       : [B, K, mz_max]  — K normalised m/z spectra per sample
        tic_weights : [B, K]          — TIC weights, each row sums to 1

        Returns
        -------
        logits : [B, num_classes]
        """
        B, K, mz = scans.shape

        # Flatten samples × scans for a single batched encoder call
        flat = scans.reshape(B * K, mz)                  # [B*K, mz_max]

        if self.freeze_encoder:
            with torch.no_grad():
                embeddings = self.encoder(flat)          # [B*K, hidden_dim]
        else:
            embeddings = self.encoder(flat)

        embeddings = embeddings.view(B, K, -1)           # [B, K, hidden_dim]

        # TIC-weighted pool: dominant peaks drive the sample embedding
        weights = tic_weights.unsqueeze(-1)              # [B, K, 1]
        pooled = (embeddings * weights).sum(dim=1)       # [B, hidden_dim]

        return self.head(pooled)                         # [B, num_classes]

    def load_pretrained_encoder(self, checkpoint_path: str) -> None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        encoder_state = {
            k.removeprefix("encoder."): v
            for k, v in ckpt["model_state"].items()
            if k.startswith("encoder.")
        }
        missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")
        print(f"  Loaded pretrained encoder from {checkpoint_path}")


class EncodedScanPoolHead(nn.Module):
    """
    TIC-weighted pool + classification head for pre-encoded scan embeddings.

    Used by the linear-probe fast path: the encoder runs once before training
    to produce [N, K, hidden_dim] embeddings, then only this lightweight head
    is trained.  Forward signature is identical to ScanPoolClassifier so the
    same _eval_loss / _compute_metrics helpers work for both.
    """

    def __init__(self, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, embeddings: Tensor, tic_weights: Tensor) -> Tensor:
        """
        Parameters
        ----------
        embeddings  : [B, K, hidden_dim]
        tic_weights : [B, K]

        Returns
        -------
        logits : [B, num_classes]
        """
        weights = tic_weights.unsqueeze(-1)          # [B, K, 1]
        pooled = (embeddings * weights).sum(dim=1)   # [B, hidden_dim]
        return self.head(pooled)
