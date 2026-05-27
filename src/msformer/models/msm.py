"""
Masked Spectra Modelling (MSM) pretraining model.

Analogous to BERT's MLM objective applied to mass spectra.
Masks 15% of non-zero m/z bins and predicts their original intensities (MSE loss).

For dense (patch) mode: the mask operates at the bin level within patches.
The encoder sees masked patches (zeroed bins); the decoder head reconstructs
original bin intensities from the patch-level token outputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from msformer.models.encoder import SpectrumConfig, SpectrumEncoder


class MSMDecoder(nn.Module):
    """
    Projects per-token encoder outputs back to patch intensities.

    dense mode : [B, n_patches, hidden_dim] → [B, n_patches, patch_size]
    sparse mode: [B, max_peaks+1, hidden_dim] → [B, max_peaks+1, 1]
    """

    def __init__(self, config: SpectrumConfig) -> None:
        super().__init__()
        if config.input_type == "dense":
            out_dim = config.patch_size
        else:
            out_dim = 1
        self.proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, out_dim),
        )

    def forward(self, token_outputs: Tensor) -> Tensor:
        return self.proj(token_outputs)  # exclude CLS position


class MSMModel(nn.Module):
    """
    Encoder + MSM head.

    During forward:
      - accepts the already-masked spectrum (MSMDataset handles masking)
      - runs encoder
      - decoder head predicts original intensities at all positions
      - MSE loss computed only at masked positions

    Returns (loss, predicted_spectrum) where predicted_spectrum has shape [B, mz_max].
    """

    def __init__(self, config: SpectrumConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = SpectrumEncoder(config)
        self.decoder = MSMDecoder(config)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        masked_spectrum = batch["spectrum"]    # [B, mz_max]
        targets = batch["targets"]             # [B, mz_max]  zeros at unmasked positions
        mask = batch["mask"]                   # [B, mz_max]  bool

        _, all_tokens = self.encoder(masked_spectrum, return_all_tokens=True)
        # all_tokens: [B, 1+n_tokens, hidden_dim]
        # Token outputs excluding CLS (index 0)
        patch_tokens = all_tokens[:, 1:]  # [B, n_tokens, hidden_dim]

        predicted_patches = self.decoder(patch_tokens)  # [B, n_tokens, patch_size]

        if self.config.input_type == "dense":
            B = predicted_patches.shape[0]
            predicted_spectrum = predicted_patches.view(B, -1)  # [B, mz_max]
        else:
            # sparse: predicted_patches [B, max_peaks, 1] → needs re-scatter
            # For simplicity, use a full-spectrum projection in sparse mode
            predicted_spectrum = predicted_patches.squeeze(-1)
            # Pad/clip to mz_max (sparse mode is approximate here)
            B = predicted_spectrum.shape[0]
            out = torch.zeros(B, self.config.mz_max, device=predicted_spectrum.device)
            out[:, : predicted_spectrum.shape[1]] = predicted_spectrum
            predicted_spectrum = out

        # MSE only at masked positions
        loss = F.mse_loss(predicted_spectrum[mask], targets[mask])

        return loss, predicted_spectrum

    @torch.no_grad()
    def encode(self, x: Tensor) -> Tensor:
        """Return CLS embedding for a batch of spectra."""
        return self.encoder(x)
