"""
Self-supervised next-frame prediction model for GC-MS chromatograms.

Task: given RT bins [0..t], predict the mass spectrum at bin t+1.

This trains two components that transfer directly to ChromatogramCNN:
  spec_proj — Linear(mz_max, cnn_channels): learns which m/z features are
              informative for predicting chromatographic progression.
  cnn       — three dilated causal ResBlocks: learns the temporal structure
              of eluting peaks (appearance, apex, tail-off, co-elution).

Causal vs non-causal ResBlocks:
  Pretraining uses left-only padding so each position only sees past context.
  Fine-tuning (ChromatogramCNN) uses symmetric padding to see both directions.
  The Conv1d weight shapes are identical, so conv weights transfer directly.

Normalisation:
  The causal blocks use LayerNorm over channels at each RT position.  BatchNorm
  would pool statistics over the whole RT axis in training mode, letting
  position t see positions > t.  Only the conv weights transfer to
  ChromatogramCNN, whose (non-causal) blocks use BatchNorm.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ChromaPretrainConfig:
    mz_max: int = 1000
    n_bins: int = 200
    cnn_channels: int = 128    # must match ChromaCNNConfig.cnn_channels
    kernel_size: int = 7
    dropout: float = 0.1


class _CausalResBlock1d(nn.Module):
    """
    Residual Conv1d block with causal (left-only) padding.

    Output at position t depends only on positions ≤ t, in both training and
    eval mode, making the block usable for autoregressive prediction along the
    RT axis.  LayerNorm normalises over channels at each position, so no
    statistics are shared across time.

    The Conv1d weight shapes match the non-causal _ResBlock1d used by
    ChromatogramCNN, so weights transfer between the two.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.ln1   = nn.LayerNorm(channels)
        self.drop  = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.ln2   = nn.LayerNorm(channels)
        self.act   = nn.ReLU()

    @staticmethod
    def _norm(ln: nn.LayerNorm, x: Tensor) -> Tensor:
        """Apply LayerNorm over channels of a channel-first [B, C, N] tensor."""
        return ln(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, x: Tensor) -> Tensor:   # x: [B, C, N]
        res = x
        x = self.act(self._norm(self.ln1, self.conv1(F.pad(x, (self.left_pad, 0)))))
        x = self.drop(x)
        x = self._norm(self.ln2, self.conv2(F.pad(x, (self.left_pad, 0))))
        return self.act(res + x)


class ChromaNextFramePredictor(nn.Module):
    """
    Next-frame predictor for GC-MS chromatograms.

    Architecture:
      spec_proj  — Linear(mz_max → cnn_channels): per-bin spectral encoder
      pos_embed  — learned positional embedding along the RT axis
      cnn        — three causal dilated ResBlocks (dilation 1 / 2 / 4)
      pred_head  — Linear(cnn_channels → mz_max): reconstruct next-frame spectrum

    Pretraining objective: MSE between predicted and actual next frame.

    Weight transfer to ChromatogramCNN after pretraining:
      model.spec_proj → ChromatogramCNN.spec_proj   (direct)
      model.cnn[i].*  → ChromatogramCNN.cnn[i].*    (Conv1d weights are compatible)
    """

    def __init__(self, config: ChromaPretrainConfig) -> None:
        super().__init__()
        ch = config.cnn_channels

        self.spec_proj = nn.Linear(config.mz_max, ch)
        self.pos_embed = nn.Embedding(config.n_bins, ch)

        k, d = config.kernel_size, config.dropout
        self.cnn = nn.Sequential(
            _CausalResBlock1d(ch, k, dilation=1, dropout=d),
            _CausalResBlock1d(ch, k, dilation=2, dropout=d),
            _CausalResBlock1d(ch, k, dilation=4, dropout=d),
        )
        self.pred_head = nn.Linear(ch, config.mz_max)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        x : [B, N_bins, mz_max]

        Returns
        -------
        pred   : [B, N_bins-1, mz_max]  — predicted spectra (positions 1..N)
        target : [B, N_bins-1, mz_max]  — actual next frames (x[:, 1:, :])
        """
        B, N, mz = x.shape

        # Per-bin encoding + positional embedding
        h = self.spec_proj(x.reshape(B * N, mz)).view(B, N, -1)         # [B, N, ch]
        h = h + self.pos_embed(torch.arange(N, device=x.device))        # [B, N, ch]

        # Causal CNN: each position aggregates only past context
        h = self.cnn(h.transpose(1, 2)).transpose(1, 2)                 # [B, N, ch]

        # Position t predicts bin t+1
        pred   = self.pred_head(h[:, :-1, :])   # [B, N-1, mz_max]
        target = x[:, 1:, :]                    # [B, N-1, mz_max]
        return pred, target
