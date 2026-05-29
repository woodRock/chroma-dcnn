"""
ChromatogramCNN: 1D CNN classifier for 2D GC-MS chromatograms.

Architecture:
  1. Per-bin spectral projection  Linear(mz_max → cnn_channels)  →  [B, N_bins, cnn_channels]
  2. Three dilated 1D ResBlocks (kernel=7, dilation=1/2/4)
       receptive field: 7 → 19 → 43 bins  (spans ~8 co-eluting peak groups)
  3. Dual temporal pooling: global max-pool ∥ soft attention-pool → [B, 2·cnn_channels]
       max   — picks the single most discriminative RT window
       attn  — weighted sum over all bins (multi-peak evidence)
  4. Linear classification head
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ChromaCNNConfig:
    mz_max: int = 1000
    cnn_channels: int = 128
    kernel_size: int = 7
    num_classes: int = 4
    dropout: float = 0.3


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _ResBlock1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad        = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=pad)
        self.bn1   = nn.BatchNorm1d(channels)
        self.drop  = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=pad)
        self.bn2   = nn.BatchNorm1d(channels)
        self.act   = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x   = self.act(self.bn1(self.conv1(x)))
        x   = self.drop(x)
        x   = self.bn2(self.conv2(x))
        return self.act(res + x)


class _AttentionPool1d(nn.Module):
    """Soft attention pooling over the temporal (RT) dimension.  x: [B,C,N] → [B,C]"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.score = nn.Linear(channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        x_t = x.transpose(1, 2)                    # [B, N, C]
        w = F.softmax(self.score(x_t), dim=1)      # [B, N, 1]
        return (x_t * w).sum(dim=1)                # [B, C]


# ---------------------------------------------------------------------------
# ChromatogramCNN
# ---------------------------------------------------------------------------

class ChromatogramCNN(nn.Module):
    """
    Parameters
    ----------
    config    : ChromaCNNConfig
    condition : ignored (kept for API compatibility) — always trains from scratch
    """

    def __init__(self, config: ChromaCNNConfig, condition: str = "from_scratch") -> None:
        super().__init__()
        ch = config.cnn_channels
        self.spec_proj = nn.Linear(config.mz_max, ch)

        k, d = config.kernel_size, config.dropout
        self.cnn = nn.Sequential(
            _ResBlock1d(ch, k, dilation=1, dropout=d),
            _ResBlock1d(ch, k, dilation=2, dropout=d),
            _ResBlock1d(ch, k, dilation=4, dropout=d),
        )
        self.attn_pool = _AttentionPool1d(ch)
        self.head = nn.Linear(ch * 2, config.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """[B, N_bins, mz_max] → [B, num_classes]"""
        B, N, mz = x.shape
        emb = self.spec_proj(x.reshape(B * N, mz)).view(B, N, -1)  # [B, N, ch]
        x   = emb.transpose(1, 2)                                    # [B, ch, N]
        x   = self.cnn(x)
        x_max  = x.max(dim=-1).values                                # [B, ch]
        x_attn = self.attn_pool(x)                                   # [B, ch]
        return self.head(torch.cat([x_max, x_attn], dim=-1))

    def load_pretrained_chroma_encoder(self, checkpoint_path: str) -> None:
        """Load spec_proj and CNN weights from a ChromaNextFramePredictor checkpoint."""
        import sys
        import chroma_dcnn as _pkg
        sys.modules.setdefault("msformer", _pkg)          # compat: checkpoint saved under old name
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.spec_proj.load_state_dict(ckpt["spec_proj_state"])
        print(f"  Loaded pretrained spec_proj from {checkpoint_path}")
        if "cnn_state" in ckpt:
            self.cnn.load_state_dict(ckpt["cnn_state"])
            print(f"  Loaded pretrained CNN from {checkpoint_path}")
