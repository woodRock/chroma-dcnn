"""
ChromatogramCNN: 1D CNN classifier for 2D GC-MS chromatograms.

Two conditions:

  frozen_embed  — Frozen DensePatchEmbedding (pretrained) applied independently to
                  each RT bin; patches are max-pooled to a hidden_dim vector;
                  a trainable to_cnn projection + dilated CNN + head are learned.
                  Encoder runs once to pre-encode; training touches only CNN+head.

  from_scratch  — No pretrained weights.  A small Linear(mz_max → cnn_channels)
                  replaces the transformer encoder; the full model is trained
                  end-to-end on raw chromatograms.  Fast because there is no
                  transformer: ~30 s per 200-epoch run vs >2 min with the full
                  6-layer SpectrumEncoder.

Shared CNN architecture:
  1. Per-bin spectral embedding → [B, N_bins, cnn_channels]
  2. Three dilated 1D ResBlocks (kernel=7, dilation=1/2/4)
       receptive field: 7 → 19 → 43 bins  (spans ~8 co-eluting peak groups)
  3. Dual temporal pooling: global max-pool ∥ soft attention-pool → [B, 2·cnn_channels]
       max   — picks the single most discriminative RT window (peak detection)
       attn  — learns a weighted sum over all bins (multi-peak evidence)
  4. Linear classification head
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from msformer.models.encoder import SpectrumConfig, SpectrumEncoder


@dataclass
class ChromaCNNConfig:
    # SpectrumEncoder params — must match pretrained checkpoint (frozen_embed only)
    mz_max: int = 1000
    patch_size: int = 50
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    ffn_dim: int = 1024
    encoder_dropout: float = 0.1
    # CNN params
    cnn_channels: int = 128
    kernel_size: int = 7
    num_classes: int = 4
    dropout: float = 0.3


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _ResBlock1d(nn.Module):
    """
    Residual dilated Conv1d block with same-size output.

    Attribute names (conv1, bn1, conv2, bn2) match _CausalResBlock1d in
    chroma_pretrain.py so that state dicts transfer between pretraining
    (causal, left-only padding) and fine-tuning (non-causal, symmetric padding).
    """

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
    """
    Soft attention pooling over the temporal (RT) dimension.

    Learns a scalar score per time-step; softmax over all steps gives the
    weighting.  Compared to global max-pool (which picks one peak), this
    attends to multiple discriminative RT windows simultaneously.

    x : [B, C, N]  →  [B, C]
    """

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
    condition : "frozen_embed" | "from_scratch"
    """

    def __init__(self, config: ChromaCNNConfig, condition: str = "frozen_embed") -> None:
        super().__init__()
        self.condition = condition
        ch = config.cnn_channels

        if condition == "frozen_embed":
            spec_cfg = SpectrumConfig(
                input_type      = "dense",
                mz_max          = config.mz_max,
                patch_size      = config.patch_size,
                hidden_dim      = config.hidden_dim,
                num_layers      = config.num_layers,
                num_heads       = config.num_heads,
                ffn_dim         = config.ffn_dim,
                dropout         = config.encoder_dropout,
                num_classes     = 0,
            )
            self.encoder = SpectrumEncoder(spec_cfg)
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.to_cnn = (
                nn.Linear(config.hidden_dim, ch)
                if config.hidden_dim != ch else nn.Identity()
            )
        else:
            # Direct per-bin linear projection — no pretrained weights, fast
            self.spec_proj = nn.Linear(config.mz_max, ch)

        k, d = config.kernel_size, config.dropout
        self.cnn = nn.Sequential(
            _ResBlock1d(ch, k, dilation=1, dropout=d),
            _ResBlock1d(ch, k, dilation=2, dropout=d),
            _ResBlock1d(ch, k, dilation=4, dropout=d),
        )
        self.attn_pool = _AttentionPool1d(ch)
        self.head = nn.Linear(ch * 2, config.num_classes)  # max ∥ attn

    def encode_bins(self, x: Tensor) -> Tensor:
        """
        Per-bin spectral embedding.

        Parameters
        ----------
        x : [B, N_bins, mz_max]

        Returns
        -------
        frozen_embed : [B, N_bins, hidden_dim]   (further projected by to_cnn)
        from_scratch : [B, N_bins, cnn_channels]
        """
        B, N, mz = x.shape
        flat = x.reshape(B * N, mz)
        if self.condition == "frozen_embed":
            with torch.no_grad():
                patches = self.encoder.embedding(flat)   # [B*N, n_patches, hidden_dim]
            emb = patches.max(dim=1).values              # [B*N, hidden_dim]
        else:
            emb = self.spec_proj(flat)                   # [B*N, cnn_channels]
        return emb.view(B, N, -1)

    def forward_from_embeddings(self, emb: Tensor) -> Tensor:
        """
        CNN + dual-pool head on per-bin embeddings.

        Parameters
        ----------
        emb : [B, N_bins, hidden_dim]    for frozen_embed
              [B, N_bins, cnn_channels]  for from_scratch

        Returns
        -------
        logits : [B, num_classes]
        """
        if self.condition == "frozen_embed":
            x = self.to_cnn(emb)          # [B, N, cnn_channels]
        else:
            x = emb
        x = x.transpose(1, 2)             # [B, cnn_channels, N]
        x = self.cnn(x)                   # [B, cnn_channels, N]
        x_max  = x.max(dim=-1).values     # [B, cnn_channels]
        x_attn = self.attn_pool(x)        # [B, cnn_channels]
        return self.head(torch.cat([x_max, x_attn], dim=-1))

    def forward(self, x: Tensor) -> Tensor:
        """[B, N_bins, mz_max] → [B, num_classes]"""
        return self.forward_from_embeddings(self.encode_bins(x))

    def load_pretrained_encoder(self, checkpoint_path: str) -> None:
        """Load DensePatchEmbedding weights from MSM checkpoint (frozen_embed only)."""
        assert self.condition == "frozen_embed", (
            "load_pretrained_encoder is only for the frozen_embed condition"
        )
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

    def load_pretrained_chroma_encoder(self, checkpoint_path: str) -> None:
        """
        Load spec_proj and CNN weights from a ChromaNextFramePredictor checkpoint.

        The causal ResBlock Conv1d weights are shape-compatible with the non-causal
        ResBlocks used here — only the padding strategy differs between pretraining
        and fine-tuning, not the learned weight matrices.
        """
        assert self.condition == "from_scratch", (
            "load_pretrained_chroma_encoder is for the from_scratch / chroma_pretrain condition"
        )
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.spec_proj.load_state_dict(ckpt["spec_proj_state"])
        print(f"  Loaded pretrained spec_proj from {checkpoint_path}")
        if "cnn_state" in ckpt:
            self.cnn.load_state_dict(ckpt["cnn_state"])
            print(f"  Loaded pretrained CNN from {checkpoint_path}")
