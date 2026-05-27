"""
Transformer encoder for mass spectra.

Two input representations (controlled by config.input_type):

  dense  — ViT-style patch embedding.
           The 1000-dim spectrum is divided into non-overlapping patches of
           `patch_size` bins.  Each patch → linear projection → token.
           Sequence length = mz_max / patch_size (e.g. 1000/50 = 20 tokens).

  sparse — Peak-token embedding.
           Only non-zero bins are tokenised.  m/z index → learned embedding,
           scaled by intensity.  Variable-length sequences padded per batch.
           Max sequence length capped at config.max_peaks.

Both variants prepend a [CLS] token; its output is the spectrum representation
passed to the classification or pretraining head.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class SpectrumConfig:
    input_type: Literal["dense", "sparse"] = "dense"
    mz_max: int = 1000
    patch_size: int = 50       # dense mode: bins per patch
    max_peaks: int = 200       # sparse mode: max tokens (excl. CLS)
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    ffn_dim: int = 1024
    dropout: float = 0.1
    num_classes: int = 0       # set >0 for downstream classification head


# ---------------------------------------------------------------------------
# Embedding layers
# ---------------------------------------------------------------------------

class DensePatchEmbedding(nn.Module):
    """Linear projection of fixed-size m/z patches."""

    def __init__(self, config: SpectrumConfig) -> None:
        super().__init__()
        assert config.mz_max % config.patch_size == 0, (
            f"mz_max ({config.mz_max}) must be divisible by patch_size ({config.patch_size})"
        )
        self.patch_size = config.patch_size
        self.n_patches = config.mz_max // config.patch_size
        self.proj = nn.Linear(config.patch_size, config.hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, mz_max]
        B = x.shape[0]
        x = x.view(B, self.n_patches, self.patch_size)  # [B, n_patches, patch_size]
        return self.proj(x)                              # [B, n_patches, hidden_dim]


class SparsePeakEmbedding(nn.Module):
    """
    Embed non-zero peaks as (m/z position embedding) × intensity.

    Returns:
      tokens : [B, max_peaks, hidden_dim]  (padded)
      pad_mask: [B, max_peaks]  True = padding position (for attention mask)
    """

    def __init__(self, config: SpectrumConfig) -> None:
        super().__init__()
        self.max_peaks = config.max_peaks
        self.mz_embed = nn.Embedding(config.mz_max, config.hidden_dim, padding_idx=0)
        self.intensity_proj = nn.Linear(1, config.hidden_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # x: [B, mz_max]
        B, mz_max = x.shape
        tokens = torch.zeros(B, self.max_peaks, self.mz_embed.embedding_dim, device=x.device)
        pad_mask = torch.ones(B, self.max_peaks, dtype=torch.bool, device=x.device)

        for i in range(B):
            nz_idx = x[i].nonzero(as_tuple=True)[0]  # non-zero positions
            nz_int = x[i][nz_idx]

            # Keep top-max_peaks by intensity if truncation needed
            if len(nz_idx) > self.max_peaks:
                top_k = nz_int.topk(self.max_peaks)
                nz_idx = nz_idx[top_k.indices]
                nz_int = top_k.values

            n = len(nz_idx)
            if n == 0:
                continue

            mz_emb = self.mz_embed(nz_idx)                   # [n, hidden_dim]
            int_emb = self.intensity_proj(nz_int.unsqueeze(-1))  # [n, hidden_dim]
            tokens[i, :n] = mz_emb + int_emb
            pad_mask[i, :n] = False  # not padding

        return tokens, pad_mask


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class SpectrumEncoder(nn.Module):
    """
    Transformer encoder for mass spectra.

    forward(x) → cls_output: [B, hidden_dim]

    Also returns all token outputs when return_all_tokens=True (used by MSM head).
    """

    def __init__(self, config: SpectrumConfig) -> None:
        super().__init__()
        self.config = config

        if config.input_type == "dense":
            self.embedding = DensePatchEmbedding(config)
            n_tokens = config.mz_max // config.patch_size
        else:
            self.embedding = SparsePeakEmbedding(config)
            n_tokens = config.max_peaks

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        self.pos_embed = nn.Embedding(n_tokens + 1, config.hidden_dim)  # +1 for CLS

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,  # pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(config.hidden_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, x: Tensor, return_all_tokens: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        B = x.shape[0]

        if self.config.input_type == "dense":
            tokens = self.embedding(x)          # [B, n_patches, hidden_dim]
            src_key_padding_mask = None
        else:
            tokens, pad_mask = self.embedding(x)  # [B, max_peaks, hidden_dim]
            src_key_padding_mask = torch.cat(
                [torch.zeros(B, 1, dtype=torch.bool, device=x.device), pad_mask], dim=1
            )  # [B, 1+max_peaks]

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)               # [B, 1, hidden_dim]
        tokens = torch.cat([cls, tokens], dim=1)              # [B, 1+n_tokens, hidden_dim]

        # Add positional embeddings
        seq_len = tokens.shape[1]
        pos_ids = torch.arange(seq_len, device=x.device)
        tokens = tokens + self.pos_embed(pos_ids).unsqueeze(0)

        # Transformer
        if self.config.input_type == "dense":
            out = self.transformer(tokens)
        else:
            out = self.transformer(tokens, src_key_padding_mask=src_key_padding_mask)

        out = self.norm(out)

        cls_output = out[:, 0]  # [B, hidden_dim]

        if return_all_tokens:
            return cls_output, out  # out: [B, 1+n_tokens, hidden_dim]
        return cls_output


# ---------------------------------------------------------------------------
# Classification head (for fine-tuning)
# ---------------------------------------------------------------------------

class SpectrumClassifier(nn.Module):
    """
    Encoder + linear classification head for downstream fine-tuning.

    Supports both full fine-tuning and linear probe (freeze_encoder=True).
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

    def forward(self, x: Tensor) -> Tensor:
        if self.freeze_encoder:
            with torch.no_grad():
                cls = self.encoder(x)
        else:
            cls = self.encoder(x)
        return self.head(cls)

    def load_pretrained_encoder(self, checkpoint_path: str) -> None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        encoder_state = {
            k.removeprefix("encoder."): v
            for k, v in ckpt["model_state"].items()
            if k.startswith("encoder.")
        }
        missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")
        print(f"Loaded pretrained encoder from {checkpoint_path}")
