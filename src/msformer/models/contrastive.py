"""
Contrastive pretraining with NT-Xent loss (SimCLR-style).

Two independently augmented views of each spectrum are pushed together in
representation space while repelling representations of different spectra.

The projection head (encoder → projector → normalised embedding) follows
the SimCLR recipe; the projector is discarded at fine-tuning time.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from msformer.models.encoder import SpectrumConfig, SpectrumEncoder


class ProjectionHead(nn.Module):
    """
    2-layer MLP projection head: hidden_dim → hidden_dim → proj_dim.
    Used only during contrastive pretraining; discarded at fine-tuning.
    """

    def __init__(self, hidden_dim: int, proj_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(self.net(x), dim=-1)


def nt_xent_loss(z1: Tensor, z2: Tensor, temperature: float = 0.1) -> Tensor:
    """
    NT-Xent loss for a batch of paired representations.

    z1, z2 : [B, proj_dim]  already L2-normalised
    temperature : scalar τ

    The loss treats (z1[i], z2[i]) as a positive pair and all other
    combinations within the batch as negatives — including (z1[i], z1[j])
    and (z2[i], z2[j]) cross-terms.
    """
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # [2B, proj_dim]

    # Pairwise cosine similarity (already normalised → dot product)
    sim = torch.mm(z, z.T) / temperature  # [2B, 2B]

    # Mask self-similarity
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, float("-inf"))

    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat(
        [torch.arange(B, 2 * B), torch.arange(0, B)], dim=0
    ).to(z.device)

    loss = F.cross_entropy(sim, labels)
    return loss


def alignment_uniformity(z1: Tensor, z2: Tensor) -> tuple[Tensor, Tensor]:
    """
    Alignment (mean ||z1-z2||²) and uniformity (Gaussian potential on hypersphere).
    Wang & Isola (2020) metrics for monitoring contrastive representation quality.
    """
    align = (z1 - z2).pow(2).sum(dim=-1).mean()
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=-1)
    sq_dist = torch.cdist(z, z, p=2).pow(2)
    uniform = torch.log(torch.exp(-2.0 * sq_dist).mean())
    return align, uniform


class ContrastiveModel(nn.Module):
    """
    Encoder + projection head for NT-Xent pretraining.

    forward(batch) where batch = {'view1': [B, mz_max], 'view2': [B, mz_max]}
    returns (loss, alignment, uniformity)
    """

    def __init__(self, config: SpectrumConfig, proj_dim: int = 128, temperature: float = 0.1) -> None:
        super().__init__()
        self.config = config
        self.temperature = temperature
        self.encoder = SpectrumEncoder(config)
        self.projector = ProjectionHead(config.hidden_dim, proj_dim)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        v1, v2 = batch["view1"], batch["view2"]

        h1 = self.encoder(v1)   # [B, hidden_dim]
        h2 = self.encoder(v2)

        z1 = self.projector(h1)  # [B, proj_dim]  normalised
        z2 = self.projector(h2)

        loss = nt_xent_loss(z1, z2, self.temperature)
        align, uniform = alignment_uniformity(z1, z2)

        return loss, align, uniform

    @torch.no_grad()
    def encode(self, x: Tensor) -> Tensor:
        """Return CLS embedding (pre-projection) for downstream use."""
        return self.encoder(x)
