"""
PyTorch Dataset classes for pretraining and downstream fine-tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Base pretraining dataset
# ---------------------------------------------------------------------------

class PretrainingDataset(Dataset):
    """
    Loads normalised spectra from HDF5.  Returns raw [mz_max] float32 tensors.
    """

    def __init__(
        self,
        h5_path: str,
        split: Literal["train", "val"] = "train",
    ) -> None:
        super().__init__()
        with h5py.File(h5_path, "r") as f:
            split_arr = np.array([s.decode() for s in f["split"][:]])
            mask = split_arr == split
            self.spectra = torch.from_numpy(f["spectra"][:][mask])
            self.inchikeys = [s.decode() for s in f["inchikeys"][:][mask]]

        self.mz_max = self.spectra.shape[1]
        print(f"PretrainingDataset [{split}]: {len(self)} spectra, mz_max={self.mz_max}")

    def __len__(self) -> int:
        return len(self.spectra)

    def __getitem__(self, idx: int) -> Tensor:
        return self.spectra[idx]


# ---------------------------------------------------------------------------
# MSM (Masked Spectra Modelling) dataset
# ---------------------------------------------------------------------------

class MSMDataset(Dataset):
    """
    Wraps PretrainingDataset and applies random masking on-the-fly.

    Returns dict with:
      spectrum   : [mz_max]  masked input
      targets    : [mz_max]  original intensities (zero at unmasked positions)
      mask       : [mz_max]  bool, True = masked position
    """

    def __init__(self, base: PretrainingDataset, mask_ratio: float = 0.15) -> None:
        self.base = base
        self.mask_ratio = mask_ratio

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        spectrum = self.base[idx].clone()
        nonzero_indices = (spectrum > 0).nonzero(as_tuple=True)[0]

        n_mask = max(1, int(len(nonzero_indices) * self.mask_ratio))
        perm = torch.randperm(len(nonzero_indices))[:n_mask]
        mask_indices = nonzero_indices[perm]

        targets = torch.zeros_like(spectrum)
        targets[mask_indices] = spectrum[mask_indices]

        mask = torch.zeros(len(spectrum), dtype=torch.bool)
        mask[mask_indices] = True

        spectrum[mask_indices] = 0.0  # zero-out masked positions

        return {"spectrum": spectrum, "targets": targets, "mask": mask}


# ---------------------------------------------------------------------------
# Contrastive dataset
# ---------------------------------------------------------------------------

@dataclass
class AugmentationConfig:
    intensity_jitter: float = 0.1
    low_mass_noise_sigma: float = 0.02
    low_mass_cutoff: int = 50
    bin_dropout_rate: float = 0.05


class ContrastiveDataset(Dataset):
    """
    Returns two independently augmented views of each spectrum.

    Returns dict with:
      view1 : [mz_max]
      view2 : [mz_max]
    """

    def __init__(
        self, base: PretrainingDataset, aug_config: AugmentationConfig | None = None
    ) -> None:
        self.base = base
        self.aug = aug_config or AugmentationConfig()

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        spectrum = self.base[idx]
        return {"view1": self._augment(spectrum), "view2": self._augment(spectrum)}

    def _augment(self, spectrum: Tensor) -> Tensor:
        x = spectrum.clone()
        nonzero = x > 0

        # 1. intensity jitter on non-zero bins
        scale = 1.0 + (torch.rand(x.shape) * 2 - 1) * self.aug.intensity_jitter
        x = torch.where(nonzero, x * scale, x)

        # 2. low-mass noise injection (m/z < cutoff)
        noise = torch.randn(self.aug.low_mass_cutoff) * self.aug.low_mass_noise_sigma
        x[: self.aug.low_mass_cutoff] = (x[: self.aug.low_mass_cutoff] + noise).clamp(0)

        # 3. random bin dropout
        drop_mask = torch.rand(x.shape) < self.aug.bin_dropout_rate
        x = torch.where(drop_mask, torch.zeros_like(x), x)

        # Re-normalise after augmentation to keep unit norm
        norm = x.norm()
        if norm > 1e-8:
            x = x / norm

        return x.float()


# ---------------------------------------------------------------------------
# Downstream dataset (generic)
# ---------------------------------------------------------------------------

class DownstreamDataset(Dataset):
    """
    Simple (X, y) dataset for fine-tuning / linear probe evaluation.

    X : np.ndarray [N, mz_max]
    y : np.ndarray [N] int labels
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.X[idx], self.y[idx]
