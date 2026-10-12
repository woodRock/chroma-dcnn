"""Shared fixtures: tiny synthetic chromatograms so tests run in seconds on CPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

# Small model dimensions — every code path is exercised, just at toy scale.
N_BINS = 12
MZ_MAX = 16
CHANNELS = 8
N_CLASSES = 3


@pytest.fixture(autouse=True)
def _seed_everything():
    np.random.seed(0)
    torch.manual_seed(0)


def make_chroma(label: int, rng: np.random.Generator) -> np.ndarray:
    """Class-dependent chromatogram: a peak at an RT bin and m/z that encode the label."""
    chroma = rng.random((N_BINS, MZ_MAX)).astype(np.float32) * 0.05
    chroma[2 + 3 * label, 2 + 4 * label] += 1.0
    return chroma


def write_chroma_dataset(
    root: Path,
    n_per_class: int = 8,
    n_classes: int = N_CLASSES,
    seed: int = 0,
) -> tuple[list[Path], np.ndarray, np.ndarray]:
    """Write .npz chromatograms; returns (paths, labels, groups)."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    paths, labels, groups = [], [], []
    for c in range(n_classes):
        for i in range(n_per_class):
            p = root / f"c{c}_s{i}.npz"
            np.savez_compressed(p, chroma=make_chroma(c, rng))
            paths.append(p)
            labels.append(c)
            groups.append(f"g{c}_{i // 2}")  # pairs of samples share a group
    return paths, np.array(labels, dtype=np.int64), np.array(groups)


@pytest.fixture
def chroma_dataset(tmp_path):
    return write_chroma_dataset(tmp_path / "chroma")


@pytest.fixture
def finetune_cfg() -> dict:
    return {
        "task": {
            "num_classes": N_CLASSES,
            "cv_strategy": "kfold",
            "cv_folds": 3,
            "cv_seeds": [0],
        },
        "model": {"mz_max": MZ_MAX, "cnn_channels": CHANNELS, "kernel_size": 3, "dropout": 0.0},
        "finetuning": {
            "epochs": 3,
            "batch_size": 4,
            "lr": 1e-2,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "early_stopping_patience": 2,
            "val_fraction": 0.25,
        },
        "pretrained_checkpoints": {},
    }


@pytest.fixture
def pretrain_cfg(tmp_path) -> dict:
    return {
        "model": {
            "mz_max": MZ_MAX, "n_bins": N_BINS, "cnn_channels": CHANNELS,
            "kernel_size": 3, "dropout": 0.0,
        },
        "pretraining": {"n_iterations": 3, "batch_size": 2, "lr": 1e-3, "log_every": 2},
        "data": {"n_compounds_range": [1, 3], "sigma_range": [1.0, 2.0]},
        "output": {"checkpoint_dir": str(tmp_path / "ckpt")},
    }


@pytest.fixture
def spectra() -> np.ndarray:
    rng = np.random.default_rng(0)
    s = rng.random((20, MZ_MAX)).astype(np.float32)
    return s / np.linalg.norm(s, axis=1, keepdims=True)
