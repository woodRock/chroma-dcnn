"""Synthetic-batch generation and the pretraining loop."""

from __future__ import annotations

import re

import numpy as np
import pytest
import torch

from chroma_dcnn.training.pretrain_chroma import _make_batch, pretrain_chroma

from .conftest import MZ_MAX, N_BINS


def test_make_batch_shape_and_unit_norm(spectra):
    rng = np.random.default_rng(0)
    batch = _make_batch(spectra, 4, N_BINS, (2, 5), (1.0, 3.0), rng)
    assert batch.shape == (4, N_BINS, MZ_MAX) and batch.dtype == torch.float32
    norms = batch.norm(dim=-1)
    assert torch.all((norms - 1).abs() < 1e-4)
    assert torch.all(batch >= 0)


def test_make_batch_zero_spectra_stay_finite():
    batch = _make_batch(np.zeros((3, MZ_MAX), np.float32), 2, N_BINS, (1, 1), (1.0, 1.0),
                        np.random.default_rng(0))
    assert torch.isfinite(batch).all() and batch.abs().sum() == 0


def test_make_batch_is_reproducible(spectra):
    a = _make_batch(spectra, 2, N_BINS, (1, 3), (1.0, 2.0), np.random.default_rng(5))
    b = _make_batch(spectra, 2, N_BINS, (1, 3), (1.0, 2.0), np.random.default_rng(5))
    torch.testing.assert_close(a, b)


def test_pretrain_saves_transferable_checkpoint(pretrain_cfg, spectra):
    path = pretrain_chroma(pretrain_cfg, spectra, torch.device("cpu"))
    ckpt = torch.load(path, weights_only=False)
    assert set(ckpt) == {"spec_proj_state", "cnn_state", "config", "n_iterations", "final_loss"}
    assert ckpt["n_iterations"] == 3
    assert ckpt["spec_proj_state"]["weight"].shape == (pretrain_cfg["model"]["cnn_channels"],
                                                       MZ_MAX)
    assert np.isfinite(ckpt["final_loss"])


def test_pretrain_logged_loss_averages_partial_window(monkeypatch, pretrain_cfg, spectra, capsys):
    """With 3 iterations and log_every=2, the last window holds 1 iteration, not 2."""
    import sys

    pc = sys.modules["chroma_dcnn.training.pretrain_chroma"]  # package re-exports the function
    losses = iter([0.4, 0.2, 0.9])
    real_mse = torch.nn.functional.mse_loss

    def fake_mse(pred, target):
        real = real_mse(pred, target)
        return real - real.detach() + next(losses)

    monkeypatch.setattr(pc.nn.functional, "mse_loss", fake_mse)
    pretrain_chroma(pretrain_cfg, spectra, torch.device("cpu"))
    logged = [float(x) for x in re.findall(r"iter .* loss=([\d.]+)", capsys.readouterr().out)]
    assert logged == pytest.approx([0.3, 0.9], abs=1e-5)
