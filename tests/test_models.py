"""Model shapes, causality of the pretraining encoder, and weight transfer."""

from __future__ import annotations

import sys

import pytest
import torch

from chroma_dcnn.models import (
    ChromaCNNConfig,
    ChromaNextFramePredictor,
    ChromaPretrainConfig,
    ChromatogramCNN,
)
from chroma_dcnn.models.chroma_cnn import _AttentionPool1d, _ResBlock1d
from chroma_dcnn.models.chroma_pretrain import _CausalResBlock1d

from .conftest import CHANNELS, MZ_MAX, N_BINS


def _cnn_cfg(**kw):
    return ChromaCNNConfig(mz_max=MZ_MAX, cnn_channels=CHANNELS, kernel_size=3,
                           num_classes=3, dropout=0.0, **kw)


def _pre_cfg():
    return ChromaPretrainConfig(mz_max=MZ_MAX, n_bins=N_BINS, cnn_channels=CHANNELS,
                                kernel_size=3, dropout=0.0)


def test_chromatogram_cnn_output_shape():
    model = ChromatogramCNN(_cnn_cfg()).eval()
    out = model(torch.randn(5, N_BINS, MZ_MAX))
    assert out.shape == (5, 3)


def test_chromatogram_cnn_handles_variable_length():
    model = ChromatogramCNN(_cnn_cfg()).eval()
    assert model(torch.randn(2, 7, MZ_MAX)).shape == (2, 3)


@pytest.mark.parametrize("dilation", [1, 2, 4])
def test_resblock_preserves_shape(dilation):
    block = _ResBlock1d(CHANNELS, 7, dilation, 0.0).eval()
    x = torch.randn(2, CHANNELS, 20)
    assert block(x).shape == x.shape


def test_attention_pool_is_convex_combination():
    pool = _AttentionPool1d(4)
    x = torch.ones(2, 4, 10) * 3.0
    torch.testing.assert_close(pool(x), torch.full((2, 4), 3.0))


def test_causal_block_ignores_future():
    block = _CausalResBlock1d(CHANNELS, 3, 2, 0.0).eval()
    x = torch.randn(1, CHANNELS, 16)
    y1 = block(x)
    x2 = x.clone()
    x2[..., 10:] = torch.randn_like(x2[..., 10:])
    y2 = block(x2)
    torch.testing.assert_close(y1[..., :10], y2[..., :10])
    assert not torch.allclose(y1[..., 10:], y2[..., 10:])


def test_next_frame_predictor_shapes_and_target():
    model = ChromaNextFramePredictor(_pre_cfg()).eval()
    x = torch.randn(3, N_BINS, MZ_MAX)
    pred, target = model(x)
    assert pred.shape == target.shape == (3, N_BINS - 1, MZ_MAX)
    torch.testing.assert_close(target, x[:, 1:])


def test_next_frame_prediction_does_not_see_the_frame_it_predicts():
    """pred[t] (predicting frame t+1) must not depend on frames > t."""
    model = ChromaNextFramePredictor(_pre_cfg()).eval()
    x = torch.randn(1, N_BINS, MZ_MAX)
    pred1, _ = model(x)
    x2 = x.clone()
    x2[:, 6:] = torch.randn_like(x2[:, 6:])
    pred2, _ = model(x2)
    torch.testing.assert_close(pred1[:, :6], pred2[:, :6])


def test_pretrained_weights_transfer_to_classifier():
    pre = ChromaNextFramePredictor(_pre_cfg())
    clf = ChromatogramCNN(_cnn_cfg())
    clf.spec_proj.load_state_dict(pre.spec_proj.state_dict())
    clf.cnn.load_state_dict(pre.cnn.state_dict())
    torch.testing.assert_close(clf.cnn[0].conv1.weight, pre.cnn[0].conv1.weight)


def _save_ckpt(path, pre, with_cnn=True):
    ckpt = {"spec_proj_state": pre.spec_proj.state_dict()}
    if with_cnn:
        ckpt["cnn_state"] = pre.cnn.state_dict()
    torch.save(ckpt, path)


def test_load_pretrained_encoder(tmp_path, capsys):
    pre = ChromaNextFramePredictor(_pre_cfg())
    _save_ckpt(tmp_path / "c.pt", pre)
    clf = ChromatogramCNN(_cnn_cfg())
    clf.load_pretrained_chroma_encoder(str(tmp_path / "c.pt"))
    torch.testing.assert_close(clf.spec_proj.weight, pre.spec_proj.weight)
    torch.testing.assert_close(clf.cnn[2].conv2.weight, pre.cnn[2].conv2.weight)
    assert "Loaded pretrained CNN" in capsys.readouterr().out
    assert "msformer" in sys.modules  # legacy-name alias for old checkpoints


def test_load_pretrained_encoder_without_cnn_state(tmp_path, capsys):
    pre = ChromaNextFramePredictor(_pre_cfg())
    _save_ckpt(tmp_path / "c.pt", pre, with_cnn=False)
    clf = ChromatogramCNN(_cnn_cfg())
    before = clf.cnn[0].conv1.weight.clone()
    clf.load_pretrained_chroma_encoder(str(tmp_path / "c.pt"))
    torch.testing.assert_close(clf.spec_proj.weight, pre.spec_proj.weight)
    torch.testing.assert_close(clf.cnn[0].conv1.weight, before)
    assert "Loaded pretrained CNN" not in capsys.readouterr().out


def test_load_pretrained_encoder_rejects_mismatched_shape(tmp_path):
    pre = ChromaNextFramePredictor(_pre_cfg())
    _save_ckpt(tmp_path / "c.pt", pre)
    clf = ChromatogramCNN(ChromaCNNConfig(mz_max=MZ_MAX + 1, cnn_channels=CHANNELS,
                                          kernel_size=3))
    with pytest.raises(RuntimeError):
        clf.load_pretrained_chroma_encoder(str(tmp_path / "c.pt"))
