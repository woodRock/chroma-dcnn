"""
Attention visualisation and Grad-CAM for spectrum inputs.

Two methods:
  attention_rollout  — Abnar & Zuidema (2020): propagate attention weights
                       through all layers to get a single [seq_len] map.
  grad_attention     — gradient × attention: highlight bins where gradient
                       and attention weight are both large.

For dense (patch) mode: attention is per-patch → expand to bin-level heatmap.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from msformer.models.encoder import SpectrumConfig, SpectrumEncoder


def _register_attention_hooks(
    encoder: SpectrumEncoder,
) -> tuple[list, list[nn.modules.activation.MultiheadAttention]]:
    """Register forward hooks to capture attention weights from all layers."""
    attention_maps: list[Tensor] = []
    handles = []

    for layer in encoder.transformer.layers:
        def hook(module, input, output, store=attention_maps):
            # TransformerEncoderLayer with need_weights=True
            # output is (attn_output, attn_weights) when need_weights=True
            if isinstance(output, tuple) and len(output) == 2:
                store.append(output[1].detach().cpu())

        # Patch: force need_weights=True
        orig_forward = layer.self_attn.forward

        def patched_forward(*args, orig=orig_forward, **kwargs):
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = True
            return orig(*args, **kwargs)

        layer.self_attn.forward = patched_forward
        handles.append(layer.self_attn.register_forward_hook(hook))

    return handles, attention_maps


def attention_rollout(
    encoder: SpectrumEncoder,
    spectrum: Tensor,
    discard_ratio: float = 0.9,
) -> np.ndarray:
    """
    Compute attention rollout map for a single spectrum.

    spectrum : [mz_max]  (no batch dim)
    Returns : [n_tokens] float array, index 0 = CLS (high = attended-to)
    """
    encoder.eval()
    handles, attn_maps = _register_attention_hooks(encoder)

    with torch.no_grad():
        encoder(spectrum.unsqueeze(0), return_all_tokens=False)

    for h in handles:
        h.remove()

    if not attn_maps:
        raise RuntimeError("No attention maps captured. Check hook registration.")

    # Stack: [n_layers, n_heads, seq_len, seq_len] — already head-averaged → [n_layers, seq_len, seq_len]
    attn = torch.stack(attn_maps, dim=0).squeeze(1)  # [n_layers, seq_len, seq_len]

    # Rollout
    seq_len = attn.shape[-1]
    rollout = torch.eye(seq_len)
    for layer_attn in attn:
        layer_attn = layer_attn + torch.eye(seq_len)  # residual connection
        layer_attn = layer_attn / layer_attn.sum(dim=-1, keepdim=True)
        rollout = rollout @ layer_attn

    # CLS token (row 0) attention to each token
    cls_attention = rollout[0, 1:].numpy()  # exclude CLS itself
    return cls_attention


def plot_attention_heatmap(
    spectrum: np.ndarray,
    attention: np.ndarray,
    config: SpectrumConfig,
    title: str = "",
    output_path: str | Path | None = None,
) -> None:
    """
    Plot spectrum + attention overlay.

    spectrum  : [mz_max] normalised intensities
    attention : [n_tokens] from attention_rollout (dense mode) or [n_peaks] (sparse)
    """
    mz_max = config.mz_max

    # Dense mode: expand patch attention to bin level
    if config.input_type == "dense":
        patch_size = config.patch_size
        n_patches = mz_max // patch_size
        assert len(attention) == n_patches, (
            f"Expected {n_patches} attention values, got {len(attention)}"
        )
        attention_per_bin = np.repeat(attention, patch_size)
    else:
        attention_per_bin = np.interp(
            np.linspace(0, 1, mz_max),
            np.linspace(0, 1, len(attention)),
            attention,
        )

    # Normalise attention to [0, 1] for overlay
    if attention_per_bin.max() > 0:
        attention_per_bin = attention_per_bin / attention_per_bin.max()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    mz_axis = np.arange(mz_max)
    ax1.bar(mz_axis, spectrum, width=1.0, color="steelblue", alpha=0.8)
    ax1.set_ylabel("Normalised intensity")
    ax1.set_title(title or "Spectrum + Attention Rollout")

    ax2.bar(mz_axis, attention_per_bin, width=1.0, color="tomato", alpha=0.8)
    ax2.set_xlabel("m/z (or surrogate axis)")
    ax2.set_ylabel("Attention weight")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {output_path}")
    else:
        plt.show()
    plt.close()


def batch_visualise(
    encoder: SpectrumEncoder,
    X: np.ndarray,
    y: np.ndarray,
    sample_ids: list[str],
    config: SpectrumConfig,
    output_dir: str | Path,
    n_per_class: int = 2,
) -> None:
    """
    Generate attention heatmaps for n_per_class representative samples per class.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = np.unique(y)
    for cls in classes:
        cls_indices = np.where(y == cls)[0][:n_per_class]
        for idx in cls_indices:
            spectrum_tensor = torch.from_numpy(X[idx])
            attn = attention_rollout(encoder, spectrum_tensor)
            plot_attention_heatmap(
                spectrum=X[idx],
                attention=attn,
                config=config,
                title=f"Class {cls} — {sample_ids[idx]}",
                output_path=output_dir / f"class{cls}_{sample_ids[idx]}_attn.png",
            )
