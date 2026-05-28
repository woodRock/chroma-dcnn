"""
Fine-tuning and evaluation for ChromatogramCNN on uniform-RT-binned GC-MS data.

Two conditions:

  from_scratch  — random Linear(mz_max→cnn_channels) per bin, trained end-to-end.
                  No pretrained weights.  Fast (~30 s / 200-epoch run) because the
                  per-bin encoder is a single linear layer, not a transformer.

  frozen_embed  — pretrained DensePatchEmbedding (frozen), pre-encoded once before
                  the CV loop.  Only to_cnn + CNN + head are trained.  Very fast
                  training because the encoder never runs during training.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from msformer.data.datasets import ChromaDataset
from msformer.models.chroma_cnn import ChromaCNNConfig, ChromatogramCNN
from msformer.models.encoder import SpectrumEncoder

ConditionName = Literal["from_scratch", "chroma_pretrain", "frozen_embed"]


# ---------------------------------------------------------------------------
# Pre-encoding (frozen_embed only)
# ---------------------------------------------------------------------------

class EncodedChromaDataset(Dataset):
    """Pre-encoded RT-bin embeddings + labels."""

    def __init__(self, embeddings: torch.Tensor, labels: np.ndarray) -> None:
        self.embeddings = embeddings                     # [N, N_bins, hidden_dim]
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]


@torch.no_grad()
def _pre_encode_chroma(
    encoder: SpectrumEncoder,
    paths: list[Path],
    device: torch.device,
    batch_size: int = 16,
) -> torch.Tensor:
    """
    Max-pool DensePatchEmbedding over all samples once.

    Returns [N, N_bins, hidden_dim] on CPU.
    """
    dummy_y = np.zeros(len(paths), dtype=np.int64)
    ds = ChromaDataset(paths, dummy_y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_emb = []
    encoder.eval()
    for chroma, _ in loader:
        B, N, mz = chroma.shape
        flat = chroma.reshape(B * N, mz).to(device)
        patches = encoder.embedding(flat)                # [B*N, n_patches, hidden_dim]
        emb = patches.max(dim=1).values.view(B, N, -1).cpu()
        all_emb.append(emb)

    return torch.cat(all_emb, dim=0)                    # [N_total, N_bins, hidden_dim]


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def _build_model(
    config: ChromaCNNConfig,
    condition: ConditionName,
    msm_ckpt: str | None,
    chroma_pretrain_ckpt: str | None = None,
) -> ChromatogramCNN:
    # chroma_pretrain uses the same architecture as from_scratch but with pretrained weights
    arch = "from_scratch" if condition in ("from_scratch", "chroma_pretrain") else condition
    model = ChromatogramCNN(config, condition=arch)
    if condition == "frozen_embed" and msm_ckpt:
        model.load_pretrained_encoder(msm_ckpt)
    elif condition == "chroma_pretrain" and chroma_pretrain_ckpt:
        model.load_pretrained_chroma_encoder(chroma_pretrain_ckpt)
    return model


# ---------------------------------------------------------------------------
# Shared training helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_loss(
    fwd: Callable[[torch.Tensor], torch.Tensor],
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    total = 0.0
    for x, y in loader:
        total += criterion(fwd(x.to(device)), y.to(device)).item()
    return total / max(len(loader), 1)


@torch.no_grad()
def _compute_metrics(
    fwd: Callable[[torch.Tensor], torch.Tensor],
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    from sklearn.metrics import balanced_accuracy_score, f1_score
    all_preds, all_labels = [], []
    for x, y in loader:
        preds = fwd(x.to(device)).argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())
    return {
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
        "macro_f1":          f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }


def _class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    classes, counts = np.unique(labels, return_counts=True)
    w = np.zeros(num_classes, dtype=np.float32)
    for c, n in zip(classes, counts):
        w[c] = len(labels) / (len(classes) * n)
    return torch.tensor(w)


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def _train_scratch_fold(
    model: ChromatogramCNN,
    train_paths: list[Path],
    val_paths: list[Path],
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    cfg: dict,
    device: torch.device,
) -> dict[str, float]:
    """Train from_scratch end-to-end on raw chroma files."""
    tcfg = cfg["finetuning"]
    epochs     = tcfg["epochs"]
    batch_size = min(tcfg.get("batch_size", 16), len(train_paths))

    train_ds = ChromaDataset(train_paths, train_labels)
    val_ds   = ChromaDataset(val_paths,   val_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    lr = tcfg.get("lr_scratch", tcfg["lr"])
    wd = tcfg.get("weight_decay", 0.01)
    params = [p for p in model.parameters() if p.requires_grad]
    opt   = AdamW(params, lr=lr, weight_decay=wd)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

    criterion = nn.CrossEntropyLoss(
        weight=_class_weights(train_labels, cfg["task"]["num_classes"]).to(device)
    )

    best_val_loss = float("inf")
    best_state    = copy.deepcopy(model.state_dict())
    stale, patience = 0, tcfg.get("early_stopping_patience", 20)

    for epoch in range(1, epochs + 1):
        model.train()
        for chroma, y in train_loader:
            chroma, y = chroma.to(device), y.to(device)
            opt.zero_grad()
            criterion(model(chroma), y).backward()
            nn.utils.clip_grad_norm_(params, tcfg.get("grad_clip", 1.0))
            opt.step()
        sched.step()

        model.eval()
        val_loss = _eval_loss(model, val_loader, criterion, device)
        if val_loss < best_val_loss:
            best_val_loss, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return _compute_metrics(model, val_loader, device)


def _train_frozen_fold(
    model: ChromatogramCNN,
    train_emb: torch.Tensor,
    train_labels: np.ndarray,
    val_emb: torch.Tensor,
    val_labels: np.ndarray,
    cfg: dict,
    device: torch.device,
) -> dict[str, float]:
    """Train to_cnn + CNN + head on pre-encoded embeddings."""
    tcfg       = cfg["frozen_embed"]
    epochs     = tcfg["epochs"]
    batch_size = min(tcfg.get("batch_size", 16), len(train_emb))

    train_ds = EncodedChromaDataset(train_emb, train_labels)
    val_ds   = EncodedChromaDataset(val_emb,   val_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    lr = tcfg["lr"]
    wd = tcfg.get("weight_decay", 0.01)
    params = [p for p in model.parameters() if p.requires_grad]
    opt   = AdamW(params, lr=lr, weight_decay=wd)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

    criterion = nn.CrossEntropyLoss(
        weight=_class_weights(train_labels, cfg["task"]["num_classes"]).to(device)
    )

    fwd = lambda emb: model.forward_from_embeddings(emb)  # noqa: E731

    best_val_loss = float("inf")
    best_state    = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        for emb, y in train_loader:
            emb, y = emb.to(device), y.to(device)
            opt.zero_grad()
            criterion(model.forward_from_embeddings(emb), y).backward()
            nn.utils.clip_grad_norm_(params, tcfg.get("grad_clip", 1.0))
            opt.step()
        sched.step()

        val_loss = _eval_loss(fwd, val_loader, criterion, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    return _compute_metrics(fwd, val_loader, device)


# ---------------------------------------------------------------------------
# Finetuner
# ---------------------------------------------------------------------------

class ChromaFinetuner:
    """
    Cross-validated evaluation of ChromatogramCNN.

    Parameters
    ----------
    config    : task YAML config dict
    npz_paths : list of Path, one per sample, pointing to chroma .npz files
    y         : int64 labels [N]
    """

    def __init__(self, config: dict, npz_paths: list[Path], y: np.ndarray) -> None:
        self.cfg   = config
        self.paths = npz_paths
        self.y     = y

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        m = config["model"]
        self.model_config = ChromaCNNConfig(
            mz_max          = m["mz_max"],
            patch_size      = m["patch_size"],
            hidden_dim      = m["hidden_dim"],
            num_layers      = m.get("num_layers", 6),
            num_heads       = m.get("num_heads", 8),
            ffn_dim         = m.get("ffn_dim", 1024),
            encoder_dropout = m.get("encoder_dropout", 0.1),
            cnn_channels    = m.get("cnn_channels", 128),
            kernel_size     = m.get("kernel_size", 7),
            num_classes     = config["task"]["num_classes"],
            dropout         = m.get("dropout", 0.3),
        )
        self.msm_ckpt            = config.get("pretrained_checkpoints", {}).get("msm")
        self.chroma_pretrain_ckpt = config.get("pretrained_checkpoints", {}).get("chroma_pretrain")

    def evaluate_condition(
        self,
        condition: ConditionName,
        seeds: list[int] | None = None,
        all_emb: torch.Tensor | None = None,
    ) -> dict[str, list[float]]:
        if seeds is None:
            seeds = self.cfg["task"].get("cv_seeds", list(range(10)))

        all_ba, all_f1 = [], []

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            skf = StratifiedKFold(
                n_splits=self.cfg["task"].get("cv_folds", 5),
                shuffle=True, random_state=seed,
            )
            for train_idx, val_idx in skf.split(self.paths, self.y):
                model = _build_model(
                    self.model_config, condition,
                    self.msm_ckpt, self.chroma_pretrain_ckpt,
                ).to(self.device)

                if condition == "frozen_embed":
                    metrics = _train_frozen_fold(
                        model,
                        all_emb[train_idx], self.y[train_idx],
                        all_emb[val_idx],   self.y[val_idx],
                        self.cfg, self.device,
                    )
                else:
                    train_paths = [self.paths[i] for i in train_idx]
                    val_paths   = [self.paths[i] for i in val_idx]
                    metrics = _train_scratch_fold(
                        model, train_paths, val_paths,
                        self.y[train_idx], self.y[val_idx],
                        self.cfg, self.device,
                    )

                all_ba.append(metrics["balanced_accuracy"])
                all_f1.append(metrics["macro_f1"])

        return {"balanced_accuracy": all_ba, "macro_f1": all_f1}

    def run_all_conditions(self, seeds: list[int] | None = None) -> dict[str, dict]:
        # Pre-encode once for frozen_embed (used across all seeds/folds)
        print("  Pre-encoding all RT bins with frozen encoder (once)...")
        _tmp = _build_model(self.model_config, "frozen_embed", self.msm_ckpt).to(self.device)
        all_emb = _pre_encode_chroma(_tmp.encoder, self.paths, self.device)
        del _tmp
        print(f"  Done — embeddings shape: {list(all_emb.shape)}")

        conditions = ["from_scratch", "frozen_embed"]
        if self.chroma_pretrain_ckpt:
            conditions.insert(1, "chroma_pretrain")

        results = {}
        for cond in conditions:
            print(f"\n--- Condition: {cond} ---")
            results[cond] = self.evaluate_condition(
                cond, seeds,
                all_emb=all_emb if cond == "frozen_embed" else None,
            )
            ba = results[cond]["balanced_accuracy"]
            print(f"  balanced_accuracy: {np.mean(ba):.3f} ± {np.std(ba):.3f}"
                  f"  (n={len(ba)} fold×seed runs)")
        return results
