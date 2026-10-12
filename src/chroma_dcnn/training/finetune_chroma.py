"""
Fine-tuning and evaluation for ChromatogramCNN on uniform-RT-binned GC-MS data.

Two conditions:

  from_scratch   — random Linear(mz_max→cnn_channels) per bin, trained end-to-end.

  chroma_pretrain — same architecture, weights initialised from next-frame
                    prediction pretraining on synthetic GC-MS (MoNA/MassBank).

Evaluation protocol (nested hold-out within each outer CV fold):
  outer fold → train / test
  train      → fit / val   (stratified, grouped when groups are given)

The val split drives checkpoint selection and early stopping.  The test fold
is only used once, after training, to compute the reported metrics.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from chroma_dcnn.data.datasets import ChromaDataset
from chroma_dcnn.models.chroma_cnn import ChromaCNNConfig, ChromatogramCNN

ConditionName = Literal["from_scratch", "chroma_pretrain"]


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def _build_model(
    config: ChromaCNNConfig,
    condition: ConditionName,
    chroma_pretrain_ckpt: str | None = None,
) -> ChromatogramCNN:
    model = ChromatogramCNN(config, condition="from_scratch")
    if condition == "chroma_pretrain" and chroma_pretrain_ckpt:
        model.load_pretrained_chroma_encoder(chroma_pretrain_ckpt)
    return model


# ---------------------------------------------------------------------------
# Shared training helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_loss(model, loader, criterion, device):
    total = 0.0
    for x, y in loader:
        total += criterion(model(x.to(device)), y.to(device)).item()
    return total / max(len(loader), 1)


@torch.no_grad()
def _compute_metrics(model, loader, device) -> dict[str, float]:
    from sklearn.metrics import balanced_accuracy_score, f1_score
    all_preds, all_labels = [], []
    for x, y in loader:
        preds = model(x.to(device)).argmax(dim=-1).cpu().numpy()
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
# Training loop
# ---------------------------------------------------------------------------

def _inner_val_split(
    y: np.ndarray,
    val_fraction: float,
    seed: int,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a training fold into (fit, val) index arrays, relative to ``y``.

    Stratified by class.  When ``groups`` is given, no group is split across
    fit and val (StratifiedGroupKFold with ~1/val_fraction folds, first fold).
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    idx = np.arange(len(y))
    if groups is not None:
        n_splits = max(2, round(1.0 / val_fraction))
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fit_idx, val_idx = next(splitter.split(idx, y, groups=groups))
    else:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        fit_idx, val_idx = next(splitter.split(idx, y))
    return fit_idx, val_idx


def _train_fold(
    model: ChromatogramCNN,
    train_paths: list[Path],
    val_paths: list[Path],
    test_paths: list[Path],
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    cfg: dict,
    device: torch.device,
    lr: float | None = None,
) -> dict[str, float]:
    """
    Train on ``train``, select the checkpoint / early-stop on ``val``, and
    report metrics on ``test``.  ``test`` must be disjoint from ``val``.
    """
    tcfg = cfg["finetuning"]
    epochs     = tcfg["epochs"]
    batch_size = min(tcfg.get("batch_size", 16), len(train_paths))

    train_ds = ChromaDataset(train_paths, train_labels)
    val_ds   = ChromaDataset(val_paths,   val_labels)
    test_ds  = ChromaDataset(test_paths,  test_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    if lr is None:
        lr = tcfg.get("lr", 1e-3)
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

    for _ in range(1, epochs + 1):
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
    return _compute_metrics(model, test_loader, device)


# ---------------------------------------------------------------------------
# Finetuner
# ---------------------------------------------------------------------------

class ChromaFinetuner:
    def __init__(
        self,
        config: dict,
        npz_paths: list[Path],
        y: np.ndarray,
        device: str | None = None,
        groups: np.ndarray | None = None,
    ) -> None:
        self.cfg   = config
        self.paths = npz_paths
        self.y     = y

        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        m = config["model"]
        self.model_config = ChromaCNNConfig(
            mz_max       = m["mz_max"],
            cnn_channels = m.get("cnn_channels", 128),
            kernel_size  = m.get("kernel_size", 7),
            num_classes  = config["task"]["num_classes"],
            dropout      = m.get("dropout", 0.3),
        )
        self.chroma_pretrain_ckpt = config.get("pretrained_checkpoints", {}).get("chroma_pretrain")
        self.groups = np.asarray(groups) if groups is not None else None
        self.cv_strategy = config["task"].get("cv_strategy", "kfold")

    def evaluate_condition(
        self,
        condition: ConditionName,
        seeds: list[int] | None = None,
    ) -> dict[str, list[float]]:
        if seeds is None:
            seeds = self.cfg["task"].get("cv_seeds", list(range(10)))

        tcfg = self.cfg["finetuning"]
        if condition == "from_scratch":
            lr = tcfg.get("lr_scratch", tcfg.get("lr", 1e-3))
        else:
            lr = tcfg.get("lr", 1e-3)

        all_ba, all_f1 = [], []
        n_folds = self.cfg["task"].get("cv_folds", 5)
        val_fraction = tcfg.get("val_fraction", 0.2)
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if self.cv_strategy == "grouped" and self.groups is not None:
                splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
                split_iter = splitter.split(self.paths, self.y, groups=self.groups)
            else:
                splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
                split_iter = splitter.split(self.paths, self.y)
            for fold, (train_idx, test_idx) in enumerate(split_iter):
                train_groups = self.groups[train_idx] if self.groups is not None else None
                fit_rel, val_rel = _inner_val_split(
                    self.y[train_idx], val_fraction, seed=seed * 1000 + fold,
                    groups=train_groups if self.cv_strategy == "grouped" else None,
                )
                fit_idx, val_idx = train_idx[fit_rel], train_idx[val_rel]

                model = _build_model(
                    self.model_config, condition, self.chroma_pretrain_ckpt,
                ).to(self.device)
                metrics = _train_fold(
                    model,
                    [self.paths[i] for i in fit_idx],
                    [self.paths[i] for i in val_idx],
                    [self.paths[i] for i in test_idx],
                    self.y[fit_idx], self.y[val_idx], self.y[test_idx],
                    self.cfg, self.device, lr=lr,
                )
                all_ba.append(metrics["balanced_accuracy"])
                all_f1.append(metrics["macro_f1"])

        return {"balanced_accuracy": all_ba, "macro_f1": all_f1}

    def run_all_conditions(self, seeds: list[int] | None = None) -> dict[str, dict]:
        conditions: list[ConditionName] = ["from_scratch"]
        if self.chroma_pretrain_ckpt and Path(self.chroma_pretrain_ckpt).exists():
            conditions.append("chroma_pretrain")

        results = {}
        for cond in conditions:
            print(f"\n--- Condition: {cond} ---")
            results[cond] = self.evaluate_condition(cond, seeds)
            ba = results[cond]["balanced_accuracy"]
            print(f"  balanced_accuracy: {np.mean(ba):.3f} ± {np.std(ba):.3f}"
                  f"  (n={len(ba)} fold×seed runs)")
        return results
