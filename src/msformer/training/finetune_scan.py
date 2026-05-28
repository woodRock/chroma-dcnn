"""
Fine-tuning and linear probe evaluation for per-scan GC-MS classification.

Mirrors finetune.py but uses ScanPoolClassifier and ScanDataset.
Each sample is a set of K pre-selected GC-MS scans + TIC weights.

Three conditions:
  from_scratch      — random init, trained on downstream data only
  msm_finetune      — pretrained weights, all layers fine-tuned
  linear_probe_msm  — pretrained encoder frozen, only classification head trained
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from msformer.data.datasets import ScanDataset
from msformer.models.encoder import SpectrumConfig, SpectrumEncoder
from msformer.models.scan_pool import EncodedScanPoolHead, ScanPoolClassifier


ConditionName = Literal["from_scratch", "msm_finetune", "linear_probe_msm"]


# ---------------------------------------------------------------------------
# Pre-encoding helpers for the linear-probe fast path
# ---------------------------------------------------------------------------

class EmbeddingDataset(torch.utils.data.Dataset):
    """Pre-encoded scan embeddings + TIC weights + labels."""

    def __init__(
        self, embeddings: torch.Tensor, tic_weights: torch.Tensor, labels: np.ndarray
    ) -> None:
        self.embeddings = embeddings
        self.tic_weights = tic_weights
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> tuple:
        return self.embeddings[idx], self.tic_weights[idx], self.labels[idx]


@torch.no_grad()
def _pre_encode_all(
    encoder: SpectrumEncoder,
    paths: list[Path],
    device: torch.device,
    batch_size: int = 32,
    max_scans: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run the frozen encoder over every sample once.

    Returns
    -------
    all_emb : [N, K, hidden_dim]  (on CPU)
    all_tic : [N, K]              (on CPU)
    """
    dummy_y = np.zeros(len(paths), dtype=np.int64)
    ds = ScanDataset(paths, dummy_y, max_scans=max_scans)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_emb, all_tic = [], []
    encoder.eval()
    for scans, tic_weights, _ in loader:
        B, K, mz = scans.shape
        flat = scans.reshape(B * K, mz).to(device)
        emb = encoder(flat).view(B, K, -1).cpu()
        all_emb.append(emb)
        all_tic.append(tic_weights)

    return torch.cat(all_emb, dim=0), torch.cat(all_tic, dim=0)


def _train_probe_fold(
    head: EncodedScanPoolHead,
    train_emb: torch.Tensor,
    train_tic: torch.Tensor,
    train_labels: np.ndarray,
    val_emb: torch.Tensor,
    val_tic: torch.Tensor,
    val_labels: np.ndarray,
    cfg: dict,
    device: torch.device,
) -> dict[str, float]:
    """Train the linear probe head on pre-encoded embeddings (no encoder cost)."""
    tcfg = cfg["linear_probe"]
    epochs = tcfg["epochs"]
    batch_size = min(tcfg.get("batch_size", 8), len(train_emb))

    train_ds = EmbeddingDataset(train_emb, train_tic, train_labels)
    val_ds = EmbeddingDataset(val_emb, val_tic, val_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    params = list(head.parameters())
    lr = tcfg["lr"]
    opt = AdamW(params, lr=lr, weight_decay=tcfg.get("weight_decay", 0.01))
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

    classes, counts = np.unique(train_labels, return_counts=True)
    weights = np.zeros(cfg["task"]["num_classes"], dtype=np.float32)
    for c, n in zip(classes, counts):
        weights[c] = len(train_labels) / (len(classes) * n)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))

    best_val_loss = float("inf")
    best_state = copy.deepcopy(head.state_dict())

    for epoch in range(1, epochs + 1):
        head.train()
        for emb, tic, y in train_loader:
            emb, tic, y = emb.to(device), tic.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(head(emb, tic), y)
            loss.backward()
            nn.utils.clip_grad_norm_(params, tcfg.get("grad_clip", 1.0))
            opt.step()
        sched.step()

        val_loss = _eval_loss(head, val_loader, criterion, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(head.state_dict())

    head.load_state_dict(best_state)
    return _compute_metrics(head, val_loader, device)


def _build_scan_classifier(
    config: SpectrumConfig,
    condition: ConditionName,
    msm_ckpt: str | None,
) -> ScanPoolClassifier:
    freeze = condition == "linear_probe_msm"
    clf = ScanPoolClassifier(config, freeze_encoder=freeze)
    if condition in ("msm_finetune", "linear_probe_msm") and msm_ckpt:
        clf.load_pretrained_encoder(msm_ckpt)
    return clf


def _train_one_fold(
    model: ScanPoolClassifier,
    train_paths: list[Path],
    val_paths: list[Path],
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    cfg: dict,
    device: torch.device,
    condition: ConditionName,
    max_scans: int | None = None,
) -> dict[str, float]:
    is_probe = condition == "linear_probe_msm"
    tcfg = cfg["linear_probe"] if is_probe else cfg["finetuning"]
    epochs = tcfg["epochs"]
    batch_size = min(tcfg.get("batch_size", 8), len(train_paths))

    train_ds = ScanDataset(train_paths, train_labels, max_scans=max_scans)
    val_ds = ScanDataset(val_paths, val_labels, max_scans=max_scans)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    lr = tcfg["lr"]
    if condition == "from_scratch" and not is_probe:
        lr = tcfg.get("lr_scratch", lr * 20)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(params, lr=lr, weight_decay=tcfg.get("weight_decay", 0.01))
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

    patience = tcfg.get("early_stopping_patience", 15) if not is_probe else 9999
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale = 0

    # Class-weighted loss to handle imbalanced BCO class
    classes, counts = np.unique(train_labels, return_counts=True)
    weights = np.zeros(cfg["task"]["num_classes"], dtype=np.float32)
    for c, n in zip(classes, counts):
        weights[c] = len(train_labels) / (len(classes) * n)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))

    for epoch in range(1, epochs + 1):
        model.train()
        for scans, tic_weights, y in train_loader:
            scans = scans.to(device)
            tic_weights = tic_weights.to(device)
            y = y.to(device)
            opt.zero_grad()
            logits = model(scans, tic_weights)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(params, tcfg.get("grad_clip", 1.0))
            opt.step()
        sched.step()

        val_loss = _eval_loss(model, val_loader, criterion, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    return _compute_metrics(model, val_loader, device)


@torch.no_grad()
def _eval_loss(model, loader, criterion, device) -> float:
    model.eval()
    total = 0.0
    for scans, tic_weights, y in loader:
        scans, tic_weights, y = scans.to(device), tic_weights.to(device), y.to(device)
        total += criterion(model(scans, tic_weights), y).item()
    return total / max(len(loader), 1)


@torch.no_grad()
def _compute_metrics(model, loader, device) -> dict[str, float]:
    from sklearn.metrics import balanced_accuracy_score, f1_score
    model.eval()
    all_preds, all_labels = [], []
    for scans, tic_weights, y in loader:
        scans, tic_weights = scans.to(device), tic_weights.to(device)
        preds = model(scans, tic_weights).argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())
    ba = balanced_accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return {"balanced_accuracy": ba, "macro_f1": f1}


class ScanFinetuner:
    """
    Cross-validated evaluation of ScanPoolClassifier on a per-scan GC-MS task.

    Parameters
    ----------
    config    : task YAML config dict
    npz_paths : list of Path, one per sample, pointing to precomputed scan npz files
    y         : int64 labels [N]
    """

    def __init__(self, config: dict, npz_paths: list[Path], y: np.ndarray) -> None:
        self.cfg = config
        self.paths = npz_paths
        self.y = y

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        m = config["model"]
        self.model_config = SpectrumConfig(
            input_type=m["input_type"],
            mz_max=m["mz_max"],
            patch_size=m["patch_size"],
            max_peaks=m.get("max_peaks", 200),
            hidden_dim=m["hidden_dim"],
            num_layers=m["num_layers"],
            num_heads=m["num_heads"],
            ffn_dim=m.get("ffn_dim", m["hidden_dim"] * 4),
            dropout=m["dropout"],
            num_classes=config["task"]["num_classes"],
        )
        self.msm_ckpt = config.get("pretrained_checkpoints", {}).get("msm")
        n_splits = config["task"].get("cv_folds", 5)
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

    def evaluate_condition(
        self,
        condition: ConditionName,
        seeds: list[int] | None = None,
    ) -> dict[str, list[float]]:
        if seeds is None:
            seeds = self.cfg["task"].get("cv_seeds", list(range(10)))

        # For linear probe the encoder is frozen, so embeddings never change.
        # Encode all samples once before the seed/fold loops to avoid repeating
        # B*K transformer forward passes on every training epoch.
        if condition == "linear_probe_msm" and self.msm_ckpt:
            print("  Pre-encoding all scans with frozen encoder (once)...")
            _tmp = _build_scan_classifier(self.model_config, condition, self.msm_ckpt)
            _tmp = _tmp.to(self.device)
            all_emb, all_tic = _pre_encode_all(_tmp.encoder, self.paths, self.device)
            del _tmp
            preencoded: tuple | None = (all_emb, all_tic)
        else:
            preencoded = None

        all_ba, all_f1 = [], []

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            skf = StratifiedKFold(
                n_splits=self.cfg["task"].get("cv_folds", 5),
                shuffle=True,
                random_state=seed,
            )

            for train_idx, val_idx in skf.split(self.paths, self.y):
                if preencoded is not None:
                    emb_all, tic_all = preencoded
                    head = EncodedScanPoolHead(
                        self.model_config.hidden_dim, self.model_config.num_classes
                    ).to(self.device)
                    metrics = _train_probe_fold(
                        head,
                        emb_all[train_idx], tic_all[train_idx], self.y[train_idx],
                        emb_all[val_idx], tic_all[val_idx], self.y[val_idx],
                        self.cfg, self.device,
                    )
                else:
                    train_paths = [self.paths[i] for i in train_idx]
                    val_paths = [self.paths[i] for i in val_idx]
                    model = _build_scan_classifier(
                        self.model_config, condition, self.msm_ckpt
                    ).to(self.device)
                    metrics = _train_one_fold(
                        model, train_paths, val_paths,
                        self.y[train_idx], self.y[val_idx],
                        self.cfg, self.device, condition,
                    )

                all_ba.append(metrics["balanced_accuracy"])
                all_f1.append(metrics["macro_f1"])

        return {"balanced_accuracy": all_ba, "macro_f1": all_f1}

    def run_all_conditions(self, seeds: list[int] | None = None) -> dict[str, dict]:
        conditions: list[ConditionName] = [
            "from_scratch",
            "msm_finetune",
            "linear_probe_msm",
        ]
        results = {}
        for cond in conditions:
            print(f"\n--- Condition: {cond} ---")
            results[cond] = self.evaluate_condition(cond, seeds)
            ba = results[cond]["balanced_accuracy"]
            print(
                f"  balanced_accuracy: {np.mean(ba):.3f} ± {np.std(ba):.3f}"
                f"  (n={len(ba)} fold×seed runs)"
            )
        return results
