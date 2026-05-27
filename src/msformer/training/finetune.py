"""
Fine-tuning and linear probe evaluation for downstream classification tasks.

Implements all four conditions from the experimental protocol:
  A. from_scratch   — random init, trained on downstream data only
  B. msm_finetune   — pretrained (MSM) weights, all layers fine-tuned
  C. cont_finetune  — pretrained (contrastive) weights, all layers fine-tuned
  D. linear_probe   — pretrained weights frozen, only classification head trained

Evaluation uses either stratified k-fold CV or leave-one-out CV (LOSO),
each repeated over multiple seeds to support Mann-Whitney statistical testing.
"""

from __future__ import annotations

import copy
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

from msformer.data.datasets import DownstreamDataset
from msformer.models.encoder import SpectrumClassifier, SpectrumConfig


ConditionName = Literal["from_scratch", "msm_finetune", "cont_finetune", "linear_probe_msm", "linear_probe_cont"]


def _build_classifier(
    config: SpectrumConfig,
    condition: ConditionName,
    msm_ckpt: str | None,
    cont_ckpt: str | None,
) -> SpectrumClassifier:
    freeze = condition in ("linear_probe_msm", "linear_probe_cont")
    clf = SpectrumClassifier(config, freeze_encoder=freeze)

    if condition in ("msm_finetune", "linear_probe_msm") and msm_ckpt:
        clf.load_pretrained_encoder(msm_ckpt)
    elif condition in ("cont_finetune", "linear_probe_cont") and cont_ckpt:
        clf.load_pretrained_encoder(cont_ckpt)

    return clf


def _train_one_fold(
    model: SpectrumClassifier,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    dataset: DownstreamDataset,
    cfg: dict,
    device: torch.device,
    condition: ConditionName,
) -> dict[str, float]:
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    is_probe = condition in ("linear_probe_msm", "linear_probe_cont")
    tcfg = cfg["linear_probe"] if is_probe else cfg["finetuning"]
    epochs = tcfg["epochs"]
    batch_size = min(tcfg.get("batch_size", 16), len(train_idx))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(params, lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 0.01))
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=tcfg["lr"] * 0.01)

    patience = tcfg.get("early_stopping_patience", 10) if not is_probe else 9999
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale = 0

    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
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
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total += criterion(logits, y).item()
    return total / len(loader)


@torch.no_grad()
def _compute_metrics(model, loader, device) -> dict[str, float]:
    from sklearn.metrics import balanced_accuracy_score, f1_score

    model.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        preds = model(x).argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())

    ba = balanced_accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return {"balanced_accuracy": ba, "macro_f1": f1}


class Finetuner:
    """
    Runs cross-validated evaluation for a single condition and dataset.

    Parameters
    ----------
    config : dict
        Task-specific YAML config (loaded from finetune_*.yaml).
    X : np.ndarray [N, mz_max]
    y : np.ndarray [N] integer labels
    """

    def __init__(self, config: dict, X: np.ndarray, y: np.ndarray) -> None:
        self.cfg = config
        self.X = X
        self.y = y
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        model_cfg_d = config["model"]
        self.model_config = SpectrumConfig(
            input_type=model_cfg_d["input_type"],
            mz_max=model_cfg_d["mz_max"],
            patch_size=model_cfg_d["patch_size"],
            max_peaks=model_cfg_d.get("max_peaks", 200),
            hidden_dim=model_cfg_d["hidden_dim"],
            num_layers=model_cfg_d["num_layers"],
            num_heads=model_cfg_d["num_heads"],
            ffn_dim=model_cfg_d.get("ffn_dim", model_cfg_d["hidden_dim"] * 4),
            dropout=model_cfg_d["dropout"],
            num_classes=config["task"]["num_classes"],
        )

        self.msm_ckpt = config.get("pretrained_checkpoints", {}).get("msm")
        self.cont_ckpt = config.get("pretrained_checkpoints", {}).get("contrastive")

        cv_strategy = config["task"].get("cv_strategy", "kfold")
        if cv_strategy == "loso":
            self.cv = LeaveOneOut()
        else:
            n_splits = config["task"].get("cv_folds", 5)
            self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True)

        self.cv_strategy = cv_strategy

    def evaluate_condition(
        self,
        condition: ConditionName,
        seeds: list[int] | None = None,
    ) -> dict[str, list[float]]:
        """
        Run cross-validated evaluation over multiple seeds.

        Returns dict with keys 'balanced_accuracy' and 'macro_f1',
        each a flat list of length (n_seeds × n_folds).
        """
        if seeds is None:
            seeds = self.cfg["task"].get("cv_seeds", list(range(10)))

        dataset = DownstreamDataset(self.X, self.y)
        all_ba, all_f1 = [], []

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            if self.cv_strategy == "loso":
                splits = list(self.cv.split(self.X))
            else:
                splits = list(self.cv.split(self.X, self.y, groups=None))
                # Re-shuffle within seed for StratifiedKFold
                skf = StratifiedKFold(
                    n_splits=self.cfg["task"].get("cv_folds", 5),
                    shuffle=True,
                    random_state=seed,
                )
                splits = list(skf.split(self.X, self.y))

            for train_idx, val_idx in splits:
                model = _build_classifier(
                    self.model_config, condition, self.msm_ckpt, self.cont_ckpt
                ).to(self.device)

                metrics = _train_one_fold(
                    model, train_idx, val_idx, dataset, self.cfg, self.device, condition
                )
                all_ba.append(metrics["balanced_accuracy"])
                all_f1.append(metrics["macro_f1"])

        return {"balanced_accuracy": all_ba, "macro_f1": all_f1}

    def run_all_conditions(self, seeds: list[int] | None = None) -> dict[str, dict]:
        """Evaluate all four conditions and return combined results."""
        conditions: list[ConditionName] = [
            "from_scratch",
            "msm_finetune",
            "cont_finetune",
            "linear_probe_msm",
            "linear_probe_cont",
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
