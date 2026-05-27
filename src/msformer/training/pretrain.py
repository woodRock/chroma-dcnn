"""
Pretraining loop — Masked Spectra Modelling (MSM) objective.

Usage:
    trainer = PretrainTrainer(config)
    trainer.train()

Checkpoints are saved to config['logging']['checkpoint_dir'] every
`checkpoint_every` epochs and when validation loss improves.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

def _best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


from msformer.data.datasets import MSMDataset, PretrainingDataset
from msformer.models import MSMModel, SpectrumConfig


class PretrainTrainer:
    def __init__(self, config: dict) -> None:
        self.cfg = config
        self.device = _best_device()
        print(f"Device: {self.device}")

        self.model_cfg = SpectrumConfig(
            input_type=config["model"]["input_type"],
            mz_max=config["model"]["mz_max"],
            patch_size=config["model"]["patch_size"],
            max_peaks=config["model"].get("max_peaks", 200),
            hidden_dim=config["model"]["hidden_dim"],
            num_layers=config["model"]["num_layers"],
            num_heads=config["model"]["num_heads"],
            ffn_dim=config["model"].get("ffn_dim", config["model"]["hidden_dim"] * 4),
            dropout=config["model"]["dropout"],
        )
        self.objective = "msm"
        self._build_model()
        self._build_data()
        self._build_optimiser()

        self.checkpoint_dir = Path(config["logging"]["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float("inf")

    def _build_model(self) -> None:
        self.model = MSMModel(self.model_cfg).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {n_params:,}  objective=msm")

    def _build_data(self) -> None:
        h5_path = self.cfg["data"]["h5_path"]
        n_workers = self.cfg["data"].get("num_workers", 4)
        batch_size = self.cfg["training"]["batch_size"]

        train_base = PretrainingDataset(h5_path, split="train")
        val_base = PretrainingDataset(h5_path, split="val")

        mask_ratio = self.cfg["pretraining"]["mask_ratio"]
        train_ds = MSMDataset(train_base, mask_ratio)
        val_ds = MSMDataset(val_base, mask_ratio)

        self.train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=n_workers, pin_memory=True, drop_last=True,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=n_workers, pin_memory=True,
        )

    def _build_optimiser(self) -> None:
        tcfg = self.cfg["training"]
        self.optimiser = AdamW(
            self.model.parameters(),
            lr=tcfg["lr"],
            weight_decay=tcfg["weight_decay"],
            betas=(tcfg.get("beta1", 0.9), tcfg.get("beta2", 0.999)),
        )
        self.scheduler = CosineAnnealingLR(
            self.optimiser,
            T_max=tcfg["epochs"] - tcfg.get("warmup_epochs", 5),
            eta_min=tcfg["lr"] * 0.01,
        )
        self.warmup_epochs = tcfg.get("warmup_epochs", 5)
        self.grad_clip = tcfg.get("grad_clip", 1.0)

    def _warmup_lr(self, epoch: int) -> None:
        if epoch < self.warmup_epochs:
            lr = self.cfg["training"]["lr"] * (epoch + 1) / self.warmup_epochs
            for pg in self.optimiser.param_groups:
                pg["lr"] = lr

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in self.train_loader:
            batch = self._to_device(batch)
            self.optimiser.zero_grad()

            loss, _ = self.model(batch)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimiser.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def val_epoch(self) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        for batch in self.val_loader:
            batch = self._to_device(batch)
            loss, _ = self.model(batch)
            total_loss += loss.item()

        return {"loss": total_loss / len(self.val_loader)}

    def save_checkpoint(self, epoch: int, val_loss: float, tag: str = "") -> None:
        state = {
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state": self.model.state_dict(),
            "optimiser_state": self.optimiser.state_dict(),
            "config": self.cfg,
        }
        name = f"epoch_{epoch:04d}.pt" if not tag else f"{tag}.pt"
        torch.save(state, self.checkpoint_dir / name)

    def train(self) -> None:
        epochs = self.cfg["training"]["epochs"]
        ckpt_every = self.cfg["training"].get("checkpoint_every", 10)
        run_name = self.cfg["logging"].get("run_name", self.objective)

        print(f"\n{'='*60}")
        print(f"Pretraining  run={run_name}  epochs={epochs}")
        print(f"{'='*60}\n")

        for epoch in range(1, epochs + 1):
            self._warmup_lr(epoch - 1)
            t0 = time.time()

            train_loss = self.train_epoch(epoch)
            val_metrics = self.val_epoch()
            val_loss = val_metrics["loss"]

            if epoch >= self.warmup_epochs:
                self.scheduler.step()

            elapsed = time.time() - t0
            extra = "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items() if k != "loss")
            lr = self.optimiser.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:4d}/{epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"{extra}  lr={lr:.2e}  {elapsed:.1f}s"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, tag="best")
                print(f"  → New best checkpoint (val_loss={val_loss:.4f})")

            if epoch % ckpt_every == 0:
                self.save_checkpoint(epoch, val_loss)

        print(f"\nTraining complete. Best val loss: {self.best_val_loss:.4f}")

    def _to_device(self, batch):
        if isinstance(batch, dict):
            return {k: v.to(self.device) if isinstance(v, Tensor) else v for k, v in batch.items()}
        return batch.to(self.device)
