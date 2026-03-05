"""Vendored minimal training harness (torchharness is not available on PyPI)."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    lr: float = 1e-3
    epochs: int = 100
    batch_size: int = 256
    device: str = "cuda"
    grad_clip: float = 1.0
    early_stopping_patience: int = 10
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 10
    scheduler: str = "none"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    weight_decay: float = 0.0


class Metric(ABC):
    @abstractmethod
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        ...

    @abstractmethod
    def compute(self) -> float:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class Callback:
    def on_train_epoch_start(self, epoch: int, trainer: "Trainer") -> None:
        pass

    def on_train_epoch_end(self, epoch: int, metrics: dict, trainer: "Trainer") -> None:
        pass

    def on_validation_epoch_end(self, epoch: int, metrics: dict, trainer: "Trainer") -> None:
        pass

    def on_train_end(self, trainer: "Trainer") -> None:
        pass


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        config: TrainerConfig,
        train_loader,
        val_loader,
        loss_fn,
        metrics: list[Metric] | None = None,
        callbacks: list[Callback] | None = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.metrics = metrics or []
        self.callbacks = callbacks or []
        device_str = config.device
        if device_str == "cuda" and not torch.cuda.is_available():
            device_str = "mps" if torch.backends.mps.is_available() else "cpu"
        elif device_str == "mps" and not torch.backends.mps.is_available():
            device_str = "cpu"
        self.device = torch.device(device_str)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

        # LR scheduler
        self.scheduler = None
        if config.scheduler == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=config.scheduler_factor,
                patience=config.scheduler_patience,
            )

        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.current_epoch = 0
        self.train_history: list[dict] = []

    def fit(self) -> dict:
        for epoch in range(1, self.config.epochs + 1):
            self.current_epoch = epoch
            for cb in self.callbacks:
                cb.on_train_epoch_start(epoch, self)

            train_loss = self._train_epoch()
            train_metrics = {"train_loss": train_loss}

            for cb in self.callbacks:
                cb.on_train_epoch_end(epoch, train_metrics, self)

            val_metrics = self._validate()

            for cb in self.callbacks:
                cb.on_validation_epoch_end(epoch, val_metrics, self)

            self.train_history.append({**train_metrics, **val_metrics, "epoch": epoch})

            if epoch % self.config.log_interval == 0:
                metric_str = " | ".join(f"{k}: {v:.4f}" for k, v in {**train_metrics, **val_metrics}.items())
                logger.info(f"Epoch {epoch}/{self.config.epochs} | {metric_str}")

            # LR scheduler step
            val_loss = val_metrics.get("val_loss", float("inf"))
            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint("best.pt")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        for cb in self.callbacks:
            cb.on_train_end(self)

        return self.train_history[-1] if self.train_history else {}

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        _is_ensemble = hasattr(self.model, "ensemble_forward")
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            if _is_ensemble:
                preds = self.model.ensemble_forward(x)
                loss = self.loss_fn(preds, y)
            else:
                preds = self.model(x)
                loss = self.loss_fn(preds, y)
            loss.backward()
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self) -> dict:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        _is_ensemble = hasattr(self.model, "ensemble_forward")
        for m in self.metrics:
            m.reset()
        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            if _is_ensemble:
                preds = self.model.ensemble_forward(x)
                loss = self.loss_fn(preds, y)
                preds_mean = preds.mean(dim=1)
            else:
                preds = self.model(x)
                loss = self.loss_fn(preds, y)
                preds_mean = preds
            total_loss += loss.item()
            n_batches += 1
            for m in self.metrics:
                # Pass full ensemble to metrics that accept it, mean otherwise
                if _is_ensemble and hasattr(m, 'update') and preds.dim() == 3:
                    m.update(preds, y)
                else:
                    m.update(preds_mean, y)
        result = {"val_loss": total_loss / max(n_batches, 1)}
        for m in self.metrics:
            result[m.name] = m.compute()
        return result

    def _save_checkpoint(self, filename: str) -> None:
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / filename
        torch.save(
            {
                "epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": self.best_val_loss,
            },
            path,
        )
        logger.debug(f"Checkpoint saved to {path}")
