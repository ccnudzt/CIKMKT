from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.core.artifacts import save_checkpoint
from src.core.metrics import compute_binary_metrics
from src.trainers import TRAINER_REGISTRY
from src.trainers.base import BaseTrainer


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: float | None = None
        self.counter = 0
        self.stopped = False

    def step(self, metric: float) -> bool:
        improved = False
        if self.best is None:
            self.best = metric
            improved = True
        elif self.mode == "max" and metric > self.best + self.min_delta:
            self.best = metric
            self.counter = 0
            improved = True
        elif self.mode == "min" and metric < self.best - self.min_delta:
            self.best = metric
            self.counter = 0
            improved = True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True
        return improved


class SequenceTrainer(BaseTrainer):
    def __init__(self, config: dict, model: torch.nn.Module, fold_dir: Path):
        self.config = config
        self.model = model
        self.fold_dir = fold_dir
        self.device = torch.device(config["runtime"]["device"])
        trainer_cfg = config["trainer"]
        self.num_epochs = int(trainer_cfg["num_epochs"])
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(trainer_cfg["lr"]),
            weight_decay=float(trainer_cfg.get("weight_decay", 0.0)),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_epochs,
            eta_min=float(trainer_cfg.get("eta_min", 1e-6)),
        )
        self.early = EarlyStopping(
            patience=int(trainer_cfg["patience"]),
            min_delta=float(trainer_cfg.get("min_delta", 0.0)),
            mode=config["eval"].get("best_mode", "max"),
        )
        self.model.to(self.device)

    def _step_epoch(self, loader, train: bool, desc: str) -> dict[str, float]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        losses: list[float] = []
        aux_losses: list[float] = []
        all_labels: list[np.ndarray] = []
        all_probs: list[np.ndarray] = []

        iterator = tqdm(loader, desc=desc, leave=False)
        for batch in iterator:
            qids = batch["qids"].to(self.device)
            responses = batch["rs"].to(self.device)
            masks = batch["masks"].to(self.device)

            input_qids = qids[:, :-1]
            input_responses = responses[:, :-1]
            input_masks = masks[:, :-1]
            labels = responses[:, 1:].float()
            label_masks = masks[:, 1:].bool()
            r_pad = int(self.config["data"].get("r_pad", 2))
            if r_pad not in (0, 1):
                label_masks = label_masks & (responses[:, 1:] != r_pad)

            if input_qids.size(1) == 0:
                continue

            context = torch.enable_grad() if train else torch.no_grad()
            with context:
                logits = self.model(input_qids, input_responses, input_masks)
                loss_mat = self.criterion(logits, labels)
                denom = label_masks.sum().clamp(min=1)
                main_loss = (loss_mat * label_masks).sum() / denom
                aux_loss = torch.tensor(0.0, device=self.device)
                aux_weight = float(self.config["trainer"].get("aux_loss_weight", 0.0))
                if train and aux_weight > 0 and hasattr(self.model, "compute_aux_loss"):
                    aux_loss = self.model.compute_aux_loss()
                loss = main_loss + aux_weight * aux_loss

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            losses.append(float(loss.item()))
            aux_losses.append(float(aux_loss.item()))
            probs = torch.sigmoid(logits)
            all_labels.append(labels[label_masks].detach().cpu().numpy())
            all_probs.append(probs[label_masks].detach().cpu().numpy())
            iterator.set_postfix(loss=f"{loss.item():.4f}")

        labels_np = np.concatenate(all_labels, axis=0) if all_labels else np.array([])
        probs_np = np.concatenate(all_probs, axis=0) if all_probs else np.array([])
        metrics = compute_binary_metrics(labels_np, probs_np)
        metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
        metrics["aux_loss"] = float(np.mean(aux_losses)) if aux_losses else 0.0
        return metrics

    def fit(self, train_loader, val_loader) -> dict:
        best_metric_name = self.config["eval"].get("save_best_by", "val_auc")
        best_metric = -math.inf
        best_epoch = -1
        history: list[dict[str, float]] = []

        for epoch in range(1, self.num_epochs + 1):
            lr = self.optimizer.param_groups[0]["lr"]
            train_metrics = self._step_epoch(train_loader, train=True, desc=f"Train {epoch}/{self.num_epochs} lr={lr:.2e}")
            val_metrics = self._step_epoch(val_loader, train=False, desc=f"Val {epoch}/{self.num_epochs}")
            self.scheduler.step()

            epoch_metrics = {
                "epoch": epoch,
                "lr": float(lr),
                "train_loss": train_metrics["loss"],
                "train_aux_loss": train_metrics["aux_loss"],
                "train_auc": train_metrics["auc"],
                "train_acc": train_metrics["acc"],
                "val_loss": val_metrics["loss"],
                "val_auc": val_metrics["auc"],
                "val_acc": val_metrics["acc"],
            }
            history.append(epoch_metrics)

            print(
                f"[Epoch {epoch:02d}] "
                f"loss={train_metrics['loss']:.4f} "
                f"aux={train_metrics['aux_loss']:.4f} "
                f"train_auc={train_metrics['auc']:.4f} train_acc={train_metrics['acc']:.4f} "
                f"val_auc={val_metrics['auc']:.4f} val_acc={val_metrics['acc']:.4f}"
            )

            monitored = epoch_metrics.get(best_metric_name, float("nan"))
            improved = False
            if not math.isnan(monitored):
                improved = self.early.step(monitored)
                if improved:
                    best_metric = monitored
                    best_epoch = epoch
                    save_checkpoint(
                        path=self.fold_dir / "best.pt",
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        metrics=epoch_metrics,
                        config=self.config,
                        meta={"best_metric_name": best_metric_name, "best_epoch": best_epoch},
                    )

            if self.config["runtime"].get("save_last", True):
                save_checkpoint(
                    path=self.fold_dir / "last.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    metrics=epoch_metrics,
                    config=self.config,
                    meta={"best_metric_name": best_metric_name, "best_epoch": best_epoch},
                )

            if self.early.stopped:
                print(f"Early stopping at epoch {epoch}. Best {best_metric_name}={best_metric:.4f}.")
                break

        return {
            "best_epoch": best_epoch,
            "best_metric_name": best_metric_name,
            "best_metric": float(best_metric),
            "history": history,
            "stopped_epoch": history[-1]["epoch"] if history else 0,
        }


@TRAINER_REGISTRY.register("sequence")
def build_sequence_trainer(config: dict, model: torch.nn.Module, fold_dir: Path) -> SequenceTrainer:
    return SequenceTrainer(config=config, model=model, fold_dir=fold_dir)
