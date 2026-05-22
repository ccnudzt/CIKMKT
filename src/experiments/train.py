from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

from src.core.artifacts import ensure_dir, save_json, save_yaml
from src.core.config import load_experiment_config
from src.data import DATASET_REGISTRY
from src.models import MODEL_REGISTRY
from src.trainers import TRAINER_REGISTRY


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(config: dict):
    builder_name = config["data"].get("builder", "kt")
    builder = DATASET_REGISTRY.get(builder_name)
    return builder(config)


def build_model(config: dict, metadata: dict):
    builder = MODEL_REGISTRY.get(config["model"]["name"])
    return builder(config, metadata)


def build_trainer(config: dict, model: torch.nn.Module, fold_dir: Path):
    builder = TRAINER_REGISTRY.get(config["trainer"]["name"])
    return builder(config, model, fold_dir)


def _make_loaders(dataset_bundle, train_idx, val_idx, batch_size: int, num_workers: int):
    train_loader = DataLoader(
        Subset(dataset_bundle.dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset_bundle.collate_fn,
        drop_last=False,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        Subset(dataset_bundle.dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset_bundle.collate_fn,
        drop_last=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def run_experiment(config_path: str | Path) -> dict:
    config, _ = load_experiment_config(config_path)
    set_seed(int(config["runtime"]["seed"]))

    experiment_name = config["runtime"]["experiment_name"]
    output_root = ensure_dir(Path(config["runtime"]["output_dir"]) / experiment_name)
    save_yaml(output_root / "config.resolved.yaml", config)

    dataset_bundle = build_dataset(config)
    dataset = dataset_bundle.dataset
    num_folds = int(config["data"]["num_folds"])
    batch_size = int(config["trainer"]["batch_size"])
    num_workers = int(config["runtime"].get("num_workers", 0))

    if len(dataset) < num_folds:
        raise RuntimeError(f"Need at least {num_folds} sequences, got {len(dataset)}.")

    print(f"Running experiment '{experiment_name}' on {len(dataset)} sequences.")
    all_indices = np.arange(len(dataset))
    splitter = KFold(n_splits=num_folds, shuffle=True, random_state=int(config["runtime"]["seed"]))

    fold_results = []
    for fold_id, (train_idx, val_idx) in enumerate(splitter.split(all_indices), start=1):
        fold_dir = ensure_dir(output_root / f"fold_{fold_id}")
        train_loader, val_loader = _make_loaders(dataset_bundle, train_idx, val_idx, batch_size, num_workers)
        model = build_model(config, dataset_bundle.metadata)
        trainer = build_trainer(config, model, fold_dir)
        result = trainer.fit(train_loader, val_loader)
        fold_metrics = {
            "fold": fold_id,
            "best_epoch": result["best_epoch"],
            "best_metric_name": result["best_metric_name"],
            "best_metric": result["best_metric"],
            "history": result["history"],
        }
        fold_results.append(fold_metrics)
        save_json(fold_dir / "metrics.json", fold_metrics)

    best_metric_name = fold_results[0]["best_metric_name"]
    best_values = np.array([item["best_metric"] for item in fold_results], dtype=float)
    summary = {
        "experiment_name": experiment_name,
        "num_folds": num_folds,
        "best_metric_name": best_metric_name,
        "mean_best_metric": float(np.nanmean(best_values)),
        "std_best_metric": float(np.nanstd(best_values)),
        "folds": fold_results,
    }
    save_json(output_root / "metrics.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified KT experiment runner")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    args = parser.parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()
