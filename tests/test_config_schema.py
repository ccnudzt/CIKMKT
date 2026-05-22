from pathlib import Path

import pytest
import yaml

from src.core.config import load_experiment_config


def test_invalid_config_missing_required_key(tmp_path: Path):
    broken = {
        "runtime": {
            "device": "cpu",
            "seed": 1,
            "output_dir": "outputs",
            "save_last": True,
            "num_workers": 0,
        },
        "data": {
            "builder": "kt",
            "dataset_name": "toy",
            "min_len": 2,
            "max_len": 4,
            "num_folds": 2,
            "r_pad": 2,
            "records_path": "data/records.txt",
        },
        "model": {
            "name": "dkt",
            "embed_dim": 8,
            "hidden_dim": 16,
        },
        "trainer": {
            "name": "sequence",
            "batch_size": 2,
            "num_epochs": 1,
            "lr": 0.01,
            "weight_decay": 0.0,
            "eta_min": 1e-5,
            "patience": 2,
            "min_delta": 0.0,
        },
        "eval": {"save_best_by": "val_auc", "best_mode": "max"},
        "artifacts": {"cache_dir": "cache", "log_dir": "logs", "checkpoint_dir": "outputs"},
    }
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(broken, sort_keys=False), encoding="utf-8")

    with pytest.raises(KeyError):
        load_experiment_config(path)


def test_invalid_gcl_drop_scheme(tmp_path: Path):
    broken = {
        "runtime": {
            "device": "cpu",
            "seed": 1,
            "output_dir": str(tmp_path / "outputs"),
            "save_last": True,
            "num_workers": 0,
        },
        "data": {
            "builder": "graph_kt",
            "dataset_name": "toy_graph",
            "min_len": 2,
            "max_len": 4,
            "num_folds": 2,
            "r_pad": 0,
            "recordings_path": str(tmp_path / "recordings.jsonl"),
            "eid2sid_path": str(tmp_path / "eid2sid.json"),
        },
        "model": {
            "name": "gcl_kaga_dkt",
            "embed_dim": 8,
            "hidden_dim": 16,
            "g_layers": 2,
            "g_dropout": 0.1,
            "g_aggr": "mean",
            "gcl": {
                "enabled": True,
                "drop_scheme": "bad",
                "temperature": 0.2,
                "mask_feat_rate": 0.1,
            },
        },
        "trainer": {
            "name": "sequence",
            "batch_size": 2,
            "num_epochs": 1,
            "lr": 0.01,
            "weight_decay": 0.0,
            "eta_min": 1e-5,
            "patience": 2,
            "min_delta": 0.0,
        },
        "eval": {"save_best_by": "val_auc", "best_mode": "max"},
        "artifacts": {
            "cache_dir": str(tmp_path / "cache"),
            "log_dir": str(tmp_path / "logs"),
            "checkpoint_dir": str(tmp_path / "outputs"),
        },
    }
    path = tmp_path / "broken_gcl.yaml"
    path.write_text(yaml.safe_dump(broken, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError):
        load_experiment_config(path)
