from pathlib import Path

import json
import pytest
import yaml

from src.experiments.train import run_experiment


torch_geometric = pytest.importorskip("torch_geometric")


def _write_recordings(path: Path) -> None:
    records = [
        {"student_id": 1, "exercises_logs": ["1", "2", "3", "4"], "is_corrects": ["1", "0", "1", "0"]},
        {"student_id": 2, "exercises_logs": ["1", "3", "4"], "is_corrects": ["0", "1", "0"]},
        {"student_id": 3, "exercises_logs": ["2", "3", "4"], "is_corrects": ["1", "1", "0"]},
        {"student_id": 4, "exercises_logs": ["1", "2", "4"], "is_corrects": ["0", "1", "1"]},
        {"student_id": 5, "exercises_logs": ["2", "4", "3"], "is_corrects": ["1", "0", "1"]},
        {"student_id": 6, "exercises_logs": ["4", "3", "2"], "is_corrects": ["0", "1", "0"]},
    ]
    path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n", encoding="utf-8")


def _write_eid2sid(path: Path) -> None:
    mapping = {
        "1": ["11", "12"],
        "2": ["12"],
        "3": ["13"],
        "4": ["11", "13"],
    }
    path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_kaga_dkt_smoke_experiment(tmp_path: Path):
    recordings_path = tmp_path / "recordings.jsonl"
    eid2sid_path = tmp_path / "eid2sid.json"
    _write_recordings(recordings_path)
    _write_eid2sid(eid2sid_path)

    config = {
        "runtime": {
            "device": "cpu",
            "seed": 7,
            "output_dir": str(tmp_path / "outputs"),
            "experiment_name": "kaga_dkt_smoke",
            "save_last": True,
            "num_workers": 0,
        },
        "data": {
            "builder": "graph_kt",
            "dataset_name": "toy_graph",
            "recordings_path": str(recordings_path),
            "eid2sid_path": str(eid2sid_path),
            "min_len": 2,
            "max_len": 8,
            "num_folds": 2,
            "r_pad": 0,
        },
        "model": {
            "name": "kaga_dkt",
            "embed_dim": 8,
            "hidden_dim": 16,
            "g_layers": 2,
            "g_dropout": 0.1,
            "g_aggr": "mean",
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

    config_path = tmp_path / "kaga.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    summary = run_experiment(config_path)
    assert summary["num_folds"] == 2
    assert (tmp_path / "outputs" / "kaga_dkt_smoke" / "metrics.json").exists()
