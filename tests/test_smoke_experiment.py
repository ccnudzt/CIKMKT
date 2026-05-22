from pathlib import Path

import json
import yaml

from src.experiments.train import run_experiment


def _write_records(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "1,2,3,4",
                "1,0,1,0",
                "1,3,4",
                "0,1,0",
                "2,3,4",
                "1,1,0",
                "1,2,4",
                "0,1,1",
                "2,4,3",
                "1,0,1",
                "4,3,2",
                "0,1,0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_qid2cog(path: Path) -> None:
    lines = [
        {"qid": "1", "desc": {"memory": "m1", "understand": "u1", "skill": "s1"}},
        {"qid": "2", "desc": {"memory": "m2", "understand": "u2", "skill": "s2"}},
        {"qid": "3", "desc": {"memory": "m3", "understand": "u3", "skill": "s3"}},
        {"qid": "4", "desc": {"memory": "m4", "understand": "u4", "skill": "s4"}},
    ]
    path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in lines) + "\n", encoding="utf-8")


def _make_config(tmp_path: Path, model_name: str, text_enabled: bool) -> Path:
    records_path = tmp_path / "records.txt"
    qid2cog_path = tmp_path / "qid2cog.jsonl"
    _write_records(records_path)
    _write_qid2cog(qid2cog_path)

    config = {
        "runtime": {
            "device": "cpu",
            "seed": 7,
            "output_dir": str(tmp_path / "outputs"),
            "experiment_name": f"{model_name}_smoke",
            "save_last": True,
            "num_workers": 0,
        },
        "data": {
            "builder": "kt",
            "dataset_name": "toy",
            "records_path": str(records_path),
            "qid2cog_path": str(qid2cog_path),
            "min_len": 2,
            "max_len": 8,
            "num_folds": 2,
            "r_pad": 2,
        },
        "model": {
            "name": model_name,
            "embed_dim": 8,
            "hidden_dim": 16,
            "freeze_q_embed": True,
            "text_encoder": {
                "enabled": text_enabled,
                "provider": "hash",
                "agg": "mean",
                "seed": 11,
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
    config_path = tmp_path / f"{model_name}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def test_dkt_smoke_experiment(tmp_path: Path):
    config_path = _make_config(tmp_path, model_name="dkt", text_enabled=False)
    summary = run_experiment(config_path)
    assert summary["num_folds"] == 2
    assert (tmp_path / "outputs" / "dkt_smoke" / "metrics.json").exists()


def test_cog_dkt_smoke_experiment(tmp_path: Path):
    config_path = _make_config(tmp_path, model_name="cog_dkt", text_enabled=True)
    summary = run_experiment(config_path)
    assert summary["num_folds"] == 2
    assert (tmp_path / "outputs" / "cog_dkt_smoke" / "metrics.json").exists()
