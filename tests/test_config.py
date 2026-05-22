from pathlib import Path

from src.core.config import load_experiment_config


def test_load_experiment_config_merges_layers():
    config, project_root = load_experiment_config("configs/experiments/dkt/mooc_processed.yaml")
    assert config["runtime"]["experiment_name"] == "dkt_mooc_processed"
    assert config["model"]["name"] == "dkt"
    assert config["data"]["dataset_name"] == "mooc_processed"
    assert Path(config["data"]["records_path"]).is_absolute()
    assert project_root == Path(__file__).resolve().parents[1]
