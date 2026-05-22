from pathlib import Path

from src.core.config import load_experiment_config


def test_all_experiment_configs_load():
    experiment_dir = Path("configs/experiments")
    config_paths = sorted(experiment_dir.glob("*/*.yaml"))
    assert config_paths, "No experiment configs found."

    for path in config_paths:
        config, _ = load_experiment_config(path)
        assert config["runtime"]["config_path"].endswith(str(path))
