from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
import yaml

from src.core.schema import validate_experiment_config_schema


TOP_LEVEL_KEYS = ("runtime", "data", "model", "trainer", "eval", "artifacts")
PATH_SUFFIXES = ("_path", "_dir")


def _deep_merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config at {path} must be a YAML mapping.")
    return data


def _expand_inherits(config: dict, config_path: Path) -> dict:
    inherits = config.get("inherits", [])
    if not inherits:
        return config

    merged: dict = {}
    for relative_path in inherits:
        inherited_path = (config_path.parent / relative_path).resolve()
        inherited = _load_config_recursive(inherited_path)
        merged = _deep_merge(merged, inherited)

    local = deepcopy(config)
    local.pop("inherits", None)
    return _deep_merge(merged, local)


def _load_config_recursive(config_path: Path) -> dict:
    raw = _read_yaml(config_path)
    return _expand_inherits(raw, config_path)


def _resolve_path_value(key: str, value: str, project_root: Path) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    if key.endswith(PATH_SUFFIXES):
        return str((project_root / path).resolve())
    return value


def _resolve_paths(data: object, project_root: Path) -> object:
    if isinstance(data, dict):
        resolved = {}
        for key, value in data.items():
            if isinstance(value, str):
                resolved[key] = _resolve_path_value(key, value, project_root)
            else:
                resolved[key] = _resolve_paths(value, project_root)
        return resolved
    if isinstance(data, list):
        return [_resolve_paths(item, project_root) for item in data]
    return data


def _validate_config(config: dict) -> None:
    missing = [key for key in TOP_LEVEL_KEYS if key not in config]
    if missing:
        joined = ", ".join(missing)
        raise KeyError(f"Resolved config is missing top-level sections: {joined}")
    validate_experiment_config_schema(config)


def _resolve_device(config: dict) -> None:
    device = config["runtime"].get("device", "auto")
    if device == "auto":
        config["runtime"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"


def load_experiment_config(config_path: str | Path) -> tuple[dict, Path]:
    path = Path(config_path).expanduser().resolve()
    project_root = path.parents[2] if path.parent.name == "experiments" else Path(__file__).resolve().parents[2]
    config = _load_config_recursive(path)
    config = _resolve_paths(config, project_root)
    _resolve_device(config)
    _validate_config(config)
    config["runtime"]["config_path"] = str(path)
    config["runtime"]["project_root"] = str(project_root)
    return config, project_root
