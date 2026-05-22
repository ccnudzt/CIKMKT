from __future__ import annotations

from pathlib import Path


def _require_keys(section_name: str, section: dict, keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in section]
    if missing:
        joined = ", ".join(missing)
        raise KeyError(f"Section '{section_name}' is missing required keys: {joined}")


def _require_number(section_name: str, section: dict, key: str) -> None:
    value = section[key]
    if not isinstance(value, int | float):
        raise TypeError(f"Config field '{section_name}.{key}' must be numeric, got {type(value).__name__}.")


def _require_string(section_name: str, section: dict, key: str) -> None:
    value = section[key]
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"Config field '{section_name}.{key}' must be a non-empty string.")


def _require_bool(section_name: str, section: dict, key: str) -> None:
    value = section[key]
    if not isinstance(value, bool):
        raise TypeError(f"Config field '{section_name}.{key}' must be boolean, got {type(value).__name__}.")


def _validate_runtime(config: dict) -> None:
    runtime = config["runtime"]
    _require_keys("runtime", runtime, ("device", "seed", "output_dir", "save_last", "num_workers"))
    _require_string("runtime", runtime, "device")
    _require_number("runtime", runtime, "seed")
    _require_string("runtime", runtime, "output_dir")
    _require_bool("runtime", runtime, "save_last")
    _require_number("runtime", runtime, "num_workers")
    if "experiment_name" in runtime:
        _require_string("runtime", runtime, "experiment_name")


def _validate_data(config: dict) -> None:
    data = config["data"]
    _require_keys("data", data, ("builder", "dataset_name", "min_len", "max_len", "num_folds", "r_pad"))
    _require_string("data", data, "builder")
    _require_string("data", data, "dataset_name")
    for key in ("min_len", "max_len", "num_folds", "r_pad"):
        _require_number("data", data, key)

    builder = data["builder"]
    if builder == "kt":
        _require_keys("data", data, ("records_path", "qid2cog_path"))
    elif builder == "graph_kt":
        _require_keys("data", data, ("recordings_path", "eid2sid_path"))
    elif builder == "text_graph_kt":
        _require_keys("data", data, ("recordings_path", "eid2sid_path", "eid2desc_path", "sid2desc_path"))
    else:
        raise ValueError(f"Unsupported data.builder '{builder}'.")

    for key, value in data.items():
        if key.endswith(("_path", "_dir")) and isinstance(value, str):
            if not Path(value).is_absolute():
                raise ValueError(f"Resolved config path '{key}' must be absolute, got '{value}'.")


def _validate_model(config: dict) -> None:
    model = config["model"]
    _require_keys("model", model, ("name", "embed_dim", "hidden_dim"))
    _require_string("model", model, "name")
    _require_number("model", model, "embed_dim")
    _require_number("model", model, "hidden_dim")

    model_name = model["name"]
    if model_name in {"dkt", "cog_dkt"}:
        if "freeze_q_embed" in model:
            _require_bool("model", model, "freeze_q_embed")
        text_encoder = model.get("text_encoder", {})
        if model_name == "cog_dkt":
            if not text_encoder.get("enabled", False):
                raise ValueError("Model 'cog_dkt' requires model.text_encoder.enabled=true.")
    elif model_name in {"kaga_dkt", "gcl_kaga_dkt", "text_gcl_dkt"}:
        for key in ("g_layers", "g_dropout", "g_aggr"):
            if key not in model:
                raise KeyError(f"Model '{model_name}' requires key '{key}'.")
    else:
        raise ValueError(f"Unsupported model.name '{model_name}'.")

    if model_name in {"gcl_kaga_dkt", "text_gcl_dkt"}:
        gcl = model.get("gcl", {})
        if not gcl.get("enabled", False):
            raise ValueError(f"Model '{model_name}' requires model.gcl.enabled=true.")
        _require_string("model.gcl", gcl, "drop_scheme")
        _require_number("model.gcl", gcl, "temperature")
        _require_number("model.gcl", gcl, "mask_feat_rate")
        if gcl["drop_scheme"] == "random":
            _require_number("model.gcl", gcl, "drop_edge_rate")
        elif gcl["drop_scheme"] == "prob":
            _require_number("model.gcl", gcl, "p_min")
            _require_number("model.gcl", gcl, "p_max")
        else:
            raise ValueError(f"Unsupported model.gcl.drop_scheme '{gcl['drop_scheme']}'.")

    if model_name == "text_gcl_dkt":
        text_encoder = model.get("text_encoder", {})
        _require_keys("model.text_encoder", text_encoder, ("provider", "embed_dim"))
        _require_string("model.text_encoder", text_encoder, "provider")
        _require_number("model.text_encoder", text_encoder, "embed_dim")


def _validate_trainer(config: dict) -> None:
    trainer = config["trainer"]
    _require_keys(
        "trainer",
        trainer,
        ("name", "batch_size", "num_epochs", "lr", "weight_decay", "eta_min", "patience", "min_delta"),
    )
    _require_string("trainer", trainer, "name")
    for key in ("batch_size", "num_epochs", "lr", "weight_decay", "eta_min", "patience", "min_delta"):
        _require_number("trainer", trainer, key)
    if "aux_loss_weight" in trainer:
        _require_number("trainer", trainer, "aux_loss_weight")


def _validate_eval(config: dict) -> None:
    eval_cfg = config["eval"]
    _require_keys("eval", eval_cfg, ("save_best_by", "best_mode"))
    _require_string("eval", eval_cfg, "save_best_by")
    _require_string("eval", eval_cfg, "best_mode")
    if eval_cfg["best_mode"] not in {"max", "min"}:
        raise ValueError("eval.best_mode must be 'max' or 'min'.")


def _validate_artifacts(config: dict) -> None:
    artifacts = config["artifacts"]
    _require_keys("artifacts", artifacts, ("cache_dir", "log_dir", "checkpoint_dir"))
    for key in ("cache_dir", "log_dir", "checkpoint_dir"):
        _require_string("artifacts", artifacts, key)
        if not Path(artifacts[key]).is_absolute():
            raise ValueError(f"Resolved artifacts path '{key}' must be absolute, got '{artifacts[key]}'.")


def validate_experiment_config_schema(config: dict) -> None:
    """Validate the resolved experiment config against the project schema."""
    _validate_runtime(config)
    _validate_data(config)
    _validate_model(config)
    _validate_trainer(config)
    _validate_eval(config)
    _validate_artifacts(config)
