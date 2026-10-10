"""Project configuration helpers."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_CONFIG_PATH = PROJECT_ROOT / "config" / "training.json"

DEFAULT_TRAINING_CONFIG: dict[str, Any] = {
    "epochs": 120,
    "batch_size": 32,
    "lr": 5e-5,
    "weight_decay": 1e-4,
    "optimizer": "adam",
    "scheduler": "cosine",
    "warmup_epochs": 5,
    "min_lr": 0.0,
    "num_workers": 4,
    "seed": 42,
    "patience": 15,
    "min_delta": 0.0,
    "checkpoint_offsets": [3, 5, 7, 10, 15],
    "val_fraction": 0.15,
    "test_fraction": 0.13,
    "image_resize_size": 256,
    "image_size": 224,
    "saliency_generation_limit": 1000,
}


@lru_cache(maxsize=None)
def load_training_config(config_path: str | Path = DEFAULT_TRAINING_CONFIG_PATH) -> dict[str, Any]:
    path = Path(config_path)
    config = DEFAULT_TRAINING_CONFIG.copy()
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            config.update(json.load(handle))
    validate_training_config(config, path)
    return config


def validate_training_config(config: dict[str, Any], path: Path) -> None:
    val_fraction = float(config["val_fraction"])
    test_fraction = float(config["test_fraction"])
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"{path}: val_fraction must be in [0, 1).")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError(f"{path}: test_fraction must be in [0, 1).")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError(f"{path}: val_fraction + test_fraction must be < 1.")

    offsets = config.get("checkpoint_offsets", [])
    if not isinstance(offsets, list) or any(int(offset) <= 0 for offset in offsets):
        raise ValueError(f"{path}: checkpoint_offsets must be a list of positive integers.")

    if int(config["image_resize_size"]) <= 0 or int(config["image_size"]) <= 0:
        raise ValueError(f"{path}: image sizes must be positive integers.")

    if int(config["saliency_generation_limit"]) < 0:
        raise ValueError(f"{path}: saliency_generation_limit must be non-negative.")
