"""Imagenette dataset splits."""

from __future__ import annotations

from pathlib import Path

from .common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_SEED,
    DEFAULT_VAL_FRACTION,
    import_torchvision_datasets,
    random_split,
    train_val_lengths,
)


IMAGENETTE_SIZE = "320px"


def get_dataset(
    root: str | Path = DEFAULT_DATA_ROOT,
    transform=None,
    target_transform=None,
    download: bool = False,
    seed: int = DEFAULT_SEED,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    test_fraction: float = 0.0,
    return_bboxes: bool = False,
):
    """Return train/validation splits and use the official validation set as test."""
    if return_bboxes:
        raise ValueError(
            "Imagenette does not provide localization annotations, so bounding-box "
            "metrics such as pointing game are unavailable."
        )

    datasets = import_torchvision_datasets()
    kwargs = {
        "root": str(root),
        "size": IMAGENETTE_SIZE,
        "transform": transform,
        "target_transform": target_transform,
        "download": download,
    }
    train_full = datasets.Imagenette(split="train", **kwargs)
    test = datasets.Imagenette(split="val", **kwargs)

    train_size, val_size = train_val_lengths(len(train_full), val_fraction)
    train, val = random_split(train_full, (train_size, val_size), seed)
    return train, val, test
