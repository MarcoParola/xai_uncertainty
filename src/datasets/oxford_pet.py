"""Oxford-IIIT Pet dataset splits."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from .common import (
    BBoxDataset,
    DEFAULT_DATA_ROOT,
    DEFAULT_SEED,
    DEFAULT_VAL_FRACTION,
    bbox_from_mask,
    import_torchvision_datasets,
    random_split,
    train_val_lengths,
)


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
    datasets = import_torchvision_datasets()

    trainval = datasets.OxfordIIITPet(
        root=str(root),
        split="trainval",
        target_types="category",
        transform=transform,
        target_transform=target_transform,
        download=download,
    )
    test = datasets.OxfordIIITPet(
        root=str(root),
        split="test",
        target_types="category",
        transform=transform,
        target_transform=target_transform,
        download=download,
    )

    if return_bboxes:
        trainval = BBoxDataset(trainval, lambda index: oxford_pet_bbox(trainval, index))
        test = BBoxDataset(test, lambda index: oxford_pet_bbox(test, index))

    train_size, val_size = train_val_lengths(len(trainval), val_fraction)
    train, val = random_split(trainval, (train_size, val_size), seed)
    return train, val, test


def oxford_pet_bbox(dataset, index: int):
    trimap = Image.open(dataset._segs[index])
    mask = trimap.point(lambda value: 255 if value != 2 else 0)
    return bbox_from_mask(mask)
