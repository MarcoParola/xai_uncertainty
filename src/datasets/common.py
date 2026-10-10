"""Shared helpers for dataset loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.config import load_training_config


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
TRAINING_CONFIG = load_training_config()
DEFAULT_SEED = int(TRAINING_CONFIG["seed"])
DEFAULT_VAL_FRACTION = float(TRAINING_CONFIG["val_fraction"])
DEFAULT_TEST_FRACTION = float(TRAINING_CONFIG["test_fraction"])
IMAGE_RESIZE_SIZE = int(TRAINING_CONFIG["image_resize_size"])
EVAL_IMAGE_SIZE = int(TRAINING_CONFIG["image_size"])


BBOX_MASK_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(IMAGE_RESIZE_SIZE, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.CenterCrop(EVAL_IMAGE_SIZE),
        transforms.ToTensor(),
    ]
)


def import_torchvision_datasets():
    try:
        from torchvision import datasets
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torchvision is required for dataset loading. Install it with "
            "`env/bin/python -m pip install torch torchvision`."
        ) from exc
    return datasets


def random_split(dataset, lengths: Sequence[int], seed: int):
    try:
        import torch
        from torch.utils.data import random_split as torch_random_split
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torch is required for deterministic dataset splits. Install it with "
            "`env/bin/python -m pip install torch torchvision`."
        ) from exc

    generator = torch.Generator().manual_seed(seed)
    return torch_random_split(dataset, lengths, generator=generator)


def train_val_lengths(dataset_size: int, val_fraction: float) -> tuple[int, int]:
    val_size = round(dataset_size * val_fraction)
    train_size = dataset_size - val_size
    return train_size, val_size


def train_val_test_lengths(
    dataset_size: int,
    val_fraction: float,
    test_fraction: float,
) -> tuple[int, int, int]:
    test_size = round(dataset_size * test_fraction)
    val_size = round(dataset_size * val_fraction)
    train_size = dataset_size - val_size - test_size
    return train_size, val_size, test_size


def bbox_from_mask(mask) -> torch.Tensor:
    if isinstance(mask, Image.Image):
        mask = BBOX_MASK_TRANSFORM(mask)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)

    foreground = mask.squeeze(0) > 0
    rows, cols = torch.where(foreground)
    if rows.numel() == 0:
        return torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)

    x1 = cols.min().float()
    y1 = rows.min().float()
    x2 = cols.max().float() + 1.0
    y2 = rows.max().float() + 1.0
    return torch.stack([x1, y1, x2, y2])


def bbox_from_original_box(
    image_path: str | Path,
    box: tuple[float, float, float, float],
) -> torch.Tensor:
    with Image.open(image_path) as image:
        mask = Image.new("L", image.size, 0)
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    mask.paste(255, (max(x1, 0), max(y1, 0), max(x2, 0), max(y2, 0)))
    return bbox_from_mask(mask)


class BBoxDataset(Dataset):
    """Wrap a classification dataset to return ``(image, label, bbox)``."""

    def __init__(self, dataset, bbox_getter) -> None:
        self.dataset = dataset
        self.bbox_getter = bbox_getter

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return image, label, self.bbox_getter(index)

    def __getattr__(self, name: str):
        return getattr(self.dataset, name)
