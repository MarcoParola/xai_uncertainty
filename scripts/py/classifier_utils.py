"""Utilities shared by classifier training and evaluation scripts."""

from __future__ import annotations

import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torchvision import transforms

from src.datasets import get_dataset
from src.config import load_training_config
from src.datasets.common import DEFAULT_DATA_ROOT
from src.models import get_model
from src.models.models import normalize_model_name


TRAINING_CONFIG = load_training_config()
IMAGE_RESIZE_SIZE = int(TRAINING_CONFIG["image_resize_size"])
IMAGE_SIZE = int(TRAINING_CONFIG["image_size"])


class AddGaussianNoise:
    def __init__(self, std: float = 0.01) -> None:
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.std <= 0:
            return tensor
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)


def convert_to_rgb(image):
    if hasattr(image, "convert"):
        return image.convert("RGB")
    return image


def get_device(device_name: str | None = None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transform(model_name: str, train: bool = False):
    model_name = normalize_model_name(model_name)
    from torchvision import models

    weights = models.get_model_weights(model_name).DEFAULT
    mean = weights.transforms().mean
    std = weights.transforms().std

    if train:
        return transforms.Compose(
            [
                transforms.Lambda(convert_to_rgb),
                transforms.Resize(IMAGE_RESIZE_SIZE),
                transforms.RandomAffine(
                    degrees=25,
                    translate=(0.09, 0.09),
                    fill=0,
                ),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.20,
                    saturation=0.20,
                    hue=0.02,
                ),
                transforms.ToTensor(),
                AddGaussianNoise(std=0.008),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    return transforms.Compose(
        [
            transforms.Lambda(convert_to_rgb),
            transforms.Resize(IMAGE_RESIZE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def make_dataloader(
    dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    shuffle: bool = False,
):
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def select_split(splits, split_name: str):
    return splits[{"train": 0, "val": 1, "test": 2}[split_name]]


def unwrap_dataset(dataset):
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


def get_class_names(*datasets) -> list[str]:
    for dataset in datasets:
        base_dataset = unwrap_dataset(dataset)
        if hasattr(base_dataset, "classes"):
            return list(base_dataset.classes)
        if hasattr(base_dataset, "categories"):
            return list(base_dataset.categories)
    raise ValueError("Could not infer class names from dataset.")


def load_dataset_splits(
    dataset_name: str,
    root: str | Path = DEFAULT_DATA_ROOT,
    transform=None,
    train_transform=None,
    eval_transform=None,
    **dataset_kwargs,
):
    if train_transform is None and eval_transform is None:
        return get_dataset(
            dataset_name,
            root=root,
            transform=transform,
            **dataset_kwargs,
        )

    train, _, _ = get_dataset(
        dataset_name,
        root=root,
        transform=train_transform,
        **dataset_kwargs,
    )
    _, val, test = get_dataset(
        dataset_name,
        root=root,
        transform=eval_transform,
        **dataset_kwargs,
    )
    return train, val, test


def replace_classifier_head(model: nn.Module, num_classes: int) -> nn.Module:
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if hasattr(model, "heads") and hasattr(model.heads, "head"):
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(
        f"Do not know how to replace the classifier head for {type(model).__name__}."
    )


def build_classifier(
    model_name: str,
    num_classes: int,
) -> nn.Module:
    model = get_model(model_name)
    return replace_classifier_head(model, num_classes)


def evaluate(model: nn.Module, dataloader, criterion, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_examples += batch_size

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
        "examples": total_examples,
    }


def checkpoint_payload(
    model: nn.Module,
    model_name: str,
    dataset_name: str,
    class_names: list[str],
    metrics: dict,
) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "model_name": normalize_model_name(model_name),
        "dataset_name": dataset_name,
        "num_classes": len(class_names),
        "class_names": class_names,
        "pretrained": True,
        "metrics": metrics,
    }
