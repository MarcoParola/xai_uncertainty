"""Shared interface and helpers for saliency methods."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.nn import functional as F


class SaliencyMethod(ABC):
    """Base class for batch saliency map generators."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    @abstractmethod
    def generate(
        self,
        images: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return saliency maps with shape ``(B, 1, H, W)``."""

    def __call__(
        self,
        images: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.generate(images, targets)


def model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def validate_images(images: torch.Tensor) -> None:
    if images.ndim != 4:
        raise ValueError(
            f"Expected images with shape (B, C, H, W), got {tuple(images.shape)}."
        )


def normalize_maps(saliency: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    validate_images(saliency)
    batch_size = saliency.size(0)
    flat = saliency.flatten(start_dim=1)
    min_values = flat.min(dim=1).values.view(batch_size, 1, 1, 1)
    max_values = flat.max(dim=1).values.view(batch_size, 1, 1, 1)
    return (saliency - min_values) / (max_values - min_values + eps)


@torch.no_grad()
def infer_targets(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    was_training = model.training
    model.eval()
    logits = model(images)
    if was_training:
        model.train()
    return logits.argmax(dim=1)


def gather_scores(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return logits.gather(1, targets.view(-1, 1)).squeeze(1)


def upsample_to_input(saliency: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    return F.interpolate(
        saliency,
        size=images.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
