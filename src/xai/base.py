"""Shared interface and helpers for saliency methods."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.nn import functional as F


class SaliencyMethod(ABC):
    """Base class for batch saliency map generators."""

    score_type = "logit"

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


def find_spatial_target_layer(model: nn.Module) -> nn.Module:
    """Return the final spatial feature layer of a CNN classifier."""
    if hasattr(model, "layer4") and isinstance(model.layer4, nn.Sequential):
        return model.layer4[-1]

    if hasattr(model, "features") and isinstance(model.features, nn.Sequential):
        for module in reversed(model.features):
            if isinstance(module, (nn.Conv2d, nn.Sequential)):
                return module

    raise ValueError(
        f"Could not identify a final spatial CNN layer for {type(model).__name__}. "
        "Pass target_layer explicitly. Original CAM methods require 2D CNN features."
    )


def find_linear_classifier(model: nn.Module) -> nn.Linear:
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        return model.fc
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        return model.classifier
    raise ValueError(
        f"Could not identify a linear classifier for {type(model).__name__}. "
        "Original CAM requires global-average-pooled CNN features followed by a linear layer."
    )


def capture_activations(
    model: nn.Module,
    target_layer: nn.Module,
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    activations = None

    def forward_hook(_module, _inputs, output):
        nonlocal activations
        activations = output

    handle = target_layer.register_forward_hook(forward_hook)
    try:
        logits = model(images)
    finally:
        handle.remove()

    if not isinstance(activations, torch.Tensor) or activations.ndim != 4:
        shape = None if not isinstance(activations, torch.Tensor) else tuple(activations.shape)
        raise ValueError(
            "The CAM target layer must return a tensor with shape (B, C, H, W), "
            f"got {shape}."
        )
    return activations, logits


def upsample_to_input(saliency: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    return F.interpolate(
        saliency,
        size=images.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
