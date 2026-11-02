"""Original class activation mapping (CAM)."""

from __future__ import annotations

import torch
from torch import nn

from .base import (
    SaliencyMethod,
    capture_activations,
    find_linear_classifier,
    find_spatial_target_layer,
    model_device,
    upsample_to_input,
    validate_images,
)


class CAM(SaliencyMethod):
    """Generate the original CAM for GAP-based CNN classifiers.

    CAM directly uses the target class's linear-classifier weights. It is valid
    only when the target feature maps feed a global average pool followed by the
    classifier.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module | None = None,
        classifier: nn.Linear | None = None,
    ) -> None:
        super().__init__(model)
        self.target_layer = target_layer
        self.classifier = classifier

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        validate_images(images)
        images = images.to(model_device(self.model))
        target_layer = self.target_layer or find_spatial_target_layer(self.model)
        classifier = self.classifier or find_linear_classifier(self.model)
        was_training = self.model.training
        self.model.eval()

        try:
            activations, logits = capture_activations(self.model, target_layer, images)
            targets = (
                logits.argmax(dim=1)
                if targets is None
                else targets.to(device=images.device, dtype=torch.long)
            )
            if classifier.in_features != activations.size(1):
                raise ValueError(
                    "Original CAM requires the target layer's channel count to match "
                    f"the classifier input ({activations.size(1)} != "
                    f"{classifier.in_features})."
                )
            weights = classifier.weight.index_select(0, targets).view(
                images.size(0), activations.size(1), 1, 1
            )
            maps = (weights * activations).sum(dim=1, keepdim=True)
            return upsample_to_input(maps, images).detach()
        finally:
            if was_training:
                self.model.train()


__all__ = ["CAM"]
