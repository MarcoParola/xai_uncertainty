"""Gradient-free Score-CAM and Ablation-CAM implementations."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .base import (
    SaliencyMethod,
    capture_activations,
    find_spatial_target_layer,
    gather_scores,
    model_device,
    upsample_to_input,
    validate_images,
)


def _targets_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor | None,
) -> torch.Tensor:
    if targets is None:
        return logits.argmax(dim=1)
    return targets.to(device=logits.device, dtype=torch.long)


def _nonzero_denominator(values: torch.Tensor, eps: float) -> torch.Tensor:
    signs = torch.where(values < 0, -torch.ones_like(values), torch.ones_like(values))
    return torch.where(values.abs() < eps, signs * eps, values)


class ScoreCAM(SaliencyMethod):
    """Generate Score-CAM using post-softmax target probabilities."""

    score_type = "probability"

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module | None = None,
        batch_size: int = 32,
    ) -> None:
        super().__init__(model)
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        self.target_layer = target_layer
        self.batch_size = batch_size

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        validate_images(images)
        images = images.to(model_device(self.model))
        target_layer = self.target_layer or find_spatial_target_layer(self.model)
        was_training = self.model.training
        self.model.eval()

        try:
            activations, logits = capture_activations(self.model, target_layer, images)
            targets = _targets_from_logits(logits, targets)
            maps = [
                self._single_image(image, activation, target)
                for image, activation, target in zip(images, activations, targets)
            ]
            return torch.cat(maps).detach()
        finally:
            if was_training:
                self.model.train()

    def _single_image(
        self,
        image: torch.Tensor,
        activation: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        channels = activation.size(0)
        activation_maps = F.interpolate(
            activation.unsqueeze(0),
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        flat = activation_maps.flatten(start_dim=1)
        minima = flat.min(dim=1).values[:, None, None]
        ranges = flat.max(dim=1).values[:, None, None] - minima
        valid = ranges.flatten() > 0
        masks = torch.zeros_like(activation_maps)
        masks[valid] = (activation_maps[valid] - minima[valid]) / ranges[valid]

        weights = torch.zeros(channels, device=image.device, dtype=activation.dtype)
        for start in range(0, channels, self.batch_size):
            end = min(start + self.batch_size, channels)
            masked_images = image.unsqueeze(0) * masks[start:end].unsqueeze(1)
            probabilities = self.model(masked_images).softmax(dim=1)
            weights[start:end] = probabilities[:, target]

        saliency = (weights[:, None, None] * activation_maps).sum(dim=0, keepdim=True)
        return F.relu(saliency).unsqueeze(0)


class AblationCAM(SaliencyMethod):
    """Generate Ablation-CAM using drops in pre-softmax class scores."""

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module | None = None,
        batch_size: int = 32,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(model)
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        self.target_layer = target_layer
        self.batch_size = batch_size
        self.eps = eps

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        validate_images(images)
        images = images.to(model_device(self.model))
        target_layer = self.target_layer or find_spatial_target_layer(self.model)
        was_training = self.model.training
        self.model.eval()

        try:
            activations, logits = capture_activations(self.model, target_layer, images)
            targets = _targets_from_logits(logits, targets)
            original_scores = gather_scores(logits, targets)
            maps = [
                self._single_image(
                    image,
                    activation,
                    target,
                    original_score,
                    target_layer,
                )
                for image, activation, target, original_score in zip(
                    images, activations, targets, original_scores
                )
            ]
            return torch.cat(maps).detach()
        finally:
            if was_training:
                self.model.train()

    def _single_image(
        self,
        image: torch.Tensor,
        activation: torch.Tensor,
        target: torch.Tensor,
        original_score: torch.Tensor,
        target_layer: nn.Module,
    ) -> torch.Tensor:
        channels = activation.size(0)
        ablated_scores = torch.empty(
            channels, device=image.device, dtype=activation.dtype
        )

        for start in range(0, channels, self.batch_size):
            channel_indices = torch.arange(
                start, min(start + self.batch_size, channels), device=image.device
            )

            def ablate_channels(_module, _inputs, output):
                modified = output.clone()
                rows = torch.arange(output.size(0), device=output.device)
                modified[rows, channel_indices] = 0
                return modified

            handle = target_layer.register_forward_hook(ablate_channels)
            try:
                repeated = image.unsqueeze(0).expand(channel_indices.numel(), -1, -1, -1)
                logits = self.model(repeated)
            finally:
                handle.remove()
            ablated_scores[start : start + channel_indices.numel()] = logits[:, target]

        denominator = _nonzero_denominator(original_score, self.eps)
        weights = (original_score - ablated_scores) / denominator
        low_resolution = (weights[:, None, None] * activation).sum(dim=0, keepdim=True)
        saliency = upsample_to_input(F.relu(low_resolution).unsqueeze(0), image.unsqueeze(0))
        return saliency


__all__ = ["AblationCAM", "ScoreCAM"]
