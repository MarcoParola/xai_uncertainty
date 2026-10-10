"""Shared helpers for perturbation-based XAI metrics."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F


ScoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def validate_inputs(images: torch.Tensor, saliency_maps: torch.Tensor) -> torch.Tensor:
    if images.ndim != 4:
        raise ValueError(f"Expected images with shape (B, C, H, W), got {tuple(images.shape)}.")

    if saliency_maps.ndim == 3:
        saliency_maps = saliency_maps.unsqueeze(1)
    if saliency_maps.ndim != 4:
        raise ValueError(
            "Expected saliency maps with shape (B, 1, H, W) or (B, H, W), "
            f"got {tuple(saliency_maps.shape)}."
        )
    if saliency_maps.size(0) != images.size(0):
        raise ValueError("Images and saliency maps must have the same batch size.")
    if saliency_maps.size(1) != 1:
        raise ValueError("Saliency maps must have one channel.")

    if saliency_maps.shape[-2:] != images.shape[-2:]:
        saliency_maps = F.interpolate(
            saliency_maps,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    return saliency_maps


def make_baseline(images: torch.Tensor, baseline: float | torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(baseline):
        return baseline.to(device=images.device, dtype=images.dtype).expand_as(images)
    return torch.full_like(images, float(baseline))


def class_scores(
    logits: torch.Tensor,
    targets: torch.Tensor,
    score_mode: str,
) -> torch.Tensor:
    if score_mode == "logit":
        scores = logits
    elif score_mode == "probability":
        scores = logits.softmax(dim=1)
    else:
        raise ValueError("score_mode must be either 'probability' or 'logit'.")
    return scores.gather(1, targets.view(-1, 1)).squeeze(1)


@torch.no_grad()
def infer_targets(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    return model(images).argmax(dim=1)


def saliency_order(saliency_maps: torch.Tensor) -> torch.Tensor:
    return saliency_maps.flatten(start_dim=1).argsort(dim=1, descending=True)


def topk_masks(order: torch.Tensor, n_pixels: int, steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    device = order.device
    fractions = torch.linspace(0.0, 1.0, steps + 1, device=device)
    pixel_counts = torch.round(fractions * n_pixels).long()
    masks = torch.zeros(order.size(0), steps + 1, n_pixels, dtype=torch.bool, device=device)

    for step, pixel_count in enumerate(pixel_counts.tolist()):
        if pixel_count > 0:
            masks[:, step].scatter_(1, order[:, :pixel_count], True)
    return masks, fractions


@torch.no_grad()
def score_perturbed_images(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    score_mode: str,
    batch_size: int,
) -> torch.Tensor:
    scores = []
    for start in range(0, images.size(0), batch_size):
        logits = model(images[start : start + batch_size])
        current_targets = targets[start : start + batch_size]
        scores.append(class_scores(logits, current_targets, score_mode))
    return torch.cat(scores)


def auc(curves: torch.Tensor, fractions: torch.Tensor) -> torch.Tensor:
    return torch.trapezoid(curves, fractions, dim=1)


def restore_training_state(model: nn.Module, was_training: bool) -> None:
    if was_training:
        model.train()


def as_single_channel(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
    if tensor.ndim != 4:
        raise ValueError(
            f"Expected {name} with shape (B, 1, H, W) or (B, H, W), "
            f"got {tuple(tensor.shape)}."
        )
    if tensor.size(1) != 1:
        raise ValueError(f"{name} must have one channel.")
    return tensor
