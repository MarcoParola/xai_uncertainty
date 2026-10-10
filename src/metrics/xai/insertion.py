"""Insertion metric for saliency maps."""

from __future__ import annotations

import torch
from torch import nn

from .common import (
    auc,
    infer_targets,
    make_baseline,
    restore_training_state,
    saliency_order,
    score_perturbed_images,
    topk_masks,
    validate_inputs,
)


def insertion(
    model: nn.Module,
    images: torch.Tensor,
    saliency_maps: torch.Tensor,
    targets: torch.Tensor | None = None,
    *,
    steps: int = 20,
    baseline: float | torch.Tensor = 0.0,
    score_mode: str = "probability",
    batch_size: int = 64,
) -> dict[str, torch.Tensor]:
    """Measure how fast confidence rises when salient pixels are inserted.

    Returns a dict with ``fractions``, ``curves``, and ``auc``. Curves have
    shape ``(B, steps + 1)`` and AUC has shape ``(B,)``.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1.")

    was_training = model.training
    model.eval()

    try:
        saliency_maps = validate_inputs(images, saliency_maps)
        device = next(model.parameters()).device
        images = images.to(device)
        saliency_maps = saliency_maps.to(device)
        targets = infer_targets(model, images) if targets is None else targets.to(device)

        batch_size_, channels, height, width = images.shape
        n_pixels = height * width
        order = saliency_order(saliency_maps)
        masks, fractions = topk_masks(order, n_pixels, steps)
        masks = masks.view(batch_size_, steps + 1, 1, height, width)

        baseline_images = make_baseline(images, baseline)
        perturbed = baseline_images[:, None] + (
            images[:, None] - baseline_images[:, None]
        ) * masks
        perturbed = perturbed.reshape(batch_size_ * (steps + 1), channels, height, width)
        repeated_targets = targets.repeat_interleave(steps + 1)

        scores = score_perturbed_images(
            model,
            perturbed,
            repeated_targets,
            score_mode=score_mode,
            batch_size=batch_size,
        )
        curves = scores.view(batch_size_, steps + 1)
        return {
            "fractions": fractions.detach().cpu(),
            "curves": curves.detach().cpu(),
            "auc": auc(curves, fractions).detach().cpu(),
        }
    finally:
        restore_training_state(model, was_training)


__all__ = ["insertion"]
