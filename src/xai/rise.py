"""RISE saliency maps."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .base import (
    SaliencyMethod,
    infer_targets,
    model_device,
    normalize_maps,
    validate_images,
)


class RISE(SaliencyMethod):
    """Randomized Input Sampling for Explanation.

    This implementation accepts a batch of images and returns a batch of saliency
    maps. The same random mask bank is reused for every image in the batch.
    """

    def __init__(
        self,
        model: nn.Module,
        n_masks: int = 4000,
        mask_size: tuple[int, int] = (7, 7),
        p: float = 0.5,
        batch_size: int = 64,
        use_softmax: bool = True,
    ) -> None:
        super().__init__(model)
        self.n_masks = n_masks
        self.mask_size = mask_size
        self.p = p
        self.batch_size = batch_size
        self.use_softmax = use_softmax

    def generate_masks(
        self,
        n_masks: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        masks = torch.bernoulli(
            torch.full((n_masks, 1, *self.mask_size), self.p, device=device)
        )
        return F.interpolate(
            masks,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        validate_images(images)
        device = model_device(self.model)
        images = images.to(device)
        batch_size, _channels, height, width = images.shape

        was_training = self.model.training
        self.model.eval()

        if targets is None:
            targets = infer_targets(self.model, images)
        else:
            targets = targets.to(device)

        masks = self.generate_masks(self.n_masks, height, width, device)
        saliency = torch.zeros(batch_size, 1, height, width, device=device)

        for start in range(0, self.n_masks, self.batch_size):
            current_masks = masks[start : start + self.batch_size]
            n_current = current_masks.size(0)
            masked_images = (
                images[:, None] * current_masks[None]
            ).reshape(batch_size * n_current, *images.shape[1:])

            logits = self.model(masked_images)
            if self.use_softmax:
                logits = logits.softmax(dim=1)

            repeated_targets = targets.repeat_interleave(n_current)
            scores = logits.gather(1, repeated_targets.view(-1, 1)).view(
                batch_size,
                n_current,
                1,
                1,
                1,
            )
            saliency += (scores * current_masks[None]).sum(dim=1)

        saliency = saliency / (self.n_masks * max(self.p, 1e-8))
        if was_training:
            self.model.train()
        return normalize_maps(saliency)


__all__ = ["RISE"]
