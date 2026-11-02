"""Layer-wise relevance propagation."""

from __future__ import annotations

import torch
from torch import nn

from .base import SaliencyMethod, model_device, validate_images


class LRP(SaliencyMethod):
    """Propagate a target pre-softmax score to input pixels with Captum LRP."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)
        try:
            from captum.attr import LRP as CaptumLRP
        except ImportError as exc:
            raise ImportError(
                "LRP requires Captum. Install the project requirements first."
            ) from exc
        self.explainer = CaptumLRP(model)

    def generate(
        self,
        images: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        validate_images(images)
        images = images.to(model_device(self.model))
        was_training = self.model.training
        self.model.eval()

        try:
            if targets is None:
                with torch.no_grad():
                    targets = self.model(images).argmax(dim=1)
            else:
                targets = targets.to(device=images.device, dtype=torch.long)
            relevance = self.explainer.attribute(images, target=targets)
            return relevance.sum(dim=1, keepdim=True).detach()
        except (TypeError, RuntimeError) as exc:
            raise ValueError(
                f"LRP does not have propagation rules for every layer in "
                f"{type(self.model).__name__}."
            ) from exc
        finally:
            self.model.zero_grad(set_to_none=True)
            if was_training:
                self.model.train()


__all__ = ["LRP"]
