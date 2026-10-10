"""Grid-mask LIME saliency maps."""

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


class LIME(SaliencyMethod):
    """LIME-style local explanation using grid patches as interpretable features."""

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 1000,
        grid_size: tuple[int, int] = (7, 7),
        p: float = 0.5,
        kernel_width: float = 0.25,
        batch_size: int = 64,
        ridge: float = 1e-3,
        baseline: float = 0.0,
        positive_only: bool = True,
        use_softmax: bool = True,
    ) -> None:
        super().__init__(model)
        self.n_samples = n_samples
        self.grid_size = grid_size
        self.p = p
        self.kernel_width = kernel_width
        self.batch_size = batch_size
        self.ridge = ridge
        self.baseline = baseline
        self.positive_only = positive_only
        self.use_softmax = use_softmax

    def sample_feature_masks(self, device: torch.device) -> torch.Tensor:
        n_features = self.grid_size[0] * self.grid_size[1]
        masks = torch.bernoulli(
            torch.full((self.n_samples, n_features), self.p, device=device)
        )
        masks[0].fill_(1.0)
        return masks

    def masks_to_images(
        self,
        feature_masks: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        masks = feature_masks.view(-1, 1, *self.grid_size)
        return F.interpolate(
            masks,
            size=(height, width),
            mode="nearest",
        )

    def kernel_weights(self, feature_masks: torch.Tensor) -> torch.Tensor:
        distances = 1.0 - feature_masks.mean(dim=1)
        return torch.exp(-(distances**2) / (self.kernel_width**2))

    def fit_weighted_ridge(
        self,
        feature_masks: torch.Tensor,
        scores: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        ones = torch.ones(feature_masks.size(0), 1, device=feature_masks.device)
        design = torch.cat([ones, feature_masks], dim=1)
        weighted_design = design * weights.sqrt().unsqueeze(1)
        weighted_scores = scores * weights.sqrt()
        penalty = torch.eye(design.size(1), device=design.device) * self.ridge
        penalty[0, 0] = 0.0

        lhs = weighted_design.T @ weighted_design + penalty
        rhs = weighted_design.T @ weighted_scores
        coefficients = torch.linalg.solve(lhs, rhs)
        return coefficients[1:]

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

        saliency_maps = []
        for image, target in zip(images, targets):
            feature_masks = self.sample_feature_masks(device)
            image_masks = self.masks_to_images(feature_masks, height, width)
            scores = []

            for start in range(0, self.n_samples, self.batch_size):
                current_masks = image_masks[start : start + self.batch_size]
                perturbed = image.unsqueeze(0) * current_masks
                if self.baseline != 0:
                    perturbed = perturbed + self.baseline * (1.0 - current_masks)

                logits = self.model(perturbed)
                if self.use_softmax:
                    logits = logits.softmax(dim=1)
                scores.append(logits[:, target])

            scores_tensor = torch.cat(scores)
            weights = self.kernel_weights(feature_masks)
            coefficients = self.fit_weighted_ridge(feature_masks, scores_tensor, weights)

            if self.positive_only:
                coefficients = coefficients.clamp_min(0)

            low_res_map = coefficients.view(1, 1, *self.grid_size)
            saliency = F.interpolate(
                low_res_map,
                size=(height, width),
                mode="nearest",
            )
            saliency_maps.append(saliency)

        if was_training:
            self.model.train()
        return normalize_maps(torch.cat(saliency_maps, dim=0))


__all__ = ["LIME"]
