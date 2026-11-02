"""Original image-LIME saliency maps."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .base import SaliencyMethod, model_device, validate_images


class LIME(SaliencyMethod):
    """Explain image predictions with the reference LIME image implementation."""

    score_type = "probability"

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 1000,
        kernel_width: float = 0.25,
        batch_size: int = 64,
        random_seed: int = 42,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__(model)
        if n_samples < 1 or batch_size < 1:
            raise ValueError("n_samples and batch_size must be positive.")
        try:
            from lime.lime_image import LimeImageExplainer
        except ImportError as exc:
            raise ImportError(
                "Image LIME requires the 'lime' package. Install the project "
                "requirements first."
            ) from exc

        self.n_samples = n_samples
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        self.explainer = LimeImageExplainer(
            kernel_width=kernel_width,
            random_state=random_seed,
        )

    def generate(
        self,
        images: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        validate_images(images)
        device = model_device(self.model)
        images = images.to(device)
        was_training = self.model.training
        self.model.eval()

        try:
            if targets is None:
                with torch.no_grad():
                    targets = self.model(images).argmax(dim=1)
            else:
                targets = targets.to(device=device, dtype=torch.long)

            rgb_images = self._to_rgb(images)
            maps = [
                self._explain_image(image, int(target))
                for image, target in zip(rgb_images, targets)
            ]
            return torch.stack(maps).unsqueeze(1).to(device)
        finally:
            if was_training:
                self.model.train()

    def _to_rgb(self, images: torch.Tensor) -> np.ndarray:
        mean = self.mean.to(device=images.device, dtype=images.dtype)
        std = self.std.to(device=images.device, dtype=images.dtype)
        rgb = (images * std + mean).clamp(0, 1)
        return rgb.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.float64)

    def _predict_probabilities(self, images: np.ndarray) -> np.ndarray:
        device = model_device(self.model)
        tensor = torch.from_numpy(images).permute(0, 3, 1, 2).to(
            device=device,
            dtype=torch.float32,
        )
        mean = self.mean.to(device=device, dtype=tensor.dtype)
        std = self.std.to(device=device, dtype=tensor.dtype)
        tensor = (tensor - mean) / std
        with torch.no_grad():
            probabilities = self.model(tensor).softmax(dim=1)
        return probabilities.cpu().numpy()

    def _explain_image(self, image: np.ndarray, target: int) -> torch.Tensor:
        explanation = self.explainer.explain_instance(
            image,
            self._predict_probabilities,
            labels=(target,),
            top_labels=None,
            num_features=image.shape[0] * image.shape[1],
            num_samples=self.n_samples,
            batch_size=self.batch_size,
            random_seed=self.random_seed,
        )
        coefficients = dict(explanation.local_exp[target])
        saliency = np.zeros(explanation.segments.shape, dtype=np.float32)
        for segment, coefficient in coefficients.items():
            saliency[explanation.segments == segment] = coefficient
        return torch.from_numpy(saliency)


__all__ = ["LIME"]
