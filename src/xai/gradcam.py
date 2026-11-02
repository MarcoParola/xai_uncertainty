"""Grad-CAM family saliency maps."""

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


def find_last_conv2d(model: nn.Module) -> nn.Module:
    """Backward-compatible alias for CNN target-layer discovery."""
    return find_spatial_target_layer(model)


class GradCAM(SaliencyMethod):
    """Generate Grad-CAM maps for a batch of images."""

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None) -> None:
        super().__init__(model)
        self.target_layer = target_layer

    def generate(
        self,
        images: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        validate_images(images)
        device = model_device(self.model)
        images = images.to(device)
        activations, gradients, target_scores = self._activations_and_gradients(
            images, targets
        )
        weights = self._weights(activations, gradients, target_scores)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = upsample_to_input(F.relu(cam), images)
        return cam.detach()

    def _weights(
        self,
        _activations: torch.Tensor,
        gradients: torch.Tensor,
        _target_scores: torch.Tensor,
    ) -> torch.Tensor:
        return gradients.mean(dim=(2, 3), keepdim=True)

    def _activations_and_gradients(
        self,
        images: torch.Tensor,
        targets: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_layer = self.target_layer or find_spatial_target_layer(self.model)
        was_training = self.model.training
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        try:
            activations, logits = capture_activations(self.model, target_layer, images)
            targets = (
                logits.argmax(dim=1)
                if targets is None
                else targets.to(device=images.device, dtype=torch.long)
            )
            target_scores = gather_scores(logits, targets)
            gradients = torch.autograd.grad(
                target_scores.sum(),
                activations,
                retain_graph=False,
            )[0]
            return activations.detach(), gradients.detach(), target_scores.detach()
        finally:
            self.model.zero_grad(set_to_none=True)
            if was_training:
                self.model.train()


class GradCAMPlusPlus(GradCAM):
    """Generate Grad-CAM++ maps for a batch of images."""

    score_type = "exp_logit"

    def _weights(
        self,
        activations: torch.Tensor,
        gradients: torch.Tensor,
        target_scores: torch.Tensor,
    ) -> torch.Tensor:
        exponentiated_score = target_scores.exp().view(-1, 1, 1, 1)
        first_derivative = exponentiated_score * gradients
        second_derivative = first_derivative * gradients
        third_derivative = second_derivative * gradients
        denominator = 2.0 * second_derivative + third_derivative * activations.sum(
            dim=(2, 3), keepdim=True
        )
        alphas = torch.where(
            second_derivative != 0,
            second_derivative / (denominator + 1e-8),
            torch.zeros_like(second_derivative),
        )
        return (alphas * F.relu(first_derivative)).sum(
            dim=(2, 3), keepdim=True
        )


class XGradCAM(GradCAM):
    """Generate XGrad-CAM maps for a batch of images."""

    def _weights(
        self,
        activations: torch.Tensor,
        gradients: torch.Tensor,
        _target_scores: torch.Tensor,
    ) -> torch.Tensor:
        numerator = (gradients * activations).sum(dim=(2, 3), keepdim=True)
        denominator = activations.sum(dim=(2, 3), keepdim=True)
        return numerator / (denominator + 1e-8)


class GuidedGradCAM(GradCAM):
    """Generate Guided Grad-CAM maps for a batch of images."""

    def generate(
        self,
        images: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        validate_images(images)
        device = model_device(self.model)
        images = images.to(device)
        cam = super().generate(images, targets=targets)
        guided_grads = self._guided_backprop(images, targets)
        saliency = cam * guided_grads.abs().mean(dim=1, keepdim=True)
        return saliency.detach()

    def _guided_backprop(
        self,
        images: torch.Tensor,
        targets: torch.Tensor | None,
    ) -> torch.Tensor:
        device = model_device(self.model)
        images = images.detach().clone().to(device).requires_grad_(True)
        handles = []
        relu_inplace_states = []

        def relu_backward_hook(_module, grad_inputs, _grad_outputs):
            if grad_inputs[0] is None:
                return grad_inputs
            return (F.relu(grad_inputs[0]),)

        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                relu_inplace_states.append((module, module.inplace))
                module.inplace = False
                handles.append(module.register_full_backward_hook(relu_backward_hook))

        was_training = self.model.training
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        try:
            logits = self.model(images)
            if targets is None:
                targets = logits.argmax(dim=1)
            else:
                targets = targets.to(device)

            gather_scores(logits, targets).sum().backward()
            if images.grad is None:
                raise RuntimeError("Guided backpropagation did not produce image gradients.")
            return images.grad.detach()
        finally:
            for handle in handles:
                handle.remove()
            for module, inplace in relu_inplace_states:
                module.inplace = inplace
            self.model.zero_grad(set_to_none=True)
            if was_training:
                self.model.train()


__all__ = [
    "GradCAM",
    "GradCAMPlusPlus",
    "GuidedGradCAM",
    "XGradCAM",
    "find_last_conv2d",
]
