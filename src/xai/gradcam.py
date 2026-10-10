"""Grad-CAM family saliency maps."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .base import (
    SaliencyMethod,
    gather_scores,
    model_device,
    normalize_maps,
    upsample_to_input,
    validate_images,
)


def find_last_conv2d(model: nn.Module) -> nn.Module:
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module

    if last_conv is None:
        raise ValueError(
            "Could not automatically find a Conv2d layer for Grad-CAM. "
            "Pass target_layer explicitly."
        )
    return last_conv


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
        activations, gradients = self._activations_and_gradients(images, targets)
        weights = self._weights(activations, gradients)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = upsample_to_input(F.relu(cam), images)
        return normalize_maps(cam).detach()

    def _weights(self, _activations: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
        return gradients.mean(dim=(2, 3), keepdim=True)

    def _activations_and_gradients(
        self,
        images: torch.Tensor,
        targets: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = model_device(self.model)
        target_layer = self.target_layer or find_last_conv2d(self.model)
        captured_activations = None
        captured_gradients = None

        def forward_hook(_module, _inputs, output):
            nonlocal captured_activations
            captured_activations = output

        def backward_hook(_module, _grad_inputs, grad_outputs):
            nonlocal captured_gradients
            captured_gradients = grad_outputs[0]

        handles = [
            target_layer.register_forward_hook(forward_hook),
            target_layer.register_full_backward_hook(backward_hook),
        ]

        was_training = self.model.training
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        try:
            logits = self.model(images)
            targets = logits.argmax(dim=1) if targets is None else targets.to(device)
            gather_scores(logits, targets).sum().backward()

            if captured_activations is None or captured_gradients is None:
                raise RuntimeError("CAM hooks did not capture activations/gradients.")
            return captured_activations, captured_gradients
        finally:
            for handle in handles:
                handle.remove()
            self.model.zero_grad(set_to_none=True)
            if was_training:
                self.model.train()


class GradCAMPlusPlus(GradCAM):
    """Generate Grad-CAM++ maps for a batch of images."""

    def _weights(
        self,
        activations: torch.Tensor,
        gradients: torch.Tensor,
    ) -> torch.Tensor:
        grad_sq = gradients.pow(2)
        grad_cube = gradients.pow(3)
        denominator = 2.0 * grad_sq + (activations * grad_cube).sum(
            dim=(2, 3),
            keepdim=True,
        )
        alphas = grad_sq / (denominator + 1e-8)
        return (alphas * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)


class XGradCAM(GradCAM):
    """Generate XGrad-CAM maps for a batch of images."""

    def _weights(
        self,
        activations: torch.Tensor,
        gradients: torch.Tensor,
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
        return normalize_maps(saliency).detach()

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
