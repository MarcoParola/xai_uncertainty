"""Compactness metrics for batched saliency maps."""

from __future__ import annotations

import torch

from .common import as_single_channel


def _magnitudes(saliency_maps: torch.Tensor) -> torch.Tensor:
    saliency_maps = as_single_channel(saliency_maps, "saliency_maps")
    return saliency_maps.detach().to(dtype=torch.float32).abs().flatten(start_dim=1)


def compactness_metrics(saliency_maps: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute per-sample compactness metrics over saliency magnitude.

    Shannon entropy is reported in nats. The Gini coefficient uses the
    population definition, whose maximum for ``N`` pixels is ``1 - 1/N``.
    All-zero maps receive zero for every metric.
    """
    values = _magnitudes(saliency_maps)
    n_pixels = values.size(1)
    if n_pixels == 0:
        raise ValueError("Saliency maps must contain at least one pixel.")

    total = values.sum(dim=1)
    nonzero = total > 0
    safe_total = total.clamp_min(torch.finfo(values.dtype).tiny)

    mean = values.mean(dim=1)
    sparsity = torch.where(nonzero, values.max(dim=1).values / mean, 0.0)

    probabilities = values / safe_total.unsqueeze(1)
    log_probabilities = probabilities.clamp_min(torch.finfo(values.dtype).tiny).log()
    entropy = -(probabilities * log_probabilities).sum(dim=1)
    entropy = torch.where(nonzero, entropy, 0.0)

    sorted_values = values.sort(dim=1).values
    ranks = torch.arange(
        1,
        n_pixels + 1,
        device=values.device,
        dtype=values.dtype,
    )
    weighted_sum = (sorted_values * ranks).sum(dim=1)
    gini = 2.0 * weighted_sum / (n_pixels * safe_total) - (n_pixels + 1) / n_pixels
    gini = torch.where(nonzero, gini.clamp(min=0.0, max=1.0), 0.0)

    return {
        "sparsity_ratio": sparsity.cpu(),
        "saliency_entropy": entropy.cpu(),
        "saliency_gini": gini.cpu(),
    }


def sparsity_ratio(saliency_maps: torch.Tensor) -> torch.Tensor:
    """Return ``max(abs(S)) / mean(abs(S))`` for every saliency map."""
    return compactness_metrics(saliency_maps)["sparsity_ratio"]


def saliency_entropy(saliency_maps: torch.Tensor) -> torch.Tensor:
    """Return Shannon entropy in nats for every saliency-magnitude distribution."""
    return compactness_metrics(saliency_maps)["saliency_entropy"]


def saliency_gini(saliency_maps: torch.Tensor) -> torch.Tensor:
    """Return the Gini coefficient for every saliency-magnitude distribution."""
    return compactness_metrics(saliency_maps)["saliency_gini"]


__all__ = [
    "compactness_metrics",
    "sparsity_ratio",
    "saliency_entropy",
    "saliency_gini",
]
