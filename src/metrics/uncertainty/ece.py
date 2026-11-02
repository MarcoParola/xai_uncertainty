"""Expected Calibration Error for classifier predictions."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def expected_calibration_error(
    scores: torch.Tensor,
    targets: torch.Tensor,
    *,
    n_bins: int = 15,
    from_logits: bool = True,
) -> torch.Tensor:
    """Compute top-label ECE using equally spaced confidence bins."""
    if scores.ndim != 2:
        raise ValueError(f"Expected scores with shape (B, C), got {tuple(scores.shape)}.")
    if targets.ndim != 1 or targets.size(0) != scores.size(0):
        raise ValueError(f"Expected targets with shape (B,), got {tuple(targets.shape)}.")
    if scores.size(1) < 1:
        raise ValueError("Expected at least one class.")
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")
    if scores.size(0) == 0:
        return scores.new_zeros(())

    probabilities = F.softmax(scores, dim=1) if from_logits else scores
    confidences, predictions = probabilities.max(dim=1)
    correct = predictions.eq(targets).to(dtype=confidences.dtype)

    boundaries = torch.linspace(
        0,
        1,
        n_bins + 1,
        device=confidences.device,
        dtype=confidences.dtype,
    )
    bin_indices = torch.bucketize(confidences, boundaries[1:-1], right=False)
    counts = torch.bincount(bin_indices, minlength=n_bins).to(confidences.dtype)
    confidence_sums = torch.zeros_like(counts).scatter_add_(
        0,
        bin_indices,
        confidences,
    )
    accuracy_sums = torch.zeros_like(counts).scatter_add_(0, bin_indices, correct)

    populated = counts > 0
    mean_confidence = confidence_sums[populated] / counts[populated]
    mean_accuracy = accuracy_sums[populated] / counts[populated]
    weights = counts[populated] / scores.size(0)
    return (weights * (mean_accuracy - mean_confidence).abs()).sum()


__all__ = ["expected_calibration_error"]
