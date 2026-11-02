"""Brier score for classifier uncertainty."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def brier_score(
    scores: torch.Tensor,
    targets: torch.Tensor,
    *,
    from_logits: bool = True,
    reduction: str = "none",
) -> torch.Tensor:
    """Compute the multiclass Brier score per sample."""
    if scores.ndim != 2:
        raise ValueError(f"Expected scores with shape (B, C), got {tuple(scores.shape)}.")
    if targets.ndim != 1 or targets.size(0) != scores.size(0):
        raise ValueError(f"Expected targets with shape (B,), got {tuple(targets.shape)}.")

    probs = F.softmax(scores, dim=1) if from_logits else scores
    one_hot = F.one_hot(targets, num_classes=scores.size(1)).to(dtype=probs.dtype)
    values = (probs - one_hot).pow(2).sum(dim=1)

    if reduction == "none":
        return values
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    raise ValueError("reduction must be one of 'none', 'mean', or 'sum'.")


__all__ = ["brier_score"]
