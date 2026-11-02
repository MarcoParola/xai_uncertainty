"""Predictive entropy for classifier uncertainty."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def entropy(
    scores: torch.Tensor,
    *,
    from_logits: bool = True,
    normalize: bool = False,
    eps: float = 1e-12,
    reduction: str = "none",
) -> torch.Tensor:
    """Compute per-sample predictive entropy.

    Args:
        scores: Tensor with shape ``(batch, classes)``. Interpreted as logits
            when ``from_logits=True`` and probabilities otherwise.
        from_logits: Convert scores with softmax before computing entropy.
        normalize: Divide by ``log(num_classes)`` so values are in ``[0, 1]``.
        eps: Small clamp value for numerical stability.
        reduction: One of ``"none"``, ``"mean"``, or ``"sum"``.
    """
    if scores.ndim != 2:
        raise ValueError(f"Expected scores with shape (B, C), got {tuple(scores.shape)}.")

    probs = F.softmax(scores, dim=1) if from_logits else scores
    probs = probs.clamp_min(eps)
    values = -(probs * probs.log()).sum(dim=1)

    if normalize:
        if scores.size(1) < 2:
            raise ValueError("Cannot normalize entropy with fewer than two classes.")
        values = values / math.log(scores.size(1))

    if reduction == "none":
        return values
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    raise ValueError("reduction must be one of 'none', 'mean', or 'sum'.")


__all__ = ["entropy"]
