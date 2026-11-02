"""Negative log-likelihood for classifier uncertainty."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def nll(
    scores: torch.Tensor,
    targets: torch.Tensor,
    *,
    from_logits: bool = True,
    eps: float = 1e-12,
    reduction: str = "none",
) -> torch.Tensor:
    """Compute per-sample negative log-likelihood."""
    if scores.ndim != 2:
        raise ValueError(f"Expected scores with shape (B, C), got {tuple(scores.shape)}.")
    if targets.ndim != 1 or targets.size(0) != scores.size(0):
        raise ValueError(f"Expected targets with shape (B,), got {tuple(targets.shape)}.")

    if from_logits:
        values = F.cross_entropy(scores, targets, reduction="none")
    else:
        values = -scores.gather(1, targets.view(-1, 1)).squeeze(1).clamp_min(eps).log()

    if reduction == "none":
        return values
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    raise ValueError("reduction must be one of 'none', 'mean', or 'sum'.")


__all__ = ["nll"]
