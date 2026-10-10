"""Pointing game metric for saliency maps and bounding boxes."""

from __future__ import annotations

import torch

from .common import as_single_channel


def pointing_game(
    saliency_maps: torch.Tensor,
    bboxes: torch.Tensor,
    *,
    tolerance: int = 0,
) -> dict[str, torch.Tensor]:
    """Check whether each saliency maximum falls inside the target bbox.

    ``saliency_maps`` may have shape ``(B, H, W)`` or ``(B, 1, H, W)``.
    ``bboxes`` must have shape ``(B, 4)`` in ``x1, y1, x2, y2`` format in
    saliency-map pixel coordinates. ``tolerance`` expands boxes by that many
    pixels before scoring.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0.")
    if bboxes.ndim != 2 or bboxes.size(1) != 4:
        raise ValueError(f"Expected bboxes with shape (B, 4), got {tuple(bboxes.shape)}.")

    saliency_maps = as_single_channel(saliency_maps, "saliency_maps")
    batch_size, _channels, height, width = saliency_maps.shape
    if bboxes.size(0) != batch_size:
        raise ValueError("Saliency maps and bboxes must have the same batch size.")

    flat_index = saliency_maps.flatten(start_dim=1).argmax(dim=1)
    row = flat_index // width
    col = flat_index % width
    bboxes = bboxes.to(device=saliency_maps.device, dtype=torch.float32)

    x1 = bboxes[:, 0] - tolerance
    y1 = bboxes[:, 1] - tolerance
    x2 = bboxes[:, 2] + tolerance
    y2 = bboxes[:, 3] + tolerance
    hits = (col >= x1) & (col < x2) & (row >= y1) & (row < y2)

    return {
        "hits": hits.detach().cpu(),
        "score": hits.float().mean().detach().cpu(),
        "bboxes": bboxes.detach().cpu(),
        "points": torch.stack([row, col], dim=1).detach().cpu(),
    }


__all__ = ["pointing_game"]

