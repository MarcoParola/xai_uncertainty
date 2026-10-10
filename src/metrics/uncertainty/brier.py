"""Brier score metric placeholder.

TODO GALA:
- Implement the Brier score for classifier uncertainty evaluation.
- Expected inputs should likely be model logits/probabilities and integer labels,
  matching how the classifier and XAI evaluation scripts process test batches.
- Decide whether the function should return per-sample scores, an aggregate
  score, or both.
"""

from __future__ import annotations


def brier_score(*args, **kwargs):
    """TODO GALA: implement Brier score computation."""
    raise NotImplementedError("TODO GALA: implement Brier score.")


__all__ = ["brier_score"]
