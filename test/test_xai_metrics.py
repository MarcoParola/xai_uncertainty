"""Tests for saliency-map evaluation metrics."""

from __future__ import annotations

import math
import unittest

import torch

from src.metrics.xai import compactness_metrics


class CompactnessMetricTests(unittest.TestCase):
    def test_uniform_map(self) -> None:
        metrics = compactness_metrics(torch.ones(1, 1, 2, 2))

        torch.testing.assert_close(metrics["sparsity_ratio"], torch.tensor([1.0]))
        torch.testing.assert_close(
            metrics["saliency_entropy"],
            torch.tensor([math.log(4.0)]),
        )
        torch.testing.assert_close(metrics["saliency_gini"], torch.tensor([0.0]))

    def test_single_pixel_map(self) -> None:
        saliency = torch.tensor([[[-2.0, 0.0], [0.0, 0.0]]])
        metrics = compactness_metrics(saliency)

        torch.testing.assert_close(metrics["sparsity_ratio"], torch.tensor([4.0]))
        torch.testing.assert_close(metrics["saliency_entropy"], torch.tensor([0.0]))
        torch.testing.assert_close(metrics["saliency_gini"], torch.tensor([0.75]))

    def test_zero_map_is_finite(self) -> None:
        metrics = compactness_metrics(torch.zeros(2, 3, 3))

        for values in metrics.values():
            torch.testing.assert_close(values, torch.zeros(2))
            self.assertTrue(torch.isfinite(values).all())


if __name__ == "__main__":
    unittest.main()
