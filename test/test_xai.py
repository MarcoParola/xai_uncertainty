"""Regression tests for the shared saliency-method interface."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from src.xai import (
    AblationCAM,
    CAM,
    GradCAM,
    GradCAMPlusPlus,
    LIME,
    LRP,
    ScoreCAM,
    XGradCAM,
)


class TinyGAPClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(4, 6, 3, padding=1), nn.ReLU())
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(6, 3)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.layer4(self.stem(images))
        return self.fc(self.avgpool(features).flatten(1))


class SaliencyMethodTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = TinyGAPClassifier().eval()
        self.images = torch.randn(2, 3, 24, 24)
        self.targets = torch.tensor([0, 1])

    def test_all_methods_return_finite_batched_maps(self) -> None:
        methods = [
            CAM(self.model),
            GradCAM(self.model),
            XGradCAM(self.model),
            LRP(self.model),
            GradCAMPlusPlus(self.model),
            AblationCAM(self.model, batch_size=3),
            ScoreCAM(self.model, batch_size=3),
            LIME(self.model, n_samples=20, batch_size=10, random_seed=7),
        ]

        for method in methods:
            with self.subTest(method=type(method).__name__):
                maps = method.generate(self.images, self.targets)
                self.assertEqual(tuple(maps.shape), (2, 1, 24, 24))
                self.assertTrue(torch.isfinite(maps).all())

    def test_score_domains_are_explicit(self) -> None:
        self.assertEqual(CAM.score_type, "logit")
        self.assertEqual(GradCAM.score_type, "logit")
        self.assertEqual(XGradCAM.score_type, "logit")
        self.assertEqual(LRP.score_type, "logit")
        self.assertEqual(GradCAMPlusPlus.score_type, "exp_logit")
        self.assertEqual(AblationCAM.score_type, "logit")
        self.assertEqual(ScoreCAM.score_type, "probability")
        self.assertEqual(LIME.score_type, "probability")

if __name__ == "__main__":
    unittest.main()
