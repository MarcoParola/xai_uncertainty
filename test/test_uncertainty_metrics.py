"""Tests for uncertainty metrics."""

from __future__ import annotations

import unittest

import torch

from src.metrics.uncertainty import expected_calibration_error


class ExpectedCalibrationErrorTests(unittest.TestCase):
    def test_perfect_predictions_have_zero_ece(self) -> None:
        probabilities = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        targets = torch.tensor([0, 1])

        value = expected_calibration_error(
            probabilities,
            targets,
            n_bins=10,
            from_logits=False,
        )

        torch.testing.assert_close(value, torch.tensor(0.0))

    def test_top_label_ece_matches_known_value(self) -> None:
        probabilities = torch.tensor(
            [[0.90, 0.10], [0.80, 0.20], [0.60, 0.40], [0.55, 0.45]]
        )
        targets = torch.tensor([0, 1, 0, 1])

        value = expected_calibration_error(
            probabilities,
            targets,
            n_bins=2,
            from_logits=False,
        )

        torch.testing.assert_close(value, torch.tensor(0.2125))

    def test_logits_and_probabilities_are_equivalent(self) -> None:
        logits = torch.tensor([[2.0, -1.0], [0.1, 0.4], [-0.5, 1.5]])
        targets = torch.tensor([0, 0, 1])

        from_logits = expected_calibration_error(logits, targets, n_bins=5)
        from_probabilities = expected_calibration_error(
            logits.softmax(dim=1),
            targets,
            n_bins=5,
            from_logits=False,
        )

        torch.testing.assert_close(from_logits, from_probabilities)

    def test_invalid_bin_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            expected_calibration_error(
                torch.tensor([[0.5, 0.5]]),
                torch.tensor([0]),
                n_bins=0,
                from_logits=False,
            )


if __name__ == "__main__":
    unittest.main()
