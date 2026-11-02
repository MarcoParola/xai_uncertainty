"""Tests for the Imagenette split adapter."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.datasets.imagenette import get_dataset


class ImagenetteDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        dataset_root = self.root / "imagenette2-320"
        for split, count in (("train", 10), ("val", 4)):
            class_directory = dataset_root / split / "n01440764"
            class_directory.mkdir(parents=True)
            for index in range(count):
                Image.new("RGB", (8, 8), "white").save(
                    class_directory / f"image_{index}.jpeg"
                )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_official_validation_split_is_used_as_test(self) -> None:
        train, val, test = get_dataset(
            root=self.root,
            val_fraction=0.2,
            seed=7,
        )

        self.assertEqual((len(train), len(val), len(test)), (8, 2, 4))

    def test_split_is_deterministic(self) -> None:
        first_train, first_val, _ = get_dataset(root=self.root, val_fraction=0.2, seed=7)
        second_train, second_val, _ = get_dataset(root=self.root, val_fraction=0.2, seed=7)

        self.assertEqual(first_train.indices, second_train.indices)
        self.assertEqual(first_val.indices, second_val.indices)

    def test_bounding_boxes_are_explicitly_unavailable(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not provide localization annotations"):
            get_dataset(root=self.root, return_bboxes=True)


if __name__ == "__main__":
    unittest.main()
