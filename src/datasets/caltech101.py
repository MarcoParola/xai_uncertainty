"""Caltech101 dataset splits."""

from __future__ import annotations

from pathlib import Path

from .common import (
    BBoxDataset,
    DEFAULT_DATA_ROOT,
    DEFAULT_SEED,
    DEFAULT_TEST_FRACTION,
    DEFAULT_VAL_FRACTION,
    bbox_from_original_box,
    import_torchvision_datasets,
    random_split,
    train_val_test_lengths,
)


def get_dataset(
    root: str | Path = DEFAULT_DATA_ROOT,
    transform=None,
    target_transform=None,
    download: bool = False,
    seed: int = DEFAULT_SEED,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    return_bboxes: bool = False,
):
    datasets = import_torchvision_datasets()
    full_dataset = datasets.Caltech101(
        root=str(root),
        target_type="category",
        transform=transform,
        target_transform=target_transform,
        download=download,
    )
    if return_bboxes:
        ensure_annotations(full_dataset)
        full_dataset = BBoxDataset(
            full_dataset,
            lambda index: caltech101_bbox(full_dataset, index),
        )

    lengths = train_val_test_lengths(
        len(full_dataset),
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    return random_split(full_dataset, lengths, seed)


def ensure_annotations(dataset) -> None:
    annotations_dir = Path(dataset.root) / "Annotations"
    if not annotations_dir.is_dir():
        raise FileNotFoundError(
            f"Caltech101 annotations not found at {annotations_dir}. "
            "Run `env/bin/python download/download_dataset.py caltech101 --root data --split all`."
        )


def caltech101_bbox(dataset, index: int):
    import scipy.io

    category = dataset.categories[dataset.y[index]]
    annotation_category = dataset.annotation_categories[dataset.y[index]]
    image_index = dataset.index[index]
    image_path = Path(dataset.root) / "101_ObjectCategories" / category / f"image_{image_index:04d}.jpg"
    annotation_path = (
        Path(dataset.root)
        / "Annotations"
        / annotation_category
        / f"annotation_{image_index:04d}.mat"
    )
    contour = scipy.io.loadmat(annotation_path)["obj_contour"]
    x_values = contour[0]
    y_values = contour[1]
    box = (
        float(x_values.min()),
        float(y_values.min()),
        float(x_values.max()),
        float(y_values.max()),
    )
    return bbox_from_original_box(image_path, box)
