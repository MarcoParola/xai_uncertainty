"""Stanford Cars dataset splits."""

from __future__ import annotations

from pathlib import Path

from .common import (
    BBoxDataset,
    DEFAULT_DATA_ROOT,
    DEFAULT_SEED,
    DEFAULT_VAL_FRACTION,
    bbox_from_original_box,
    import_torchvision_datasets,
    random_split,
    train_val_lengths,
)


def get_dataset(
    root: str | Path = DEFAULT_DATA_ROOT,
    transform=None,
    target_transform=None,
    download: bool = False,
    seed: int = DEFAULT_SEED,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    test_fraction: float = 0.0,
    return_bboxes: bool = False,
):
    datasets = import_torchvision_datasets()
    ensure_stanford_cars_available(Path(root))

    train_full = datasets.StanfordCars(
        root=str(root),
        split="train",
        transform=transform,
        target_transform=target_transform,
        download=download,
    )
    test = datasets.StanfordCars(
        root=str(root),
        split="test",
        transform=transform,
        target_transform=target_transform,
        download=download,
    )

    if return_bboxes:
        train_boxes = stanford_car_boxes(train_full)
        test_boxes = stanford_car_boxes(test)
        train_full = BBoxDataset(
            train_full,
            lambda index: stanford_cars_bbox(train_full, index, train_boxes),
        )
        test = BBoxDataset(
            test,
            lambda index: stanford_cars_bbox(test, index, test_boxes),
        )

    train_size, val_size = train_val_lengths(len(train_full), val_fraction)
    train, val = random_split(train_full, (train_size, val_size), seed)
    return train, val, test


def ensure_stanford_cars_available(root: Path) -> None:
    base = root / "stanford_cars"
    required_paths = (
        base / "devkit" / "cars_train_annos.mat",
        base / "devkit" / "cars_meta.mat",
        base / "cars_train",
        base / "cars_test_annos_withlabels.mat",
        base / "cars_test",
    )
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Stanford Cars files are missing. Run "
            "`env/bin/python download/download_dataset.py stanford_cars --root data --split all` "
            f"to prepare them. Missing: {missing_text}"
        )


def stanford_car_boxes(dataset) -> list[tuple[float, float, float, float]]:
    import scipy.io as sio

    annotations = sio.loadmat(dataset._annotations_mat_path, squeeze_me=True)["annotations"]
    return [
        (
            float(annotation["bbox_x1"]),
            float(annotation["bbox_y1"]),
            float(annotation["bbox_x2"]),
            float(annotation["bbox_y2"]),
        )
        for annotation in annotations
    ]


def stanford_cars_bbox(dataset, index: int, boxes: list[tuple[float, float, float, float]]):
    image_path = dataset._samples[index][0]
    return bbox_from_original_box(image_path, boxes[index])
