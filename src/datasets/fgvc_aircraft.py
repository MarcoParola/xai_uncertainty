"""FGVC Aircraft dataset splits."""

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
    kwargs = {
        "root": str(root),
        "annotation_level": "variant",
        "transform": transform,
        "target_transform": target_transform,
        "download": download,
    }

    train = datasets.FGVCAircraft(split="train", **kwargs)
    val = datasets.FGVCAircraft(split="val", **kwargs)
    test = datasets.FGVCAircraft(split="test", **kwargs)
    if return_bboxes:
        boxes = aircraft_boxes(train)
        train = BBoxDataset(train, lambda index: fgvc_aircraft_bbox(train, index, boxes))
        val = BBoxDataset(val, lambda index: fgvc_aircraft_bbox(val, index, boxes))
        test = BBoxDataset(test, lambda index: fgvc_aircraft_bbox(test, index, boxes))
    return train, val, test


def aircraft_boxes(dataset) -> dict[str, tuple[float, float, float, float]]:
    boxes_path = Path(dataset._data_path) / "data" / "images_box.txt"
    boxes = {}
    with boxes_path.open(encoding="utf-8") as handle:
        for line in handle:
            image_id, x1, y1, x2, y2 = line.strip().split()
            boxes[image_id] = (float(x1), float(y1), float(x2), float(y2))
    return boxes


def fgvc_aircraft_bbox(dataset, index: int, boxes: dict[str, tuple[float, float, float, float]]):
    image_path = Path(dataset._image_files[index])
    return bbox_from_original_box(image_path, boxes[image_path.stem])
