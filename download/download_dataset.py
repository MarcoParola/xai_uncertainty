#!/usr/bin/env python3
"""Download supported torchvision datasets from a single CLI."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data"
CALTECH101_URL = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip"
CALTECH101_MD5 = "3138e1922a9193bfa496528edbbc45d0"

ALIASES = {
    "oxford_pet": "oxford_pet",
    "oxford_pets": "oxford_pet",
    "oxford-pet": "oxford_pet",
    "oxford-pets": "oxford_pet",
    "oxfordiiitpet": "oxford_pet",
    "pet": "oxford_pet",
    "pets": "oxford_pet",
    "caltech101": "caltech101",
    "caltech-101": "caltech101",
    "calthech101": "caltech101",
    "fgvc_aircraft": "fgvc_aircraft",
    "fgvc-aircraft": "fgvc_aircraft",
    "aircraft": "fgvc_aircraft",
    "stanford_cars": "stanford_cars",
    "stanford-cars": "stanford_cars",
    "cars": "stanford_cars",
}

DISPLAY_NAMES = {
    "oxford_pet": "Oxford-IIIT Pet",
    "caltech101": "Caltech101",
    "fgvc_aircraft": "FGVC Aircraft",
    "stanford_cars": "Stanford Cars",
}

VALID_SPLITS = {
    "oxford_pet": ("trainval", "test", "all"),
    "caltech101": ("all",),
    "fgvc_aircraft": ("trainval", "train", "val", "test", "all"),
    "stanford_cars": ("train", "test", "all"),
}

ALL_SPLITS = {
    "oxford_pet": ("trainval", "test"),
    "caltech101": (None,),
    "fgvc_aircraft": ("trainval", "test"),
    "stanford_cars": ("train", "test"),
}


def split_label(split: str | None) -> str:
    return split or "all"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a supported dataset through torchvision.datasets."
    )
    parser.add_argument(
        "dataset",
        metavar="dataset",
        help="Dataset to download: oxford_pet, caltech101, fgvc_aircraft, stanford_cars.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Download root directory. Defaults to {DEFAULT_ROOT}.",
    )
    parser.add_argument(
        "--split",
        default="all",
        help="Dataset split to download. Use 'all' to download every standard split.",
    )
    return parser.parse_args()


def normalize_dataset(dataset: str) -> str:
    normalized = dataset.lower().replace(" ", "_")
    try:
        return ALIASES[normalized]
    except KeyError as exc:
        options = ", ".join(DISPLAY_NAMES.values())
        raise SystemExit(f"Unknown dataset '{dataset}'. Choose one of: {options}.") from exc


def validate_split(dataset: str, split: str) -> str:
    split = split.lower()
    valid_splits = VALID_SPLITS[dataset]
    if split not in valid_splits:
        options = ", ".join(valid_splits)
        raise SystemExit(
            f"{DISPLAY_NAMES[dataset]} does not support split '{split}'. "
            f"Choose one of: {options}."
        )
    return split


def load_torchvision_datasets():
    try:
        from torchvision import datasets
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "torchvision is not installed in the active environment. "
            "Install it in the virtualenv first, for example: "
            "env/bin/python -m pip install torch torchvision"
        ) from exc
    return datasets


def load_torchvision_dataset_utils():
    try:
        from torchvision.datasets.utils import download_and_extract_archive, extract_archive
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "torchvision is not installed in the active environment. "
            "Install it in the virtualenv first, for example: "
            "env/bin/python -m pip install -r requirements.txt"
        ) from exc
    return download_and_extract_archive, extract_archive


def dataset_split_exists(dataset: str, root: Path, split: str | None) -> bool:
    if dataset == "oxford_pet":
        base = root / "oxford-iiit-pet"
        return (
            (base / "images").is_dir()
            and (base / "annotations").is_dir()
            and (base / "annotations" / f"{split}.txt").is_file()
        )

    if dataset == "caltech101":
        base = root / "caltech101"
        return (
            (base / "101_ObjectCategories").is_dir()
            and (base / "Annotations").is_dir()
        )

    if dataset == "fgvc_aircraft":
        base = root / "fgvc-aircraft-2013b" / "data"
        return (
            (base / "images").is_dir()
            and (base / "variants.txt").is_file()
            and (base / f"images_variant_{split}.txt").is_file()
        )

    if dataset == "stanford_cars":
        base = root / "stanford_cars"
        if split == "train":
            return (
                (base / "devkit" / "cars_train_annos.mat").is_file()
                and (base / "devkit" / "cars_meta.mat").is_file()
                and (base / "cars_train").is_dir()
            )
        return (
            (base / "cars_test_annos_withlabels.mat").is_file()
            and (base / "devkit" / "cars_meta.mat").is_file()
            and (base / "cars_test").is_dir()
        )

    raise ValueError(f"Unsupported dataset: {dataset}")


def instantiate_dataset(datasets, dataset: str, root: Path, split: str | None, download: bool):
    if dataset == "oxford_pet":
        return datasets.OxfordIIITPet(
            root=str(root),
            split=split,
            target_types="category",
            download=download,
        )
    if dataset == "caltech101":
        return datasets.Caltech101(
            root=str(root),
            target_type="category",
            download=download,
        )
    if dataset == "fgvc_aircraft":
        return datasets.FGVCAircraft(
            root=str(root),
            split=split,
            annotation_level="variant",
            download=download,
        )
    if dataset == "stanford_cars":
        return datasets.StanfordCars(
            root=str(root),
            split=split,
            download=download,
        )
    raise ValueError(f"Unsupported dataset: {dataset}")


def ensure_caltech101_annotations(root: Path) -> None:
    base = root / "caltech101"
    if (base / "101_ObjectCategories").is_dir() and (base / "Annotations").is_dir():
        return

    download_and_extract_archive, extract_archive = load_torchvision_dataset_utils()
    base.mkdir(parents=True, exist_ok=True)
    print("Downloading Caltech101 annotations/object categories archive")
    download_and_extract_archive(
        CALTECH101_URL,
        download_root=str(base),
        filename="caltech-101.zip",
        md5=CALTECH101_MD5,
    )

    gzip_folder = base / "caltech-101"
    if gzip_folder.is_dir():
        for gzip_file in gzip_folder.iterdir():
            if gzip_file.suffix == ".gz":
                extract_archive(str(gzip_file), str(base))
        shutil.rmtree(gzip_folder)

    zip_path = base / "caltech-101.zip"
    if zip_path.exists():
        zip_path.unlink()


def download_dataset(dataset: str, root: Path, split: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    datasets = None

    splits = ALL_SPLITS[dataset] if split == "all" else (split,)
    for current_split in splits:
        current_label = split_label(current_split)
        if dataset_split_exists(dataset, root, current_split):
            print(f"Skipping {DISPLAY_NAMES[dataset]} split={current_label}: already exists")
            continue

        if dataset == "stanford_cars":
            print(
                f"Skipping {DISPLAY_NAMES[dataset]} split={current_label}: "
                "torchvision no longer provides a working download URL for this dataset. "
                f"Place it under {root / 'stanford_cars'} to use the loader."
            )
            continue

        if dataset == "caltech101":
            ensure_caltech101_annotations(root)
            if dataset_split_exists(dataset, root, current_split):
                print(f"Done {DISPLAY_NAMES[dataset]} split={current_label}: annotations available")
                continue

        if datasets is None:
            datasets = load_torchvision_datasets()

        print(f"Downloading {DISPLAY_NAMES[dataset]} split={current_label}")
        instantiate_dataset(
            datasets=datasets,
            dataset=dataset,
            root=root,
            split=current_split,
            download=True,
        )

    print(f"Done: {DISPLAY_NAMES[dataset]} checked under {root}")


def main() -> None:
    args = parse_args()
    dataset = normalize_dataset(args.dataset)
    split = validate_split(dataset, args.split)
    download_dataset(dataset=dataset, root=args.root, split=split)


if __name__ == "__main__":
    main()
