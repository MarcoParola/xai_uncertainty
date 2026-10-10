#!/usr/bin/env python3
"""Load supported datasets and print train/val/test cardinalities."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import get_dataset  # noqa: E402
from src.datasets.common import DEFAULT_DATA_ROOT  # noqa: E402


DATASETS = (
    "oxford_pet",
    "caltech101",
    "fgvc_aircraft",
    "stanford_cars",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load all supported datasets and print split cardinalities."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Dataset root. Defaults to {DEFAULT_DATA_ROOT}.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise immediately if any dataset cannot be loaded.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for dataset_name in DATASETS:
        try:
            train, val, test = get_dataset(dataset_name, root=args.root)
        except Exception as exc:
            if args.strict:
                raise
            print(f"{dataset_name}: unavailable ({exc})")
            continue

        print(
            f"{dataset_name}: "
            f"train={len(train)}, val={len(val)}, test={len(test)}"
        )


if __name__ == "__main__":
    main()
