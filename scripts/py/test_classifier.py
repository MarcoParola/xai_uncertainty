#!/usr/bin/env python3
"""Evaluate a trained classifier checkpoint."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch import nn

from classifier_utils import (
    DEFAULT_DATA_ROOT,
    build_classifier,
    build_transform,
    evaluate,
    get_device,
    load_dataset_splits,
    make_dataloader,
    select_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained classifier.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", default=None, help="Override dataset from checkpoint.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="Split to evaluate.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--csv", type=Path, default=None, help="Append metrics to this CSV file.")
    return parser.parse_args()


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model_name = checkpoint["model_name"]
    dataset_name = args.dataset or checkpoint["dataset_name"]
    transform = build_transform(model_name)
    splits = load_dataset_splits(dataset_name, root=args.data_root, transform=transform)
    dataset = select_split(splits, args.split)

    dataloader = make_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    model = build_classifier(
        model_name,
        num_classes=checkpoint["num_classes"],
        use_default_weights=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate(model, dataloader, nn.CrossEntropyLoss(), device)
    row = {
        "checkpoint": str(args.checkpoint),
        "dataset": dataset_name,
        "model": model_name,
        "split": args.split,
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
        "examples": metrics["examples"],
        "pretrained": True,
        "num_classes": checkpoint["num_classes"],
    }

    print(
        f"dataset={dataset_name} "
        f"model={model_name} "
        f"split={args.split} "
        f"loss={metrics['loss']:.4f} "
        f"accuracy={metrics['accuracy']:.4f} "
        f"examples={metrics['examples']}"
    )
    if args.csv is not None:
        append_csv(args.csv, row)


if __name__ == "__main__":
    main()
