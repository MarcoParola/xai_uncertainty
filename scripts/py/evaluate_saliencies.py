#!/usr/bin/env python3
"""Evaluate saved saliency maps with XAI metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "py"))

from classifier_utils import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    build_classifier,
    build_transform,
    get_device,
    load_dataset_splits,
    select_split,
)
from src.metrics.xai import deletion, insertion, pointing_game  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved saliency maps.")
    parser.add_argument("saliency_dir", type=Path, help="Folder containing manifest.json.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--metric-batch-size", type=int, default=64)
    parser.add_argument("--pointing-tolerance", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def load_manifest(saliency_dir: Path) -> dict:
    manifest_path = saliency_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_classifier(
        checkpoint["model_name"],
        num_classes=checkpoint["num_classes"],
        use_default_weights=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def load_test_dataset(manifest: dict, checkpoint: dict, data_root: Path):
    transform = build_transform(checkpoint["model_name"], train=False)
    splits = load_dataset_splits(
        manifest["dataset"],
        root=data_root,
        transform=transform,
        return_bboxes=True,
    )
    return select_split(splits, "test")


def stack_images_and_bboxes(dataset, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    images = []
    bboxes = []
    for index in indices:
        image, _label, bbox = dataset[int(index)]
        images.append(image)
        bboxes.append(bbox)
    return torch.stack(images), torch.stack(bboxes)


def batch_metric_rows(
    batch_file: Path,
    batch: dict,
    model,
    dataset,
    args: argparse.Namespace,
) -> dict:
    indices = batch["indices"]
    images, bboxes = stack_images_and_bboxes(dataset, indices)
    saliency_maps = batch["saliency_maps"]
    targets = batch["targets"]

    insertion_out = insertion(
        model,
        images,
        saliency_maps,
        targets=targets,
        steps=args.steps,
        batch_size=args.metric_batch_size,
    )
    deletion_out = deletion(
        model,
        images,
        saliency_maps,
        targets=targets,
        steps=args.steps,
        batch_size=args.metric_batch_size,
    )

    result = {
        "batch_file": batch_file.name,
        "n_examples": int(indices.numel()),
        "insertion_auc": float(insertion_out["auc"].mean()),
        "deletion_auc": float(deletion_out["auc"].mean()),
        "pointing_game": "",
    }

    result["pointing_game"] = float(
        pointing_game(
            saliency_maps,
            bboxes,
            tolerance=args.pointing_tolerance,
        )["score"]
    )

    return result


def weighted_mean(rows: list[dict], key: str):
    values = [(row[key], row["n_examples"]) for row in rows if row[key] != ""]
    if not values:
        return ""
    total = sum(weight for _value, weight in values)
    return sum(float(value) * weight for value, weight in values) / max(total, 1)


def write_csv(path: Path, rows: list[dict], summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["batch_file", "n_examples"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({field: "" for field in fieldnames})
        writer.writerow({field: summary.get(field, "") for field in fieldnames})


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.saliency_dir)
    device = get_device(args.device)
    checkpoint_path = Path(manifest["checkpoint"])
    model, checkpoint = load_model(checkpoint_path, device)
    test_dataset = load_test_dataset(manifest, checkpoint, args.data_root)

    rows = []
    batch_files = manifest.get("batch_files") or sorted(
        path.name for path in args.saliency_dir.glob("batch_*.pt")
    )
    for batch_index, batch_name in enumerate(batch_files):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
        batch_file = args.saliency_dir / batch_name
        batch = torch.load(batch_file, map_location="cpu")
        row = batch_metric_rows(batch_file, batch, model, test_dataset, args)
        rows.append(row)
        print(
            f"evaluated {batch_name} "
            f"insertion_auc={row['insertion_auc']:.4f} "
            f"deletion_auc={row['deletion_auc']:.4f}"
        )

    summary = {
        "batch_file": "summary",
        "n_examples": sum(row["n_examples"] for row in rows),
        "insertion_auc": weighted_mean(rows, "insertion_auc"),
        "deletion_auc": weighted_mean(rows, "deletion_auc"),
        "pointing_game": weighted_mean(rows, "pointing_game"),
    }

    output = args.output or (args.saliency_dir / "metrics.csv")
    write_csv(output, rows, summary)

    print(f"localization_metrics=bbox dataset={manifest['dataset']}")
    print(f"summary={summary}")
    print(f"saved_metrics={output}")


if __name__ == "__main__":
    main()
