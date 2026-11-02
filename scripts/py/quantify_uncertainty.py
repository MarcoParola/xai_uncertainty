#!/usr/bin/env python3
"""Quantify uncertainty for a trained classifier checkpoint."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from classifier_utils import (
    DEFAULT_DATA_ROOT,
    build_classifier,
    build_transform,
    get_device,
    load_dataset_splits,
    make_dataloader,
    select_split,
)
from src.metrics.uncertainty import (
    brier_score,
    entropy,
    expected_calibration_error,
    nll,
)


AVAILABLE_METRICS = ("entropy", "brier", "nll", "ece")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantify classifier uncertainty.")
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
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Merge metrics into this checkpoint-level CSV file.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=AVAILABLE_METRICS,
        default=list(AVAILABLE_METRICS),
        help="Uncertainty metrics to compute.",
    )
    return parser.parse_args()


def merge_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    fieldnames = []
    if path.exists() and path.stat().st_size > 0:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)

    for name in row:
        if name not in fieldnames:
            fieldnames.append(name)

    identity = (Path(row["checkpoint"]).resolve(), str(row["split"]))
    for index, existing in enumerate(rows):
        existing_identity = (
            Path(existing.get("checkpoint", "")).resolve(),
            existing.get("split"),
        )
        if existing_identity != identity:
            continue
        merged = {**existing, **row}
        old_metrics = existing.get("metrics", "").split(",")
        new_metrics = str(row.get("metrics", "")).split(",")
        merged["metrics"] = ",".join(
            dict.fromkeys(name for name in old_metrics + new_metrics if name)
        )
        rows[index] = merged
        break
    else:
        rows.append(row)

    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def empty_mean(values: list[float]) -> float | str:
    if not values:
        return ""
    return sum(values) / len(values)


def summarize_metric(name: str, values: torch.Tensor, correct: list[float], incorrect: list[float]):
    if values.numel() == 0:
        return {
            f"{name}_mean": 0.0,
            f"{name}_std": 0.0,
            f"{name}_min": 0.0,
            f"{name}_max": 0.0,
            f"correct_{name}_mean": "",
            f"incorrect_{name}_mean": "",
        }
    return {
        f"{name}_mean": values.mean().item(),
        f"{name}_std": values.std(unbiased=False).item(),
        f"{name}_min": values.min().item(),
        f"{name}_max": values.max().item(),
        f"correct_{name}_mean": empty_mean(correct),
        f"incorrect_{name}_mean": empty_mean(incorrect),
    }


@torch.no_grad()
def quantify(
    model,
    dataloader,
    device: torch.device,
    metric_names: list[str],
    ece_bins: int,
) -> dict[str, float | int | str]:
    model.eval()
    stored_metrics = [name for name in metric_names if name != "ece"]
    if "entropy" in metric_names:
        stored_metrics.append("normalized_entropy")
    values_by_metric = {name: [] for name in stored_metrics}
    correct_by_metric = {name: [] for name in stored_metrics}
    incorrect_by_metric = {name: [] for name in stored_metrics}
    ece_scores = []
    ece_targets = []
    total_correct = 0
    total_examples = 0

    for batch in dataloader:
        images, targets = batch[0].to(device), batch[1].to(device)
        logits = model(images)
        predictions = logits.argmax(dim=1)
        correct = predictions == targets

        batch_values = {}
        if "entropy" in metric_names:
            batch_values["entropy"] = entropy(logits, from_logits=True, normalize=False)
            batch_values["normalized_entropy"] = entropy(
                logits,
                from_logits=True,
                normalize=True,
            )
        if "brier" in metric_names:
            batch_values["brier"] = brier_score(logits, targets, from_logits=True)
        if "nll" in metric_names:
            batch_values["nll"] = nll(logits, targets, from_logits=True)
        if "ece" in metric_names:
            ece_scores.append(logits.cpu())
            ece_targets.append(targets.cpu())

        for name, values in batch_values.items():
            values_by_metric[name].append(values.cpu())
            correct_by_metric[name].extend(values[correct].cpu().tolist())
            incorrect_by_metric[name].extend(values[~correct].cpu().tolist())

        total_correct += correct.sum().item()
        total_examples += targets.size(0)

    metrics = {
        "examples": total_examples,
        "accuracy": total_correct / max(total_examples, 1),
    }
    for name in stored_metrics:
        values = (
            torch.cat(values_by_metric[name])
            if values_by_metric[name]
            else torch.empty(0)
        )
        metrics.update(
            summarize_metric(
                name,
                values,
                correct_by_metric[name],
                incorrect_by_metric[name],
            )
        )
    if "ece" in metric_names:
        all_scores = torch.cat(ece_scores) if ece_scores else torch.empty(0, 1)
        all_targets = (
            torch.cat(ece_targets)
            if ece_targets
            else torch.empty(0, dtype=torch.long)
        )
        metrics["ece"] = expected_calibration_error(
            all_scores,
            all_targets,
            n_bins=ece_bins,
            from_logits=True,
        ).item()
        metrics["ece_bins"] = ece_bins
    return metrics


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

    metrics = quantify(model, dataloader, device, args.metrics, args.ece_bins)
    row = {
        "checkpoint": str(args.checkpoint),
        "dataset": dataset_name,
        "model": model_name,
        "split": args.split,
        "metrics": ",".join(args.metrics),
        "pretrained": True,
        "num_classes": checkpoint["num_classes"],
        **metrics,
    }

    metric_summary = " ".join(
        f"ece={metrics['ece']:.4f}"
        if name == "ece"
        else f"{name}_mean={metrics[f'{name}_mean']:.4f}"
        for name in args.metrics
    )
    print(
        f"dataset={dataset_name} "
        f"model={model_name} "
        f"split={args.split} "
        f"{metric_summary} "
        f"accuracy={metrics['accuracy']:.4f} "
        f"examples={metrics['examples']}"
    )
    if args.csv is not None:
        merge_csv(args.csv, row)


if __name__ == "__main__":
    main()
