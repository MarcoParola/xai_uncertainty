#!/usr/bin/env python3
"""Generate and store saliency maps for a dataset test split."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "py"))

from classifier_utils import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    build_classifier,
    build_transform,
    get_device,
    load_dataset_splits,
    make_dataloader,
    select_split,
)
from config import load_training_config  # noqa: E402
from src.xai import GradCAM, GradCAMPlusPlus, GuidedGradCAM, LIME, RISE, XGradCAM  # noqa: E402


METHODS = ("gradcam", "guided_gradcam", "gradcampp", "xgradcam", "rise", "lime")
CONFIG = load_training_config()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate saliency maps for a checkpoint on a dataset test split."
    )
    parser.add_argument("dataset", help="Dataset name.")
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_DATA_ROOT / "saliencies")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--num-images",
        type=int,
        default=int(CONFIG["saliency_generation_limit"]),
        help="Number of random test images to explain. Use 0 or a negative value for all images.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(CONFIG["seed"]),
        help="Seed used to select the random test images.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--target-mode",
        choices=("true", "pred", "class"),
        default="true",
        help="Class used to generate saliency maps.",
    )
    parser.add_argument("--class-id", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)

    parser.add_argument("--rise-masks", type=int, default=4000)
    parser.add_argument("--rise-mask-size", type=int, default=7)
    parser.add_argument("--rise-p", type=float, default=0.5)
    parser.add_argument("--rise-batch-size", type=int, default=64)

    parser.add_argument("--lime-samples", type=int, default=1000)
    parser.add_argument("--lime-grid-size", type=int, default=7)
    parser.add_argument("--lime-p", type=float, default=0.5)
    parser.add_argument("--lime-kernel-width", type=float, default=0.25)
    parser.add_argument("--lime-batch-size", type=int, default=64)
    return parser.parse_args()


class IndexedSubset(Dataset):
    """Subset wrapper that returns original indices with each sample."""

    def __init__(self, dataset, indices: torch.Tensor) -> None:
        self.dataset = dataset
        self.indices = indices.tolist()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        original_index = self.indices[index]
        image, label = self.dataset[original_index]
        return image, label, original_index


def random_test_subset(dataset, num_images: int, seed: int) -> IndexedSubset:
    dataset_size = len(dataset)
    if num_images <= 0 or num_images >= dataset_size:
        indices = torch.arange(dataset_size)
    else:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(dataset_size, generator=generator)[:num_images]
    return IndexedSubset(dataset, indices)


def make_output_dir(args: argparse.Namespace, checkpoint: dict) -> Path:
    checkpoint_name = args.checkpoint.stem
    model_name = checkpoint["model_name"]
    output_dir = (
        args.output_root
        / args.dataset
        / checkpoint_name
        / model_name
        / args.method
        / args.target_mode
    )

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_explainer(args: argparse.Namespace, model):
    if args.method == "gradcam":
        return GradCAM(model)
    if args.method == "guided_gradcam":
        return GuidedGradCAM(model)
    if args.method == "gradcampp":
        return GradCAMPlusPlus(model)
    if args.method == "xgradcam":
        return XGradCAM(model)
    if args.method == "rise":
        return RISE(
            model,
            n_masks=args.rise_masks,
            mask_size=(args.rise_mask_size, args.rise_mask_size),
            p=args.rise_p,
            batch_size=args.rise_batch_size,
        )
    if args.method == "lime":
        return LIME(
            model,
            n_samples=args.lime_samples,
            grid_size=(args.lime_grid_size, args.lime_grid_size),
            p=args.lime_p,
            kernel_width=args.lime_kernel_width,
            batch_size=args.lime_batch_size,
        )
    raise ValueError(f"Unsupported saliency method: {args.method}")


@torch.no_grad()
def predicted_targets(model, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    logits = model(images.to(device))
    return logits.argmax(dim=1)


def resolve_targets(
    args: argparse.Namespace,
    model,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if args.target_mode == "true":
        return labels.to(device)

    if args.target_mode == "pred":
        return predicted_targets(model, images, device)

    if args.class_id is None:
        raise ValueError("--class-id is required when --target-mode class is used.")
    return torch.full((images.size(0),), args.class_id, dtype=torch.long, device=device)


def save_manifest(output_dir: Path, manifest: dict) -> None:
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model_name = checkpoint["model_name"]
    transform = build_transform(model_name, train=False)
    splits = load_dataset_splits(args.dataset, root=args.data_root, transform=transform)
    test_dataset = select_split(splits, "test")
    sampled_dataset = random_test_subset(test_dataset, args.num_images, args.seed)

    dataloader = make_dataloader(
        sampled_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    model = build_classifier(
        model_name,
        num_classes=checkpoint["num_classes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    explainer = build_explainer(args, model)
    output_dir = make_output_dir(args, checkpoint)
    batch_files = []
    n_examples = 0

    for batch_index, (images, labels, indices) in enumerate(dataloader):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break

        targets = resolve_targets(args, model, images, labels, device)
        maps = explainer.generate(images.to(device), targets=targets)

        batch_file = output_dir / f"batch_{batch_index:06d}.pt"
        start_index = n_examples
        end_index = start_index + images.size(0)
        torch.save(
            {
                "saliency_maps": maps.detach().cpu(),
                "labels": labels.detach().cpu(),
                "targets": targets.detach().cpu(),
                "indices": indices.detach().cpu(),
            },
            batch_file,
        )

        batch_files.append(batch_file.name)
        n_examples = end_index
        print(
            f"saved batch={batch_index} examples={start_index}:{end_index} "
            f"path={batch_file}"
        )

    manifest = {
        "dataset": args.dataset,
        "split": "test",
        "method": args.method,
        "target_mode": args.target_mode,
        "class_id": args.class_id,
        "checkpoint": str(args.checkpoint),
        "model_name": model_name,
        "pretrained": True,
        "num_classes": checkpoint["num_classes"],
        "class_names": checkpoint.get("class_names"),
        "n_examples": n_examples,
        "requested_num_images": args.num_images,
        "sample_seed": args.seed,
        "sample_indices": sampled_dataset.indices[:n_examples],
        "batch_size": args.batch_size,
        "batch_files": batch_files,
        "output_dir": str(output_dir),
    }
    save_manifest(output_dir, manifest)
    print(f"saved manifest={output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
