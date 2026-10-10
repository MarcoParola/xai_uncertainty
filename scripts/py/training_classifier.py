#!/usr/bin/env python3
"""Train a torchvision classifier on a supported dataset."""

from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from classifier_utils import (
    DEFAULT_DATA_ROOT,
    build_classifier,
    build_transform,
    checkpoint_payload,
    evaluate,
    get_class_names,
    get_device,
    load_dataset_splits,
    make_dataloader,
    set_seed,
)
from src.config import DEFAULT_TRAINING_CONFIG_PATH, load_training_config


DEFAULT_OUTPUT = Path("checkpoints/classifier.pt")


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_TRAINING_CONFIG_PATH,
        help="Path to the training defaults JSON config.",
    )
    config_args, remaining_args = config_parser.parse_known_args()
    defaults = load_training_config(config_args.config)

    parser = argparse.ArgumentParser(description="Train a classifier.")
    parser.add_argument(
        "--config",
        type=Path,
        default=config_args.config,
        help="Path to the training defaults JSON config.",
    )
    parser.add_argument("--dataset", required=True, help="Dataset name.")
    parser.add_argument("--model", default="resnet50", help="Model name.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=int(defaults["epochs"]))
    parser.add_argument("--batch-size", type=int, default=int(defaults["batch_size"]))
    parser.add_argument("--lr", type=float, default=float(defaults["lr"]))
    parser.add_argument("--weight-decay", type=float, default=float(defaults["weight_decay"]))
    parser.add_argument("--optimizer", choices=("adam", "adamw"), default=defaults["optimizer"])
    parser.add_argument("--scheduler", choices=("none", "cosine"), default=defaults["scheduler"])
    parser.add_argument("--warmup-epochs", type=int, default=int(defaults["warmup_epochs"]))
    parser.add_argument("--min-lr", type=float, default=float(defaults["min_lr"]))
    parser.add_argument("--num-workers", type=int, default=int(defaults["num_workers"]))
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=int(defaults["seed"]))
    parser.add_argument(
        "--patience",
        type=int,
        default=int(defaults["patience"]),
        help="Stop after this many epochs without validation accuracy improvement.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=float(defaults["min_delta"]),
        help="Minimum validation accuracy improvement required to reset patience.",
    )
    parser.add_argument(
        "--checkpoint-offsets",
        type=int,
        nargs="+",
        default=[int(offset) for offset in defaults["checkpoint_offsets"]],
        help="Save checkpoint copies at best epoch plus/minus these offsets.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=float(defaults["val_fraction"]),
        help="Validation fraction for datasets that need deterministic splitting.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=float(defaults["test_fraction"]),
        help="Test fraction for datasets that need deterministic splitting.",
    )
    parser.add_argument(
        "--keep-all-epoch-checkpoints",
        action="store_true",
        help="Keep the temporary per-epoch checkpoint cache after training.",
    )
    return parser.parse_args(remaining_args)


@dataclass
class EarlyStoppingState:
    best_val_accuracy: float = -1.0
    best_epoch: int = 0
    epochs_without_improvement: int = 0

    def update(self, epoch: int, val_accuracy: float, min_delta: float) -> bool:
        improved = val_accuracy > self.best_val_accuracy + min_delta
        if improved:
            self.best_val_accuracy = val_accuracy
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        return improved

    def should_stop(self, patience: int) -> bool:
        return self.epochs_without_improvement >= patience


def create_unique_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        base_dir.mkdir(parents=True)
        return base_dir

    for index in range(1, 1000):
        candidate = base_dir.with_name(f"{base_dir.name}_{index}")
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate

    raise RuntimeError(f"Could not create a unique checkpoint cache near {base_dir}")


def epoch_checkpoint_path(epoch_dir: Path, epoch: int) -> Path:
    return epoch_dir / f"epoch_{epoch:04d}.pt"


def checkpoint_copy_path(output: Path, label: str) -> Path:
    return output.with_name(f"{output.stem}_{label}{output.suffix}")


def checkpoint_labels(best_epoch: int, final_epoch: int, offsets: list[int]):
    yield "best", best_epoch
    for offset in offsets:
        minus_epoch = best_epoch - offset
        plus_epoch = best_epoch + offset
        if minus_epoch >= 1:
            yield f"minus_{offset:02d}", minus_epoch
        if plus_epoch <= final_epoch:
            yield f"plus_{offset:02d}", plus_epoch


def make_epoch_metrics(epoch: int, train_metrics: dict, val_metrics: dict) -> dict:
    return {
        "epoch": epoch,
        "train": train_metrics,
        "val": val_metrics,
    }


def save_checkpoint(
    path: Path,
    model,
    args: argparse.Namespace,
    class_names: list[str],
    metrics: dict,
) -> None:
    torch.save(
        checkpoint_payload(
            model=model,
            model_name=args.model,
            dataset_name=args.dataset,
            class_names=class_names,
            metrics=metrics,
        ),
        path,
    )


def resolve_patience(requested_patience: int, offsets: list[int]) -> int:
    return max(requested_patience, max(offsets, default=0))


def copy_selected_checkpoints(
    output: Path,
    epoch_dir: Path,
    best_epoch: int,
    final_epoch: int,
    offsets: list[int],
) -> list[Path]:
    saved_checkpoints = [output]
    for label, epoch in checkpoint_labels(best_epoch, final_epoch, offsets):
        destination = checkpoint_copy_path(output, label)
        source = output if label == "best" else epoch_checkpoint_path(epoch_dir, epoch)
        if source.exists():
            shutil.copy2(source, destination)
            saved_checkpoints.append(destination)
    return saved_checkpoints


def evaluate_best_checkpoint(
    model,
    output: Path,
    test_loader,
    criterion,
    device: torch.device,
) -> tuple[dict, dict]:
    checkpoint = torch.load(output, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)
    checkpoint["metrics"]["test"] = test_metrics
    torch.save(checkpoint, output)
    return checkpoint, test_metrics


def build_optimizer(model, args: argparse.Namespace):
    optimizer_cls = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }[args.optimizer]
    return optimizer_cls(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


def build_scheduler(optimizer, args: argparse.Namespace):
    if args.scheduler == "none":
        return None

    def lr_lambda(epoch_index: int) -> float:
        current_epoch = epoch_index + 1
        if args.warmup_epochs > 0 and current_epoch <= args.warmup_epochs:
            return current_epoch / args.warmup_epochs

        decay_epochs = max(args.epochs - args.warmup_epochs, 1)
        decay_epoch = min(max(current_epoch - args.warmup_epochs, 0), decay_epochs)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * decay_epoch / decay_epochs))
        min_factor = args.min_lr / args.lr if args.lr > 0 else 0.0
        return min_factor + (1.0 - min_factor) * cosine_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def current_lr(optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


def train_one_epoch(model, dataloader, criterion, optimizer, device) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_examples += batch_size

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
        "examples": total_examples,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    train_transform = build_transform(args.model, train=True)
    eval_transform = build_transform(args.model, train=False)

    train_dataset, val_dataset, test_dataset = load_dataset_splits(
        args.dataset,
        root=args.data_root,
        train_transform=train_transform,
        eval_transform=eval_transform,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )
    class_names = get_class_names(train_dataset, val_dataset, test_dataset)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": device,
    }
    train_loader = make_dataloader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = make_dataloader(val_dataset, **loader_kwargs)
    test_loader = make_dataloader(test_dataset, **loader_kwargs)

    model = build_classifier(
        args.model,
        num_classes=len(class_names),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)

    early_stopping = EarlyStoppingState()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    epoch_dir = create_unique_dir(args.output.with_name(f"{args.output.stem}_epoch_cache"))
    max_offset = max(args.checkpoint_offsets, default=0)
    patience = resolve_patience(args.patience, args.checkpoint_offsets)
    final_epoch = 0

    if patience != args.patience:
        print(
            f"patience adjusted from {args.patience} to {patience} "
            f"to allow best+{max_offset} checkpoint selection"
        )

    for epoch in range(1, args.epochs + 1):
        final_epoch = epoch
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        val_metrics = evaluate(model, val_loader, criterion, device)

        print(
            f"epoch={epoch}/{args.epochs} "
            f"lr={current_lr(optimizer):.6g} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        epoch_metrics = make_epoch_metrics(epoch, train_metrics, val_metrics)
        epoch_metrics["lr"] = current_lr(optimizer)
        save_checkpoint(
            path=epoch_checkpoint_path(epoch_dir, epoch),
            model=model,
            args=args,
            class_names=class_names,
            metrics=epoch_metrics,
        )

        if early_stopping.update(epoch, val_metrics["accuracy"], args.min_delta):
            save_checkpoint(
                path=args.output,
                model=model,
                args=args,
                class_names=class_names,
                metrics=epoch_metrics,
            )

        if early_stopping.should_stop(patience):
            print(
                f"early_stopping epoch={epoch} "
                f"best_epoch={early_stopping.best_epoch} "
                f"patience={patience}"
            )
            break

        if scheduler is not None:
            scheduler.step()

    checkpoint, test_metrics = evaluate_best_checkpoint(
        model=model,
        output=args.output,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
    )
    saved_checkpoints = copy_selected_checkpoints(
        output=args.output,
        epoch_dir=epoch_dir,
        best_epoch=checkpoint["metrics"]["epoch"],
        final_epoch=final_epoch,
        offsets=args.checkpoint_offsets,
    )

    if args.keep_all_epoch_checkpoints:
        print(f"epoch_checkpoint_cache={epoch_dir}")
    else:
        shutil.rmtree(epoch_dir)

    print(
        f"best_epoch={checkpoint['metrics']['epoch']} "
        f"val_acc={checkpoint['metrics']['val']['accuracy']:.4f} "
        f"test_loss={test_metrics['loss']:.4f} "
        f"test_acc={test_metrics['accuracy']:.4f} "
        f"checkpoint={args.output}"
    )
    print("saved_checkpoints=" + ",".join(str(path) for path in saved_checkpoints))


if __name__ == "__main__":
    main()
