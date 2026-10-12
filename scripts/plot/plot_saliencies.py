#!/usr/bin/env python3
"""Save qualitative saliency grids: XAI methods by checkpoint, one per image.

Usage: python scripts/plot/plot_saliencies.py resnet50 oxford_pet --limit 10
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "plot_saliencies.json"
EXTENSIONS = {".npy", ".npz", ".pt", ".pth", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
VARIANT_PATTERN = re.compile(r"^(.*)_(best|minus_\d+|plus_\d+)$")
METHOD_LABELS = {
    "gradcam": "GradCAM", "gradcampp": "GradCAM++", "lime": "LIME", "rise": "RISE",
    "guided_gradcam": "Guided GradCAM", "xgradcam": "XGradCAM",
}
DEFAULT_METHODS = ["gradcam", "gradcampp", "lime", "rise"]


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path, dataset: str, require_images: bool = True) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    images = config.get("images", {}).get(dataset, [])
    if not isinstance(images, list) or (require_images and not images):
        raise ValueError(
            f"Set images.{dataset} in {path} to a nonempty list of test-image indices "
            "or individual saliency filenames. Run with --list-images to see available indices."
        )
    if any(
        not ((type(item) is int and item >= 0) or (isinstance(item, str) and item.strip()))
        for item in images
    ):
        raise ValueError("Every configured image must be a nonnegative integer or a nonempty string.")
    if len(set(images)) != len(images):
        raise ValueError("The image subset contains duplicate entries.")
    offsets = config.get("checkpoint_offsets")
    if offsets is not None and (
        not isinstance(offsets, list)
        or not offsets
        or any(type(offset) is not int or offset <= 0 for offset in offsets)
    ):
        raise ValueError("checkpoint_offsets must be null or a nonempty list of positive integers.")
    alpha = config.get("overlay_alpha", 0.7)
    if type(alpha) not in {int, float} or not 0 <= alpha <= 1:
        raise ValueError("overlay_alpha must be a number between 0 and 1.")
    methods = config.get("methods", DEFAULT_METHODS)
    if (not isinstance(methods, list) or not methods
            or any(not isinstance(method, str) or not method.strip() for method in methods)
            or len(set(methods)) != len(methods)):
        raise ValueError("methods must be a nonempty list of distinct method names.")
    return config


def discover_checkpoints(config: dict, model: str, method: str, dataset: str, require_best=True):
    root = project_path(config["saliency_root"])
    dataset_root = root / dataset
    if not dataset_root.is_dir():
        raise ValueError(f"Saliency dataset directory does not exist: {dataset_root}")

    groups = {}
    for checkpoint in sorted(dataset_root.iterdir()):
        match = VARIANT_PATTERN.fullmatch(checkpoint.name)
        if not checkpoint.is_dir():
            continue
        # The trainer's unsuffixed checkpoint also contains the best model.
        stem, label = match.groups() if match else (checkpoint.name, "base")
        if config.get("checkpoint_stem") and stem != config["checkpoint_stem"]:
            continue
        seed_match = re.search(r"(?:^|_)seed_(\d+)(?:_|$)", stem)
        if seed_match and int(seed_match.group(1)) != config.get("seed", 0):
            continue
        relative = config["directory_template"].format(
            dataset=dataset, checkpoint=checkpoint.name, model=model,
            method=method, target=config.get("target", "true"),
        )
        directory = root / relative
        if directory.is_dir():
            if not (directory / "manifest.json").is_file() and any(directory.glob("batch_*.pt")):
                continue
            if not require_best and not (directory / "manifest.json").is_file() and not any(
                path.is_file() and path.suffix.lower() in EXTENSIONS for path in directory.rglob("*")
            ):
                continue
            groups.setdefault(stem, {})[label] = directory

    groups = {
        stem: variants for stem, variants in groups.items()
        if not require_best or "best" in variants or "base" in variants
    }
    if not groups:
        raise ValueError(
            f"No best-checkpoint saliencies found for {dataset}/{model}/{method} "
            f"(seed {config.get('seed', 0)}). Check saliency_root and directory_template; "
            "batched generation must finish writing manifest.json."
        )
    if len(groups) != 1:
        raise ValueError(
            "Multiple experiments match. Set checkpoint_stem in the config to one of: "
            + ", ".join(groups)
        )
    stem, variants = next(iter(groups.items()))
    by_offset = {0: variants.get("best", variants.get("base"))}
    for label, directory in variants.items():
        if label not in {"best", "base"}:
            side, offset = label.split("_")
            by_offset[int(offset) * (-1 if side == "minus" else 1)] = directory
    offsets = config.get("checkpoint_offsets")
    if offsets is None:
        known_offsets = {abs(offset) for offset in by_offset if offset}
        checkpoint_root = project_path(config.get("checkpoint_root", "checkpoints"))
        for checkpoint in checkpoint_root.glob(f"{stem}_*.pt"):
            match = VARIANT_PATTERN.fullmatch(checkpoint.stem)
            if match and match.group(1) == stem and match.group(2) != "best":
                known_offsets.add(int(match.group(2).split("_")[1]))
        offsets = sorted(known_offsets)
    else:
        offsets = sorted(set(offsets))
    # Configured offsets retain empty slots while surrounding maps are not yet generated.
    columns = [-offset for offset in reversed(offsets)] + [0] + offsets
    return stem, [(offset, by_offset.get(offset)) for offset in columns]


def discover_grid(config: dict, model: str, dataset: str):
    methods = config.get("methods", DEFAULT_METHODS)
    discovered = {}
    for method in methods:
        try:
            discovered[method] = discover_checkpoints(config, model, method, dataset, require_best=False)
        except ValueError as error:
            if not str(error).startswith("No best-checkpoint"):
                raise
    if not discovered:
        raise ValueError(f"No completed saliencies found for {dataset}/{model} (seed {config.get('seed', 0)}).")
    stems = {stem for stem, _ in discovered.values()}
    if len(stems) != 1:
        raise ValueError("Methods refer to different experiments. Set checkpoint_stem in the config.")
    offsets = sorted({offset for _, columns in discovered.values() for offset, _ in columns})
    directories = {
        method: [dict(discovered[method][1]).get(offset) for offset in offsets]
        if method in discovered else [None] * len(offsets)
        for method in methods
    }
    if not any(row[offsets.index(0)] is not None for row in directories.values()):
        raise ValueError("No completed best-checkpoint saliencies found for the selected methods.")
    return stems.pop(), offsets, directories


def shared_image_selectors(directories: dict) -> list:
    shared = None
    for row in directories.values():
        for directory in row:
            if directory is None:
                continue
            manifest_path = directory / "manifest.json"
            if manifest_path.is_file():
                with manifest_path.open(encoding="utf-8") as handle:
                    indices = json.load(handle)["sample_indices"]
                if any(type(index) is not int or index < 0 for index in indices):
                    raise ValueError(f"Invalid sample indices in {manifest_path}")
                if len(set(indices)) != len(indices):
                    raise ValueError(f"Duplicate sample indices in {manifest_path}")
                available = set(indices)
            else:
                available = {
                    str(path.relative_to(directory)) for path in directory.rglob("*")
                    if path.is_file() and path.suffix.lower() in EXTENSIONS
                }
            shared = available if shared is None else shared & available
    if not shared:
        raise ValueError("No images are shared by all completed method/checkpoint outputs.")
    return sorted(shared, key=lambda item: (type(item).__name__, item))


def sample_images(directories: dict, limit: int, random_seed=None) -> list:
    if limit <= 0:
        raise ValueError("--limit must be a positive integer.")
    candidates = shared_image_selectors(directories)
    if limit > len(candidates):
        print(f"Warning: only {len(candidates)} shared images are available; plotting all of them.", file=sys.stderr)
    positions = np.random.default_rng(random_seed).choice(len(candidates), min(limit, len(candidates)), replace=False)
    return [candidates[int(position)] for position in positions]


def find_saliency(directory: Path, selector: str) -> Path:
    requested = Path(selector)
    if requested.is_absolute() or ".." in requested.parts:
        raise ValueError(f"Image selectors must be relative paths or IDs: {selector}")
    exact = directory / requested
    if exact.is_file() and exact.suffix.lower() in EXTENSIONS:
        return exact
    files = [
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    ]
    # Image filenames (e.g. cat.jpg) also match maps saved as cat.npy or cat.pt.
    if len(requested.parts) > 1:
        matches = [
            path for path in files
            if path.relative_to(directory).with_suffix("") == requested.with_suffix("")
        ]
    else:
        matches = [path for path in files if path.stem == requested.stem]
    if not matches:
        raise ValueError(f"No saliency for '{selector}' in {directory}")
    if len(matches) != 1:
        raise ValueError(
            f"Ambiguous image '{selector}' in {directory}. Use an exact relative filename: "
            + ", ".join(str(path.relative_to(directory)) for path in sorted(matches))
        )
    return matches[0]


def load_saliency(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            key = next((key for key in ("saliency", "saliency_map", "heatmap") if key in archive), None)
            if key is None:
                if len(archive.files) != 1:
                    raise ValueError(f"Expected one map or a 'saliency' key in {path}")
                key = archive.files[0]
            array = archive[key]
    elif suffix in {".pt", ".pth"}:
        import torch

        value = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(value, dict):
            key = next((key for key in ("saliency", "saliency_map", "heatmap") if key in value), None)
            if key is None:
                raise ValueError(f"Expected a tensor or a 'saliency' key in {path}")
            value = value[key]
        array = value.detach().cpu().float().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    else:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB") if image.mode not in {"L", "I", "F"} else image)

    return prepare_saliency(array, path)


def prepare_saliency(array: np.ndarray, source: Path) -> np.ndarray:
    array = np.squeeze(array)
    if array.ndim == 3 and array.shape[0] in {3, 4} and array.shape[-1] not in {3, 4}:
        array = np.moveaxis(array, 0, -1)
    if array.ndim != 2 and not (array.ndim == 3 and array.shape[-1] in {3, 4}):
        raise ValueError(f"Expected a single 2D map or RGB image in {source}; got {array.shape}")
    array = array.astype(np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"Saliency contains NaN or infinite values: {source}")
    if array.ndim == 3:
        if array.min() < 0 or array.max() > 255:
            raise ValueError(f"RGB values must be in [0, 1] or [0, 255]: {source}")
        if array.max() > 1:
            array /= 255
    return array


def load_selected_saliencies(directory: Path, selectors: list) -> list[np.ndarray]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        if any(not isinstance(selector, str) for selector in selectors):
            raise ValueError(f"Integer image indices require a generated manifest.json in {directory}")
        return [load_saliency(find_saliency(directory, selector)) for selector in selectors]

    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    indices = []
    for selector in selectors:
        if type(selector) is int:
            indices.append(selector)
        elif isinstance(selector, str) and selector.isdecimal():
            indices.append(int(selector))
        else:
            raise ValueError(
                f"Batched saliencies in {directory} require test-image indices, "
                f"not '{selector}'. Run with --list-images to see available indices."
            )
    if len(set(indices)) != len(indices):
        raise ValueError("The image subset contains duplicate test-image indices.")

    sample_indices = manifest["sample_indices"]
    positions = {index: position for position, index in enumerate(sample_indices)}
    if len(positions) != len(sample_indices):
        raise ValueError(f"Duplicate sample indices in {manifest_path}")
    missing = sorted(set(indices) - positions.keys())
    if missing:
        raise ValueError(f"Test-image indices {missing} were not generated in {directory}")
    batch_size = manifest["batch_size"]
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError(f"Invalid batch size in {manifest_path}")
    batch_files = manifest["batch_files"]
    batch_numbers = sorted({positions[index] // batch_size for index in indices})
    if any(number >= len(batch_files) for number in batch_numbers):
        raise ValueError(f"Sample indices do not match batch files in {manifest_path}")

    import torch

    selected = {}
    wanted = set(indices)
    for number in batch_numbers:
        batch_path = directory / batch_files[number]
        batch = torch.load(batch_path, map_location="cpu", weights_only=True)
        maps = batch["saliency_maps"]
        actual_indices = batch["indices"].tolist()
        if len(actual_indices) != len(maps):
            raise ValueError(f"Map count does not match image indices in {batch_path}")
        expected_indices = sample_indices[number * batch_size:(number + 1) * batch_size]
        if actual_indices != expected_indices:
            raise ValueError(f"Batch indices do not match {manifest_path}: {batch_path}")
        for position, index in enumerate(actual_indices):
            if index in wanted:
                array = maps[position].detach().cpu().float().numpy()
                selected[index] = prepare_saliency(array, batch_path)
    return [selected[index] for index in indices]


def list_available_images(config: dict, model: str, method: str, dataset: str) -> None:
    _, columns = discover_checkpoints(config, model, method, dataset)
    best_directory = next(directory for offset, directory in columns if offset == 0)
    manifest_path = best_directory / "manifest.json"
    print(f"Best saliency directory: {best_directory}")
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        indices = manifest["sample_indices"]
        print(f"Available test-image indices ({len(indices)}):")
        print(json.dumps(indices))
    else:
        print("Available individual saliency files:")
        for path in sorted(best_directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                print(path.relative_to(best_directory))


def load_source_images(config: dict, model: str, dataset: str, selectors: list) -> list[np.ndarray]:
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "py"))
    from classifier_utils import build_transform, load_dataset_splits, select_split

    transform = build_transform(model, train=False)
    normalization = transform.transforms[-1]
    mean = np.asarray(normalization.mean, dtype=np.float32)
    std = np.asarray(normalization.std, dtype=np.float32)
    test_dataset = None
    images = []
    for selector in selectors:
        if type(selector) is int or (isinstance(selector, str) and selector.isdecimal()):
            if test_dataset is None:
                splits = load_dataset_splits(
                    dataset, root=project_path(config.get("data_root", "data")), transform=transform,
                )
                test_dataset = select_split(splits, "test")
            index = int(selector)
            if not 0 <= index < len(test_dataset):
                raise ValueError(f"Test-image index {index} is out of range for {dataset}")
            tensor, _ = test_dataset[index]
        else:
            path = config.get("image_paths", {}).get(dataset, {}).get(str(selector))
            if path is None:
                raise ValueError(
                    f"Set image_paths.{dataset}['{selector}'] to the original image path "
                    "when using individual saliency filenames."
                )
            with Image.open(project_path(path)) as image:
                tensor = transform(image.convert("RGB"))
        # Undo normalization while retaining the generator's exact resize and crop.
        array = np.moveaxis(tensor.detach().cpu().numpy(), 0, -1)
        images.append(np.clip(array * std + mean, 0, 1))
    return images


def render_image_grid(config: dict, columns: list, maps: dict, image: np.ndarray,
                      title: str, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Rectangle

    palette = config.get("cmap", "inferno")
    colormap = (
        LinearSegmentedColormap.from_list("saliency", palette, N=257)
        if isinstance(palette, list) else matplotlib.colormaps[palette]
    )
    overlay_alpha = config.get("overlay_alpha", 0.7)

    offsets = [offset for offset, _ in columns]
    if offsets != sorted(set(offsets)) or 0 not in offsets or offsets.index(0) != len(offsets) // 2:
        raise ValueError("Checkpoint columns must be distinct, ordered, and centered on best.")
    for method, row in maps.items():
        if len(row) != len(columns):
            raise ValueError(f"Map count for {method} does not match checkpoint columns.")
        for array in row:
            if array is not None and array.shape[:2] != image.shape[:2]:
                raise ValueError(
                    f"Saliency shape {array.shape[:2]} does not match image shape {image.shape[:2]}. "
                    "Use the same image preprocessing configuration as saliency generation."
                )
    width = max(10.5, 1.3 + 1.85 * len(columns))
    height = 1.4 + 1.85 * len(maps)
    figure, axes = plt.subplots(
        len(maps), len(columns), squeeze=False, figsize=(width, height), facecolor="white",
    )
    try:
        figure.subplots_adjust(left=1.3 / width, right=0.99, bottom=0.035,
                               top=1 - 1.4 / height, wspace=0.035, hspace=0.055)
        for row_index, (method, row) in enumerate(maps.items()):
            for column_index, ((offset, _), array) in enumerate(zip(columns, row)):
                axis = axes[row_index, column_index]
                axis.set_axis_off()
                axis.imshow(image, interpolation="bilinear")
                if array is None:
                    axis.text(
                        0.5, 0.08, "Not generated", transform=axis.transAxes,
                        ha="center", va="center", fontsize=8, color="#555555",
                        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none", "pad": 3},
                    )
                elif array.ndim == 2:
                    axis.imshow(
                        array, cmap=colormap, vmin=0, vmax=1, interpolation="bilinear",
                        alpha=overlay_alpha,
                    )
                else:
                    axis.imshow(array, alpha=overlay_alpha)
                if row_index == 0:
                    axis.set_title("Best" if offset == 0 else f"Best {offset:+d}",
                                   fontsize=10, fontweight="bold" if offset == 0 else "normal", pad=9)
                if offset == 0:
                    axis.add_patch(Rectangle((0, 0), 1, 1, transform=axis.transAxes,
                                             fill=False, edgecolor="#197451", linewidth=1.8, clip_on=False))
            axes[row_index, 0].text(
                -0.08, 0.5, METHOD_LABELS.get(method, method), transform=axes[row_index, 0].transAxes,
                ha="right", va="center", fontsize=10, fontweight="bold", color="#333333",
            )

        original = figure.add_axes([0.15 / width, 1 - 1.2 / height, 0.85 / width, 0.85 / height])
        original.imshow(image, interpolation="bilinear")
        original.set_axis_off()
        original.set_title("Original", fontsize=8, pad=3)
        heading, _, subtitle = title.partition("\n")
        figure.text(1.25 / width, 1 - 0.3 / height, heading, fontsize=13, va="top", color="#222222")
        figure.text(1.25 / width, 1 - 0.65 / height, subtitle, fontsize=9, va="top", color="#666666")
        color_axis = figure.add_axes([1 - 3.3 / width, 1 - 0.7 / height, 2.7 / width, 0.12 / height])
        colorbar = figure.colorbar(
            ScalarMappable(norm=Normalize(0, 1), cmap=colormap), cax=color_axis,
            orientation="horizontal", ticks=[0, 0.5, 1],
        )
        colorbar.set_label("Saliency (shared scale)", fontsize=8, labelpad=2)
        colorbar.ax.tick_params(labelsize=8, length=2)
        colorbar.outline.set_visible(False)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=config.get("dpi", 200), bbox_inches="tight", facecolor="white")
    finally:
        plt.close(figure)


def plot_saliencies(config: dict, model: str, dataset: str, output: Path | None = None,
                   limit: int = 10, random_seed=None, selectors: list | None = None):
    stem, offsets, directories = discover_grid(config, model, dataset)
    if selectors is None:
        selectors = sample_images(directories, limit, random_seed)
    if not selectors or len(set(selectors)) != len(selectors):
        raise ValueError("Select a nonempty list of distinct images.")
    if output is not None and len(selectors) != 1:
        raise ValueError("--output names one figure; select one image with --image or --limit 1.")
    # Only read the batches containing the selected indices; RISE batch sizes differ.
    grid_maps = {
        method: [load_selected_saliencies(directory, selectors) if directory else [None] * len(selectors)
                 for directory in row]
        for method, row in directories.items()
    }
    images = load_source_images(config, model, dataset, selectors)
    missing_count = sum(directory is None for row in directories.values() for directory in row)
    if missing_count:
        print(
            f"Warning: {missing_count} method/checkpoint outputs are not generated; "
            "those panels show the source image with a 'Not generated' label.",
            file=sys.stderr,
        )

    outputs = []
    for image_index, (selector, image) in enumerate(zip(selectors, images)):
        maps = {method: [column[image_index] for column in row] for method, row in grid_maps.items()}
        destination = output
        if destination is None:
            identifier = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(selector))
            destination = project_path(config["output_dir"]) / (
                f"{stem}_qualitative_{config.get('target', 'true')}_image_{identifier}.png"
            )
        title = f"{dataset} / {model} / image {selector}\n{stem}"
        render_image_grid(config, [(offset, None) for offset in offsets], maps, image, title, destination)
        outputs.append(destination)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Model architecture, e.g. resnet50")
    parser.add_argument("dataset", help="Dataset name, e.g. oxford_pet")
    parser.add_argument("--limit", type=int, default=10, help="Number of randomly selected images (default: 10)")
    parser.add_argument("--random-seed", type=int, help="Reproduce the random image selection")
    parser.add_argument("--seed", type=int, help="Model checkpoint seed, overriding the config")
    parser.add_argument("--methods", nargs="+", help="XAI method rows, overriding the config")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, help="Output figure path (PNG, PDF, etc.)")
    parser.add_argument("--list-images", action="store_true", help="List saved test-image indices or individual files")
    parser.add_argument("--image", type=int, nargs="+", help="Select explicit test-image indices instead of random images")
    parser.add_argument("--use-configured-images", action="store_true", help="Use the configured image subset instead of sampling")
    args = parser.parse_args()
    try:
        config = load_config(
            args.config, args.dataset, require_images=args.use_configured_images,
        )
        if args.limit <= 0:
            raise ValueError("--limit must be a positive integer.")
        if args.seed is not None:
            if args.seed < 0:
                raise ValueError("--seed must be nonnegative.")
            config["seed"] = args.seed
        if args.methods:
            if len(set(args.methods)) != len(args.methods):
                raise ValueError("--methods must not contain duplicates.")
            config["methods"] = args.methods
        selectors = config["images"][args.dataset] if args.use_configured_images else None
        if args.image is not None:
            if any(index < 0 for index in args.image):
                raise ValueError("--image must contain nonnegative test-image indices.")
            selectors = args.image
        if args.list_images:
            _, _, directories = discover_grid(config, args.model, args.dataset)
            indices = shared_image_selectors(directories)
            print(f"Available shared images ({len(indices)}):")
            print(json.dumps(indices))
            return
        outputs = plot_saliencies(config, args.model, args.dataset, args.output,
                                 args.limit, args.random_seed, selectors)
    except (ValueError, OSError, KeyError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")
    for output in outputs:
        print(f"Saved saliency comparison to {output}")


if __name__ == "__main__":
    main()
