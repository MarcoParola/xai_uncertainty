"""Factory for torchvision models used in training and finetuning."""

from __future__ import annotations


MODEL_ALIASES = {
    "resnet": "resnet50",
    "resnet18": "resnet18",
    "resnet34": "resnet34",
    "resnet50": "resnet50",
    "resnet101": "resnet101",
    "resnet152": "resnet152",
    "vit": "vit_b_16",
    "vit_b_16": "vit_b_16",
    "vit-b-16": "vit_b_16",
    "vit_b_32": "vit_b_32",
    "vit-b-32": "vit_b_32",
    "vit_l_16": "vit_l_16",
    "vit-l-16": "vit_l_16",
    "vit_l_32": "vit_l_32",
    "vit-l-32": "vit_l_32",
    "vit_h_14": "vit_h_14",
    "vit-h-14": "vit_h_14",
}


def normalize_model_name(name: str) -> str:
    normalized = name.lower().replace(" ", "_")
    try:
        return MODEL_ALIASES[normalized]
    except KeyError as exc:
        options = ", ".join(sorted({"resnet", "resnet50", "vit", "vit_b_16"}))
        raise ValueError(f"Unknown model '{name}'. Common choices: {options}.") from exc


def get_model(name: str):
    """Return a torchvision ResNet or ViT with default pretrained weights."""
    from torchvision import models

    model_name = normalize_model_name(name)
    weights = models.get_model_weights(model_name).DEFAULT
    return models.get_model(model_name, weights=weights)
