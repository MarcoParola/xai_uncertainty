"""Dispatcher for supported dataset splits."""

from __future__ import annotations

from . import caltech101, fgvc_aircraft, oxford_pet, stanford_cars


DATASET_LOADERS = {
    "oxford_pet": oxford_pet.get_dataset,
    "oxford_pets": oxford_pet.get_dataset,
    "oxford-pet": oxford_pet.get_dataset,
    "oxford-pets": oxford_pet.get_dataset,
    "oxfordiiitpet": oxford_pet.get_dataset,
    "pet": oxford_pet.get_dataset,
    "pets": oxford_pet.get_dataset,
    "caltech101": caltech101.get_dataset,
    "caltech-101": caltech101.get_dataset,
    "calthech101": caltech101.get_dataset,
    "fgvc_aircraft": fgvc_aircraft.get_dataset,
    "fgvc-aircraft": fgvc_aircraft.get_dataset,
    "aircraft": fgvc_aircraft.get_dataset,
    "stanford_cars": stanford_cars.get_dataset,
    "stanford-cars": stanford_cars.get_dataset,
    "cars": stanford_cars.get_dataset,
}


def get_dataset(dataset_name: str, **kwargs):
    """Return train, validation, and test splits for a supported dataset."""
    normalized_name = dataset_name.lower().replace(" ", "_")
    try:
        loader = DATASET_LOADERS[normalized_name]
    except KeyError as exc:
        options = ", ".join(
            ["oxford_pet", "caltech101", "fgvc_aircraft", "stanford_cars"]
        )
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Choose one of: {options}."
        ) from exc

    return loader(**kwargs)
