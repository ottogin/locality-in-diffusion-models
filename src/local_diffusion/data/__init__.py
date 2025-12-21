"""Dataset loading utilities for locality diffusion models."""

from .datasets import (  # noqa: F401
    DatasetBundle,
    DatasetFactoryOutput,
    build_dataset,
    register_dataset,
)

# Ensure default datasets are registered on import.
from . import torchvision_datasets as _torchvision_datasets  # noqa: F401
from . import image_folder_datasets as _image_folder_datasets  # noqa: F401

__all__ = [
    "DatasetBundle",
    "DatasetFactoryOutput",
    "build_dataset",
    "register_dataset",
]

