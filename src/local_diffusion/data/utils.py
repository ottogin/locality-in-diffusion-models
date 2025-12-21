"""Utility helpers for dataset handling."""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


LOGGER = logging.getLogger(__name__)


def identity(tensor: torch.Tensor) -> torch.Tensor:
    return tensor


def compose_transform(
    resolution: int,
    *,
    in_channels: int,
) -> transforms.Compose:
    ops: List = []

    if in_channels == 1:
        ops.append(transforms.Grayscale(num_output_channels=1))

    ops.extend(
        [
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
        ]
    )

    # Always normalize to [-1, 1] range
    mean = (0.5,) * in_channels
    std = (0.5,) * in_channels
    ops.append(transforms.Normalize(mean, std))

    return transforms.Compose(ops)


def get_postprocess_fn() -> Callable[[torch.Tensor], torch.Tensor]:
    """Return postprocess function to convert [-1, 1] normalized tensors to [0, 1] for visualization."""
    def postprocess(tensor: torch.Tensor) -> torch.Tensor:
        return ((tensor + 1.0) / 2.0).clamp(0, 1)
    return postprocess


def maybe_apply_subset(dataset: Dataset, subset_size: Optional[int]) -> Dataset:
    if subset_size is None:
        return dataset

    total = len(dataset)
    if subset_size >= total:
        LOGGER.warning(
            "Requested subset_size %s exceeds dataset length %s. Using full dataset.",
            subset_size,
            total,
        )
        return dataset

    indices = torch.arange(subset_size)
    return Subset(dataset, indices.tolist())

