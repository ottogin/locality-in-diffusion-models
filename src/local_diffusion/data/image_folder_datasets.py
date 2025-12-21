"""Dataset registrations for image folder datasets (CelebA-HQ, AFHQ, etc.)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from torch.utils.data import Dataset
from torchvision import datasets
from PIL import Image

from local_diffusion.configuration import DatasetConfig

from . import utils
from .datasets import DatasetFactoryOutput, register_dataset


LOGGER = logging.getLogger(__name__)


class ImageFolderDataset(Dataset):
    """Generic dataset for loading images from a folder."""

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Get all image files
        self.image_files = sorted(
            [
                f
                for f in os.listdir(root_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
            ]
        )

        if not self.image_files:
            raise ValueError(f"No image files found in {root_dir}")

        LOGGER.info("Found %d images in %s", len(self.image_files), root_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, 0  # Return 0 as dummy label


@register_dataset("celeba_hq")
def build_celeba_hq(cfg: DatasetConfig) -> DatasetFactoryOutput:
    """Build CelebA-HQ dataset.
    
    Expected directory structure:
        data/datasets/celebahq-resized-256x256/versions/1/celeba_hq_256/<images>
    
    If not found, you can download from:
    https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256
    """
    # Use resolution from config or default to 256
    resolution = cfg.resolution or 256
    
    # Look for dataset in root directory
    celeba_relative_path = "celebahq-resized-256x256/versions/1/celeba_hq_256"
    celeba_path = Path(cfg.root) / celeba_relative_path
    
    if not celeba_path.exists():
        error_msg = (
            f"CelebA-HQ dataset not found at {celeba_path}.\n"
            f"Please download from: https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256\n"
            f"Extract to: {Path(cfg.root)}"
            f"Expected directory structure: {celeba_path}/<images>"
        )
        raise FileNotFoundError(error_msg)
    
    transform = utils.compose_transform(resolution, in_channels=3)
    dataset = ImageFolderDataset(root_dir=str(celeba_path), transform=transform)
    
    postprocess = utils.get_postprocess_fn()
    
    return DatasetFactoryOutput(
        dataset=dataset,
        resolution=resolution,
        in_channels=3,
        postprocess=postprocess,
    )


@register_dataset("afhq")
def build_afhq(cfg: DatasetConfig) -> DatasetFactoryOutput:
    """Build AFHQ dataset.
    
    Expected directory structure:
        <root>/afhq/<split>/<class>/<images>
    
    Download from: https://github.com/clovaai/stargan-v2#animal-faces-hq-dataset-afhq
    """
    # Use resolution from config or default to 512
    resolution = cfg.resolution or 512
    
    # Look for dataset in root directory
    afhq_path = Path(cfg.root) / "afhq" / cfg.split
    
    if not afhq_path.exists():
        error_msg = (
            f"AFHQ dataset not found at {afhq_path}.\n"
            f"Please download from: https://github.com/clovaai/stargan-v2#animal-faces-hq-dataset-afhq\n"
            f"Extract to: {Path(cfg.root) / 'afhq'}"
        )
        raise FileNotFoundError(error_msg)
    
    transform = utils.compose_transform(resolution, in_channels=3)
    
    # AFHQ is organized as ImageFolder (with class subdirectories)
    dataset = datasets.ImageFolder(root=str(afhq_path), transform=transform)
    
    postprocess = utils.get_postprocess_fn()
    
    return DatasetFactoryOutput(
        dataset=dataset,
        resolution=resolution,
        in_channels=3,
        postprocess=postprocess,
    )

