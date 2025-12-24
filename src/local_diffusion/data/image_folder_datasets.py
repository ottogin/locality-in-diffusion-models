"""Dataset registrations for image folder datasets (CelebA-HQ, AFHQ, etc.)."""

from __future__ import annotations

import logging
import os
import subprocess
import zipfile
from pathlib import Path

from torch.utils.data import Dataset
from torchvision import datasets
from PIL import Image

from local_diffusion.configuration import DatasetConfig

from . import utils
from .datasets import DatasetFactoryOutput, register_dataset


LOGGER = logging.getLogger(__name__)


def _download_celeba_hq(target_dir: Path) -> None:
    """Download CelebA-HQ dataset from Kaggle using curl."""
    LOGGER.info("Downloading CelebA-HQ dataset from Kaggle...")

    # Create parent directory
    parent_dir = target_dir.parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    zip_path = parent_dir / "celebahq-resized-256x256.zip"
    url = "https://www.kaggle.com/api/v1/datasets/download/badasstechie/celebahq-resized-256x256"

    # Download using curl
    LOGGER.info("Downloading from %s", url)
    subprocess.run(
        ["curl", "-L", "-o", str(zip_path), url],
        check=True,
    )

    # Extract zip file
    LOGGER.info("Extracting to %s", parent_dir)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(parent_dir)

    # Clean up zip file
    zip_path.unlink()
    LOGGER.info("CelebA-HQ dataset ready at %s", target_dir)


def _download_afhq(target_dir: Path) -> None:
    """Download AFHQ dataset from Dropbox using wget."""
    LOGGER.info("Downloading AFHQ dataset from Dropbox...")

    target_dir.mkdir(parents=True, exist_ok=True)

    zip_path = target_dir.parent / "afhq.zip"
    url = "https://www.dropbox.com/s/vkzjokiwof5h8w6/afhq_v2.zip?dl=0"

    LOGGER.info("Downloading from %s", url)
    subprocess.run(
        ["curl", "-L", url, "-o", str(zip_path)],
        check=True,
    )

    LOGGER.info("Extracting to %s", target_dir)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)

    zip_path.unlink()
    LOGGER.info("AFHQ dataset ready at %s", target_dir)


# NOTE: Allows to load flat-structured directories unlike class-based like datasets.ImageFolder
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

    Set download=True in config to auto-download from Kaggle.
    """
    resolution = cfg.resolution or 256

    celeba_relative_path = "celebahq-resized-256x256/versions/1/celeba_hq_256"
    celeba_path = Path(cfg.root) / celeba_relative_path

    if not celeba_path.exists():
        if cfg.download:
            _download_celeba_hq(celeba_path)
        else:
            raise FileNotFoundError(
                f"CelebA-HQ dataset not found at {celeba_path}. "
                f"Set download=True in config or download manually from: "
                f"https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256"
            )

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

    Set download=True in config to auto-download AFHQv2 from
        https://github.com/clovaai/stargan-v2#animal-faces-hq-dataset-afhq.
    """
    resolution = cfg.resolution or 512

    afhq_base = Path(cfg.root) / "afhq"
    afhq_path = afhq_base / cfg.split

    if not afhq_path.exists():
        if cfg.download:
            _download_afhq(afhq_base)
        else:
            raise FileNotFoundError(
                f"AFHQ dataset not found at {afhq_path}. "
                f"Set download=True in config or download manually from: "
                f"https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip"
            )

    transform = utils.compose_transform(resolution, in_channels=3)
    dataset = datasets.ImageFolder(root=str(afhq_path), transform=transform)
    postprocess = utils.get_postprocess_fn()

    return DatasetFactoryOutput(
        dataset=dataset,
        resolution=resolution,
        in_channels=3,
        postprocess=postprocess,
    )
