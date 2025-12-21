"""Default dataset registrations based on torchvision."""

from __future__ import annotations

from torchvision import datasets

from local_diffusion.configuration import DatasetConfig

from . import utils
from .datasets import DatasetFactoryOutput, register_dataset


@register_dataset("mnist")
def build_mnist(cfg: DatasetConfig) -> DatasetFactoryOutput:
    transform = utils.compose_transform(28, in_channels=1)
    dataset = datasets.MNIST(
        root=cfg.root,
        train=cfg.split == "train",
        download=cfg.download,
        transform=transform,
    )
    postprocess = utils.get_postprocess_fn()
    return DatasetFactoryOutput(
        dataset=dataset,
        resolution=28,
        in_channels=1,
        postprocess=postprocess,
    )


@register_dataset("fashion_mnist")
def build_fashion_mnist(cfg: DatasetConfig) -> DatasetFactoryOutput:
    transform = utils.compose_transform(28, in_channels=1)
    dataset = datasets.FashionMNIST(
        root=cfg.root,
        train=cfg.split == "train",
        download=cfg.download,
        transform=transform,
    )
    postprocess = utils.get_postprocess_fn()
    return DatasetFactoryOutput(
        dataset=dataset,
        resolution=28,
        in_channels=1,
        postprocess=postprocess,
    )


@register_dataset("cifar10")
def build_cifar10(cfg: DatasetConfig) -> DatasetFactoryOutput:
    transform = utils.compose_transform(32, in_channels=3)
    dataset = datasets.CIFAR10(
        root=cfg.root,
        train=cfg.split == "train",
        download=cfg.download,
        transform=transform,
    )
    postprocess = utils.get_postprocess_fn()
    return DatasetFactoryOutput(
        dataset=dataset,
        resolution=32,
        in_channels=3,
        postprocess=postprocess,
    )

