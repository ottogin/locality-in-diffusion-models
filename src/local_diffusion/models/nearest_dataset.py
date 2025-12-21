from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

from local_diffusion.data import DatasetBundle
from local_diffusion.models import register_model
from local_diffusion.models.base import BaseDenoiser

LOGGER = logging.getLogger(__name__)


@register_model("nearest_dataset")
class NearestDatasetDenoiser(BaseDenoiser):
    """Baseline that retrieves the nearest dataset image at each diffusion step."""

    def __init__(
        self,
        dataset: DatasetBundle,
        device: str,
        num_steps: int,
        *,
        params: Optional[Dict[str, object]] = None,
        **kwargs,
    ) -> None:
        params = params or {}
        super().__init__(
            resolution=dataset.resolution,
            device=device,
            num_steps=num_steps,
            in_channels=dataset.in_channels,
            dataset_name=dataset.name,
            **kwargs,
        )

        self.dataset = dataset


    def train(self, dataset: DatasetBundle):  # type: ignore[override]
        """Load dataset reference images for nearest-neighbor lookup."""
        LOGGER.info("Loading dataset '%s' split '%s' into memory", dataset.name, dataset.split)
        
        images: List[torch.Tensor] = []
        for batch in dataset.dataloader:
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            images.append(batch)
        
        dataset_tensor = torch.cat(images, dim=0).contiguous().to(self.device)
        LOGGER.info("Loaded dataset tensor shape: %s", tuple(dataset_tensor.shape))
        
        self.register_buffer("dataset_images", dataset_tensor)
        self.to(self.device)
        
        return self

    @torch.no_grad()
    def denoise(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
        **_: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        if self.dataset_images is None:
            raise RuntimeError(
                "Model not trained. Call model.train(dataset) before sampling."
            )

        latents_flat = latents.flatten(start_dim=1)
        distances = torch.cdist(latents_flat, self.dataset_images.flatten(start_dim=1))
        min_dist, indices = torch.min(distances, dim=1)
        pred_x0 = self.dataset_images[indices]

        return pred_x0
