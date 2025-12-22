"""Optimal denoiser baseline using FAISS for efficient KNN search over full-resolution images."""

import logging
import os
import platform
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Fix OpenMP conflict between FAISS and PyTorch on macOS
# Must be set before importing faiss
if platform.system() == "Darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import faiss
# On macOS, FAISS threading can cause hangs due to fork safety issues
# Force single-threaded mode to avoid this
if platform.system() == "Darwin":
    faiss.omp_set_num_threads(1)

import numpy as np
import torch
from tqdm import tqdm
from local_diffusion.data import DatasetBundle
from local_diffusion.models.base import BaseDenoiser
from local_diffusion.models import register_model

LOGGER = logging.getLogger(__name__)


def save_optimal_index(
    faiss_index: faiss.Index,
    dataset_images: torch.Tensor,
    save_path: Path,
    dim: int,
) -> None:
    """Save the FAISS index and dataset images for optimal denoiser.
    
    Parameters
    ----------
    faiss_index : faiss.Index
        Trained FAISS index.
    dataset_images : torch.Tensor
        Flattened dataset images of shape [n_images, n_pixels].
    save_path : Path
        Directory in which to save the index.
    dim : int
        Dimension of the feature vectors.
    """
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save FAISS index
    index_path = save_path / "index.index"
    faiss.write_index(faiss_index, str(index_path))
    
    # Save metadata and dataset images
    torch.save(
        {
            "data": dataset_images.cpu(),
            "dim": dim,
        },
        save_path / "data.pt",
    )
    
    LOGGER.info(
        "Saved optimal denoiser index (%d images, dim=%d) to %s",
        dataset_images.shape[0],
        dim,
        save_path,
    )


def load_optimal_index(
    load_path: Path,
) -> Tuple[faiss.Index, torch.Tensor, int]:
    """Load the FAISS index and dataset images for optimal denoiser.
    
    Parameters
    ----------
    load_path : Path
        Directory from which to load the index.
    
    Returns
    -------
    faiss_index : faiss.Index
        Loaded FAISS index.
    dataset_images : torch.Tensor
        Flattened dataset images of shape [n_images, n_pixels].
    dim : int
        Dimension of the feature vectors.
    """
    index_path = load_path / "index.index"
    data_path = load_path / "data.pt"
    
    if not index_path.exists() or not data_path.exists():
        raise FileNotFoundError(
            f"Optimal denoiser index not found at {load_path}. "
            f"Expected index.index and data.pt to be present."
        )
    
    # Load FAISS index
    faiss_index = faiss.read_index(str(index_path))
    
    # Load metadata and dataset images
    saved_data = torch.load(data_path, weights_only=True)
    dataset_images = saved_data["data"]
    dim = saved_data["dim"]
    
    LOGGER.info(
        "Loaded optimal denoiser index (%d images, dim=%d) from %s",
        dataset_images.shape[0],
        dim,
        load_path,
    )
    
    return faiss_index, dataset_images, dim


@register_model("optimal")
class OptimalDenoiser(BaseDenoiser):
    """Optimal denoiser using softmax-weighted average over dataset.
    
    This implements the optimal denoiser formula.
    """
    
    def __init__(
        self,
        dataset: DatasetBundle,
        device: str,
        num_steps: int,
        *,
        params: Optional[Dict[str, object]] = None,
        **kwargs,
    ):
        params = params or {}
        super().__init__(
            resolution=dataset.resolution,
            device=device,
            num_steps=num_steps,
            in_channels=dataset.in_channels,
            dataset_name=dataset.name,
            **kwargs,
        )
        
        self.index_path = params.get("index_path", None)
        self.temperature = params.get("temperature", 1.0)
        self.num_neighbors = params.get("num_neighbors", 2000)
        
        # If path not provided, default to data/models/optimal/<dataset>_<resolution>
        if self.index_path is None:
            default_root = Path("data/models/optimal")
            self.index_path = default_root / f"{dataset.name}_{dataset.resolution}"
        else:
            self.index_path = Path(self.index_path)
        
        # Will be set during training
        self.faiss_index: Optional[faiss.Index] = None
        self.dataset_images: Optional[torch.Tensor] = None
        self.dim: Optional[int] = None

    def train(self, dataset: DatasetBundle):  # type: ignore[override]
        """Build or load the FAISS index for optimal denoising."""
        
        try:
            # Try to load existing index
            (
                self.faiss_index,
                self.dataset_images,
                self.dim,
            ) = load_optimal_index(self.index_path)
        except FileNotFoundError:
            # Build new index from dataset
            LOGGER.info("Optimal denoiser index not found. Building from dataset...")
            
            all_images = []
            for batch in tqdm(dataset.dataloader, desc="Building index"):
                if isinstance(batch, (tuple, list)):
                    images = batch[0]
                else:
                    images = batch
                
                images = images.to(self.device)
                
                # Flatten to [batch, n_pixels]
                images_flat = images.flatten(start_dim=1)
                all_images.append(images_flat.cpu())
            
            self.dataset_images = torch.cat(all_images, dim=0)  # [n_images, n_pixels]
            self.dim = self.dataset_images.shape[1]
            num_images = self.dataset_images.shape[0]
            
            # Create FAISS index
            LOGGER.info(f"Training FAISS index with {num_images} images (dim={self.dim})...")
            
            if num_images > 1_000_000:
                # For very large datasets, use IVF
                nlist = min(4096, num_images // 39)
                quantizer = faiss.IndexFlatL2(self.dim)
                index = faiss.IndexIVFFlat(quantizer, self.dim, nlist)
                # Train on a subset if dataset is too large
                train_size = min(100_000, num_images)
                train_data = self.dataset_images[:train_size].numpy().astype(np.float32)
                index.train(train_data)
                LOGGER.info("Done training FAISS index")
            else:
                # For smaller datasets, use exact search
                index = faiss.IndexFlatL2(self.dim)
            
            # Add vectors to index
            LOGGER.info("Adding data to FAISS index...")
            index.add(self.dataset_images.numpy().astype(np.float32))
            LOGGER.info("Done adding data to FAISS index")
            
            self.faiss_index = index
            
            # Save the index
            save_optimal_index(
                self.faiss_index,
                self.dataset_images,
                self.index_path,
                self.dim,
            )
            LOGGER.info(
                "Built and saved optimal denoiser index (%d images, dim=%d) to %s",
                num_images,
                self.dim,
                self.index_path,
            )
        
        return self

    @torch.no_grad()
    def denoise(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Denoise using optimal formula with FAISS-based softmax-weighted average.
        
        For each latent x_t, we:
        1. Scale x_t: x_scaled = x_t / sqrt(alpha_t)
        2. Search FAISS for nearest neighbors
        3. Compute distances and weights
        4. Return weighted average of neighbors
        """
        del generator, kwargs
        
        if self.faiss_index is None or self.dataset_images is None:
            raise RuntimeError(
                "Model not trained. Call model.train(dataset) before sampling."
            )
        
        timestep_index = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)
        
        # Get scheduler values
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep_index].to(self.device)
        beta_prod_t = 1 - alpha_prod_t
        
        # Scale latents: x_scaled = x_t / sqrt(alpha_t)
        # This is equivalent to comparing x_t with sqrt(alpha_t) * x_0
        sqrt_alpha = torch.sqrt(alpha_prod_t)
        latents_scaled = latents / sqrt_alpha  # [B, C, H, W]
        latents_flat = latents_scaled.flatten(start_dim=1)  # [B, n_pixels]
        
        # Search FAISS for nearest neighbors
        # Use numpy array for FAISS (it expects float32)
        query_vectors = latents_flat.cpu().numpy().astype(np.float32)
        k = min(self.num_neighbors, self.dataset_images.shape[0])
        distances_np, indices_np = self.faiss_index.search(query_vectors, k)  # [B, k]
        distances = torch.from_numpy(distances_np).to(self.device)  # [B, k]
        indices = torch.from_numpy(indices_np)  # [B, k]
        
        # Scale distances back down (to compensate for the scaling in the query)
        scaled_distances = distances * alpha_prod_t  # [B, k]
        neighbor_images = self.dataset_images[indices].to(self.device)  # [B, k, n_pixels]
        
        # Compute the softmax
        logits = -scaled_distances / (2 * beta_prod_t * self.temperature)  # [B, k]
        weights = torch.softmax(logits, dim=1)  # [B, k]
        
        # Weighted average: pred_x0 = sum(weights * neighbor_images)
        # [B, k] @ [B, k, n_pixels] -> [B, n_pixels]
        pred_x0_flat = torch.bmm(weights.unsqueeze(1), neighbor_images).squeeze(1)  # [B, n_pixels]
        pred_x0 = pred_x0_flat.view_as(latents)  # [B, C, H, W]
        return pred_x0
