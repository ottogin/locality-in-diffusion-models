from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from diffusers import DDIMScheduler
from tqdm import tqdm


@dataclass
class SamplingOutput:
    images: torch.Tensor
    timesteps: Optional[List[int]]
    trajectory_xt: Optional[List[torch.Tensor]]
    trajectory_x0: Optional[List[torch.Tensor]]


class BaseDenoiser(torch.nn.Module):
    """Base diffusion interface shared by analytic and learned models."""

    prediction_type: str = "epsilon"

    def __init__(
        self,
        resolution: int,
        device: str,
        num_steps: int,
        *args,
        beta_1: float = 0.0001,
        beta_T: float = 0.02,
        dataset_name: str = "cifar10",
        scheduler_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.n_channels = kwargs.get("in_channels", 3)
        self.img_resolution = resolution
        self.resolution = resolution
        self.dataset_name = dataset_name
        self.num_steps = num_steps

        scheduler_kwargs = scheduler_kwargs or {}
        self.scheduler = DDIMScheduler(
            beta_start=beta_1,
            beta_end=beta_T,
            beta_schedule="linear",
            prediction_type=self.prediction_type,
            **scheduler_kwargs,
        )
        self.scheduler.set_timesteps(num_steps)

    def denoise(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        """Predict the denoised sample ``x_0`` for a given latent tensor.

        Sub-classes **must** override this method to implement their analytic denoiser.
        The return value should be the predicted clean sample together with any
        auxiliary information that should be tracked (e.g. nearest-neighbour indices).
        """
        raise NotImplementedError

    def build_sample_output(
        self,
        images: torch.Tensor,
        trajectory_xt: Optional[List[torch.Tensor]],
        trajectory_x0: Optional[List[torch.Tensor]],
        timesteps: Optional[List[int]],
    ) -> SamplingOutput:
        return SamplingOutput(
            images=images,
            trajectory_xt=trajectory_xt,
            trajectory_x0=trajectory_x0,
            timesteps=timesteps,
        )

    def train(self, dataset):
        raise NotImplementedError

    def set_timesteps(self, num_steps: int) -> None:
        self.scheduler.set_timesteps(num_steps)
        self.num_steps = num_steps

    def prepare_latents(
        self,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        shape = (batch_size, self.n_channels, self.resolution, self.resolution)
        latents = torch.randn(shape, generator=generator, device=self.device)
        return latents * self.scheduler.init_noise_sigma

    def compute_noise_from_x0(
        self,
        x_t: torch.Tensor,
        pred_x0: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        t = int(timestep.item() if isinstance(timestep, torch.Tensor) else timestep)
        alpha_prod = self.scheduler.alphas_cumprod[t].to(x_t.device)
        beta_prod = 1 - alpha_prod
        sqrt_alpha = torch.sqrt(alpha_prod)
        sqrt_beta = torch.sqrt(beta_prod + 1e-8)
        return (x_t - sqrt_alpha * pred_x0) / sqrt_beta

    @torch.no_grad()
    def _image_preprocess(self, img: torch.Tensor) -> torch.Tensor:
        imgs = torch.nn.functional.interpolate(
            img[:, : self.n_channels, ...].to(self.device),
            size=(self.img_resolution, self.img_resolution),
            mode="bilinear",
            align_corners=False,
        )
        img_rescaled = (imgs - 0.5) * 2
        return img_rescaled

    @torch.no_grad()
    def _image_postprocess(self, img: torch.Tensor) -> torch.Tensor:
        img_rescaled = (img + 1) / 2
        return img_rescaled.clamp(0, 1)

    @torch.no_grad()
    def sample(
        self,
        *,
        num_samples: int,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
        return_intermediates: bool = False,
    ) -> SamplingOutput:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        batches: List[SamplingOutput] = []
        total_generated = 0
        while total_generated < num_samples:
            current_batch = min(batch_size, num_samples - total_generated)
            batch_result = self._sample_batch(
                batch_size=current_batch,
                generator=generator,
                return_intermediates=return_intermediates,
            )
            batches.append(batch_result)
            total_generated += current_batch

        images = torch.cat([b.images for b in batches], dim=0)

        trajectory_xt: Optional[List[torch.Tensor]] = None
        trajectory_x0: Optional[List[torch.Tensor]] = None
        timesteps: Optional[List[int]] = None

        if return_intermediates:
            for batch in batches:
                if batch.trajectory_xt is not None:
                    if trajectory_xt is None:
                        trajectory_xt = [tensor.clone() for tensor in batch.trajectory_xt]
                    else:
                        for idx, tensor in enumerate(batch.trajectory_xt):
                            trajectory_xt[idx] = torch.cat([trajectory_xt[idx], tensor], dim=0)

                if batch.trajectory_x0 is not None:
                    if trajectory_x0 is None:
                        trajectory_x0 = [tensor.clone() for tensor in batch.trajectory_x0]
                    else:
                        for idx, tensor in enumerate(batch.trajectory_x0):
                            trajectory_x0[idx] = torch.cat([trajectory_x0[idx], tensor], dim=0)

                if batch.timesteps is not None and timesteps is None:
                    timesteps = list(batch.timesteps)

        return self.build_sample_output(
            images=images,
            trajectory_xt=trajectory_xt,
            trajectory_x0=trajectory_x0,
            timesteps=timesteps,
        )

    def _sample_batch(
        self,
        *,
        batch_size: int,
        generator: Optional[torch.Generator],
        return_intermediates: bool,
    ) -> SamplingOutput:
        latents = self.prepare_latents(batch_size, generator=generator)

        trajectory_xt: Optional[List[torch.Tensor]] = [] if return_intermediates else None
        trajectory_x0: Optional[List[torch.Tensor]] = [] if return_intermediates else None
        timesteps_list: Optional[List[int]] = [] if return_intermediates else None
        last_pred_x0: Optional[torch.Tensor] = None

        timesteps_iter = tqdm(
            enumerate(self.scheduler.timesteps),
            total=len(self.scheduler.timesteps),
            desc="Sampling",
            unit="step",
        )
        for step_idx, timestep in timesteps_iter:
            pred_x0 = self.denoise(
                latents,
                timestep,
                generator=generator,
            )

            if not isinstance(pred_x0, torch.Tensor):
                raise TypeError("denoise must return a torch.Tensor as the first value")

            predicted_noise = self.compute_noise_from_x0(latents, pred_x0, timestep)
            step_output = self.scheduler.step(
                model_output=predicted_noise,
                timestep=timestep,
                sample=latents,
            )

            if return_intermediates:
                if trajectory_xt is not None:
                    trajectory_xt.append(latents.detach().cpu())
                if trajectory_x0 is not None:
                    trajectory_x0.append(pred_x0.detach().cpu())
                if timesteps_list is not None:
                    timestep_value = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)
                    timesteps_list.append(timestep_value)

            last_pred_x0 = pred_x0
            latents = step_output.prev_sample

        if last_pred_x0 is None:
            raise RuntimeError("Sampling loop did not execute any timesteps.")

        return SamplingOutput(
            images=last_pred_x0.detach().cpu(),
            trajectory_xt=trajectory_xt if return_intermediates else None,
            trajectory_x0=trajectory_x0 if return_intermediates else None,
            timesteps=timesteps_list if return_intermediates else None
        )