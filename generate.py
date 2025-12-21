"""Entry-point script for generating samples with analytical diffuision models."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision.utils import make_grid, save_image

from local_diffusion.configuration import (
    RunPaths,
    config_to_dict,
    ensure_run_directory,
    get_git_tracked_paths,
    load_config,
    save_config,
    snapshot_codebase,
)
from local_diffusion.data import DatasetBundle, build_dataset
from local_diffusion.models import create_model
from local_diffusion.models.base import SamplingOutput
from local_diffusion.models.baseline_unet import BaselineUNet
from local_diffusion.metrics import calculate_r2_score, calculate_mse

LOGGER = logging.getLogger("local_diffusion.generate")


def evaluate_main_model(
    model,
    dataset_bundle: DatasetBundle,
    result: SamplingOutput,
    cfg: OmegaConf,
    run_paths: RunPaths,
    wandb_run,
    sampling_time_total: float,
) -> None:
    """
    Handles reporting and logging for the main model's results.
    1. Saves final images and grid.
    2. Saves trajectories (tensors and images).
    3. Computes and logs basic metrics.
    4. Logs trajectory metrics (per timestep, including batch values).
    """
    LOGGER.info("Evaluating main model results...")
    metrics = {}
    num_batches = math.ceil(cfg.sampling.num_samples / cfg.sampling.batch_size)
    total_steps = num_batches * cfg.sampling.num_inference_steps
    avg_step_time = sampling_time_total / total_steps if total_steps > 0 else 0.0
    metrics["main_sampling_time_total"] = sampling_time_total
    metrics["main_sampling_time_per_step"] = avg_step_time
    
    # --- 1. Prepare and Save Images ---
    images_tensor = result.images
    if images_tensor.device != model.device:
        images_tensor = images_tensor.to(model.device)
    
    # Detach and postprocess
    processed_images = dataset_bundle.postprocess(images_tensor).detach().cpu()
    
    if cfg.metrics.output.save_final_images:
        run_paths.images.mkdir(parents=True, exist_ok=True)
        for idx, image in enumerate(processed_images):
            save_image(image, run_paths.images / f"sample_{idx:04d}.png")

    grid_tensor = None
    if cfg.metrics.output.save_image_grid:
        grid_rows = 4 # max(1, math.ceil(math.sqrt(processed_images.size(0))))
        grid_tensor = make_grid(processed_images, nrow=grid_rows, normalize=False)
        save_image(grid_tensor, run_paths.run_dir / "grid.png")
        if wandb_run:
            import wandb
            # Log grid with step=0 to avoid conflicts with trajectory step logging
            wandb_run.log({"grid": wandb.Image(grid_tensor, caption="Generated samples grid")}, step=0)
        LOGGER.info("Saved image grid to %s", run_paths.run_dir / "grid.png")

    # --- 2. Save Trajectories ---
    if result.trajectory_xt and result.trajectory_x0:
        if cfg.metrics.output.save_intermediate_images:
            _save_intermediates(dataset_bundle, result, run_paths)

    _log_metrics(metrics, run_paths, wandb_run)


def evaluate_comparison(
    dataset_bundle: DatasetBundle,
    main_result: SamplingOutput,
    baseline_result: SamplingOutput,
    baseline_model: BaselineUNet,
    cfg: OmegaConf,
    run_paths: RunPaths,
    wandb_run,
) -> None:
    """
    Handles reporting and logging for baseline comparisons.
    1. Computes R2/MSE between main model and baseline per step.
    2. Generates and saves comparison grids per step.
    3. Logs comparison metrics per step to WandB.
    4. Logs final comparison metrics as summary.
    """
    LOGGER.info("Evaluating comparison metrics...")
    metrics = {}
    
    if not (main_result.trajectory_x0 and baseline_result.trajectory_x0):
        LOGGER.warning("Trajectories missing, skipping detailed comparison.")
        return

    r2_traj_means, mse_traj_means = [], []
    r2_single_means, mse_single_means = [], []
    
    comp_dir = run_paths.run_dir / "comparison"
    comp_dir.mkdir(exist_ok=True)

    steps = min(len(main_result.trajectory_x0), len(baseline_result.trajectory_x0))
    
    # Structure for JSON details
    detailed_comparison_metrics = {}

    import wandb

    for i in range(steps):
        t = main_result.timesteps[i]
        step_key = f"step_{1000 - t}" # Consistent naming with main evaluation
        
        # 1. Trajectory vs Trajectory (x0)
        main_x0 = main_result.trajectory_x0[i]
        base_x0 = baseline_result.trajectory_x0[i]
        
        r2 = calculate_r2_score(main_x0, base_x0)
        mse = calculate_mse(main_x0, base_x0)
        
        r2_traj_means.append(r2)
        mse_traj_means.append(mse)
        
        # 2. Single Step Prediction (Main xt -> Baseline x0)
        xt = main_result.trajectory_xt[i].to(baseline_model.device)
        
        with torch.no_grad():
            base_pred_single = baseline_model.denoise(xt, t)
        base_pred_single = base_pred_single.cpu()
        
        r2_single = calculate_r2_score(main_x0, base_pred_single)
        mse_single = calculate_mse(main_x0, base_pred_single)
        
        r2_single_means.append(r2_single)
        mse_single_means.append(mse_single)

        # 3. Save Comparison Image & Log Step Metrics
        # WandB log payload
        diffusion_step = 1000 - t  # Inverted for WandB consistency
        log_payload = {
            "r2_score_vs_unet": r2,
            "mse_score_vs_unet": mse,
            "r2_score_vs_unet_single": r2_single,
            "mse_score_vs_unet_single": mse_single,
        }
        
        # JSON payload - keep full precision/structure if needed, 
        # but R2/MSE are usually scalars (averaged over batch inside calc function?)
        # Let's check calculate_r2_score implementation. 
        # Usually these return a single float. If so, just saving them is enough.
        detailed_comparison_metrics[step_key] = {
            "diffusion_step": diffusion_step,
            "r2_score_vs_unet": r2,
            "mse_score_vs_unet": mse,
            "r2_score_vs_unet_single": r2_single,
            "mse_score_vs_unet_single": mse_single,
        }

        if cfg.metrics.output.save_intermediate_images:
            # Save x0 comparison grid
            grid = _save_comparison_step_grid(
                dataset_bundle,
                [main_x0, base_pred_single, base_x0],
                t,
                comp_dir,
                filename_suffix="comparison_x0"
            )
            if wandb_run:
                log_payload["comparison_grid_x0"] = wandb.Image(grid, caption=f"Comparison Step {t}")
            
            # Save xt comparison grid (only trajectory vs trajectory)
            main_xt = main_result.trajectory_xt[i]
            base_xt = baseline_result.trajectory_xt[i]
            grid_xt = _save_comparison_step_grid(
                dataset_bundle,
                [main_xt, base_xt],
                t,
                comp_dir,
                filename_suffix="comparison_xt"
            )
            if wandb_run:
                log_payload["comparison_grid_xt"] = wandb.Image(grid_xt, caption=f"Comparison xt Step {t}")

        if wandb_run:
            wandb_run.log(log_payload, step=diffusion_step)

    metrics["comparison_details"] = detailed_comparison_metrics

    if steps > 0:
        metrics["r2_score_vs_unet_final"] = r2_traj_means[-1]
        metrics["mse_score_vs_unet_final"] = mse_traj_means[-1]

    _log_metrics(metrics, run_paths, wandb_run)


def _save_intermediates(dataset, result, run_paths):
    xt_dir = run_paths.intermediate_images / "x_t"
    x0_dir = run_paths.intermediate_images / "x0_pred"
    xt_dir.mkdir(parents=True, exist_ok=True)
    x0_dir.mkdir(parents=True, exist_ok=True)

    for i, (xt, x0) in enumerate(zip(result.trajectory_xt, result.trajectory_x0)):
        t_label = result.timesteps[i] if result.timesteps else i
        batch_size = xt.shape[0]
        
        for j, img in enumerate(dataset.postprocess(xt)):
             global_idx = i * batch_size + j
             save_image(img.detach().cpu(), xt_dir / f"step_{t_label:04d}_sample_{global_idx:05d}.png")
        
        for j, img in enumerate(dataset.postprocess(x0)):
             global_idx = i * batch_size + j
             save_image(img.detach().cpu(), x0_dir / f"step_{t_label:04d}_sample_{global_idx:05d}.png")


def _save_comparison_step_grid(dataset, trajectories, t, save_dir, filename_suffix="comparison"):
    """
    Saves comparison grid to disk and returns the tensor for logging.
    
    Args:
        dataset: Dataset bundle for postprocessing
        trajectories: List of trajectory tensors to compare (e.g., [main_x0, base_single, base_x0])
        t: Timestep value
        save_dir: Directory to save the grid
        filename_suffix: Suffix for the filename (default: "comparison")
    
    Returns:
        Grid tensor
    """
    if not trajectories:
        raise ValueError("trajectories list cannot be empty")
    
    # Determine batch size from first trajectory
    n = min(8, trajectories[0].shape[0])
    
    # Postprocess all trajectories
    combined_list = []
    for traj in trajectories:
        traj_img = dataset.postprocess(traj[:n])
        combined_list.extend([img for img in traj_img])

    grid = make_grid(combined_list, nrow=n, padding=2, normalize=False)
    save_image(grid, save_dir / f"step_{t:04d}_{filename_suffix}.png")
    return grid


def _log_metrics(metrics: Dict[str, Any], run_paths: RunPaths, wandb_run):
    """Helper to append metrics to json file and log to wandb."""
    if not metrics:
        return

    # Update/Create metrics.json
    metrics_path = run_paths.run_dir / "metrics.json"
    current_metrics = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            try:
                current_metrics = json.load(f)
            except json.JSONDecodeError:
                pass
    
    # We don't want to log the huge 'details' dictionaries to WandB summaries or CLI console,
    # but we DO want them in the JSON file.
    # Separate summary metrics (scalars) from detailed metrics (nested dicts/lists)
    
    summary_metrics = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
    detailed_metrics = {k: v for k, v in metrics.items() if isinstance(v, dict)}
    
    # Merge everything into JSON
    current_metrics.update(summary_metrics)
    current_metrics.update(detailed_metrics)
    
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(current_metrics, f, indent=2)
    
    LOGGER.info("Updated metrics in %s", metrics_path)
    
    # Report ONLY summary metrics to CLI
    if summary_metrics:
        print("\n=== Run Metrics ===")
        for k, v in sorted(summary_metrics.items()):
            if isinstance(v, float):
                print(f"{k}: {v:.6g}")
            else:
                print(f"{k}: {v}")
        print("===================\n")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate samples using locality baselines")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a configuration file (relative to configs/ or absolute)",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional config overrides in dotlist form (e.g. sampling.num_samples=8)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    run_paths = ensure_run_directory(cfg)
    _setup_file_logging(run_paths.logs)
    LOGGER.info("Run directory: %s", run_paths.run_dir)

    save_config(cfg, run_paths.config)

    if cfg.metrics.output.code_snapshot:
        snapshot_codebase(run_paths.run_dir / "code_snapshot", project_root=_project_root())

    set_random_seeds(cfg.experiment.seed)

    dataset = build_dataset(cfg.dataset)

    model_params = (
        OmegaConf.to_container(cfg.model.params, resolve=True)
        if not isinstance(cfg.model.params, dict)
        else cfg.model.params
    )

    model = create_model(
        cfg.model.name,
        dataset=dataset,
        device=cfg.experiment.device,
        num_steps=cfg.sampling.num_inference_steps,
        params=model_params,
    )
    
    # Train the model (load/compute dataset-dependent parameters)
    model.train(dataset)

    generator = torch.Generator(device=model.device)
    generator.manual_seed(cfg.experiment.seed)

    wandb_run = init_wandb(cfg, run_paths.run_dir)
    
    # if wandb_run:
    #     wandb_run.define_metric("diffusion_step")
    #     wandb_run.define_metric("*", step_metric="diffusion_step")

    try:
        # 1. Run sampling with the original method
        LOGGER.info("Generating samples with main model...")
        start_time = time.perf_counter()
        # Return intermediates if we need them for saving trajectories, intermediate images, or comparison
        return_intermediates = (
            cfg.metrics.output.save_intermediate_images 
            or cfg.metrics.baseline_path is not None  # Need intermediates for comparison
        )
        
        result: SamplingOutput = model.sample(
            num_samples=cfg.sampling.num_samples,
            batch_size=cfg.sampling.batch_size,
            generator=generator,
            return_intermediates=return_intermediates,
        )
        end_time = time.perf_counter()

        total_time = end_time - start_time

        # 2. Call a function to report/log everything that is based only on the original method
        evaluate_main_model(
            model,
            dataset,
            result,
            cfg,
            run_paths,
            wandb_run,
            sampling_time_total=total_time,
        )

        # Check if baseline is requested
        baseline_path = cfg.get("metrics", {}).get("baseline_path")
        
        if baseline_path:
            # 3. Run baseline neural network
            LOGGER.info(f"Running baseline comparison against {baseline_path}")
            baseline_model = BaselineUNet(
                resolution=dataset.resolution,
                device=cfg.experiment.device,
                num_steps=cfg.sampling.num_inference_steps,
                model_path=baseline_path,
                dataset_name=cfg.dataset.name,
                in_channels=dataset.in_channels,
                out_channels=dataset.in_channels,
            )
            
            # Sample baseline with same seed to ensure comparable noise if possible
            baseline_gen = torch.Generator(device=model.device)
            baseline_gen.manual_seed(cfg.experiment.seed)
            
            # Baseline always needs intermediates for comparison
            baseline_result = baseline_model.sample(
                num_samples=cfg.sampling.num_samples,
                batch_size=cfg.sampling.batch_size,
                generator=baseline_gen,
                return_intermediates=return_intermediates,
            )

            # 4. Call a function to report/log comparison results
            evaluate_comparison(
                dataset, 
                result, 
                baseline_result, 
                baseline_model, 
                cfg, 
                run_paths, 
                wandb_run
            )

    finally:
        if wandb_run is not None:
            wandb_run.finish()


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_wandb(cfg: OmegaConf, run_dir: Path):
    if not cfg.metrics.wandb.enabled:
        return None
    mode = cfg.metrics.wandb.mode.lower()
    if mode == "disabled":
        os.environ.setdefault("WANDB_MODE", "disabled")
        return None

    import wandb

    os.environ.setdefault("WANDB_MODE", mode)
    os.makedirs(cfg.paths.wandb, exist_ok=True)

    # Combine tags: wandb tags (which may inherit from experiment) + model name
    wandb_tags = list(cfg.metrics.wandb.tags) if cfg.metrics.wandb.tags else []
    all_tags = list({*wandb_tags, cfg.model.name})

    project_root = _project_root()

    run = wandb.init(
        project=cfg.metrics.wandb.project,
        entity=cfg.metrics.wandb.entity,
        config=config_to_dict(cfg),
        dir=cfg.paths.wandb,
        name=cfg.experiment.run_name,
        job_type=cfg.metrics.wandb.job_type,
        tags=all_tags,
    )
    run.config.update({"run_dir": str(run_dir)}, allow_val_change=True)

    tracked_paths = get_git_tracked_paths(project_root)
    if tracked_paths:
        tracked_rel_paths = {p.as_posix() for p in tracked_paths}

        def _include_only_tracked(path: str) -> bool:
            # wandb passes paths relative to `root`; compare using POSIX style
            path_obj = Path(path)
            try:
                rel = path_obj.resolve().relative_to(project_root)
                return rel.as_posix() in tracked_rel_paths
            except ValueError:
                # Fall back to the given path if it is already relative
                return path_obj.as_posix() in tracked_rel_paths

        run.log_code(root=project_root, include_fn=_include_only_tracked)
    else:
        LOGGER.warning("Skipping wandb code logging: no git-tracked files found in %s", project_root)
    return run


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _setup_file_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "generate.log"

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(file_handler)
    LOGGER.info("File logging enabled at %s", log_path)


if __name__ == "__main__":
    main()
