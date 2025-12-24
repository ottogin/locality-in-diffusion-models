import torch
import numpy as np
from typing import Dict, Any, Optional, List

def calculate_r2_score(x: torch.Tensor, y: torch.Tensor) -> float:
    """Calculate R² score between two sets of images.
    
    Computes the coefficient of determination R² = 1 - (SS_res / SS_tot)
    averaged over the batch.
    
    Args:
        x: Ground truth images (or reference), shape (N, ...)
        y: Predicted images, shape (N, ...)
        
    Returns:
        float: R² score
    """
    # Ensure tensors are on CPU and flattened properly
    x_flat = x.detach().reshape(x.size(0), -1).cpu()
    y_flat = y.detach().reshape(y.size(0), -1).cpu()

    # Calculate R² score for each sample pair
    var_y = torch.var(y_flat, dim=1)
    ss_res = torch.sum((x_flat - y_flat) ** 2, dim=1)
    
    # Avoid division by zero
    var_y = torch.where(var_y == 0, torch.ones_like(var_y), var_y)
    
    r2 = 1 - (ss_res / (var_y * x_flat.size(1)))
    return r2.mean().item()


def calculate_mse(x: torch.Tensor, y: torch.Tensor) -> float:
    """Calculate Mean Squared Error between two sets of images.
    
    Args:
        x: First set of images, shape (N, ...)
        y: Second set of images, shape (N, ...)
        
    Returns:
        float: Mean Squared Error
    """
    x_flat = x.detach().reshape(x.size(0), -1).cpu()
    y_flat = y.detach().reshape(y.size(0), -1).cpu()
    mse = torch.mean((x_flat - y_flat) ** 2, dim=1)
    return mse.mean().item()


def calculate_l2_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    """Calculate L2 Euclidean distance between two sets of images.
    
    Args:
        x: First set of images, shape (N, ...)
        y: Second set of images, shape (N, ...)
        
    Returns:
        float: Mean L2 distance
    """
    x_flat = x.detach().reshape(x.size(0), -1).cpu()
    y_flat = y.detach().reshape(y.size(0), -1).cpu()
    dist = torch.norm(x_flat - y_flat, p=2, dim=1)
    return dist.mean().item()


