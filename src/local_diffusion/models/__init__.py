"""Model registry for locality diffusion baselines."""

from __future__ import annotations

from typing import Any, Callable, Dict


MODEL_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_model(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(cls_or_factory: Callable[..., Any]) -> Callable[..., Any]:
        MODEL_REGISTRY[name.lower()] = cls_or_factory
        return cls_or_factory

    return decorator


def create_model(name: str, **kwargs: Any) -> Any:
    factory = MODEL_REGISTRY.get(name.lower())
    if factory is None:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    return factory(**kwargs)


# Ensure baseline models are registered by default.
from . import nearest_dataset  # noqa: E402,F401
from . import optimal  # noqa: E402,F401
from . import wiener  # noqa: E402,F401
from . import pca_locality  # noqa: E402,F401
