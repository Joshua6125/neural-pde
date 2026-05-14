"""Deep Ritz Method configuration."""

from dataclasses import dataclass, field
from typing import Callable, Literal

import jax.numpy as jnp

from ...models import AnyModelConfig, MLPModelConfig
from ..base import AlgorithmConfig


@dataclass(frozen=True)
class DRMConfig(AlgorithmConfig):
    """Configuration for the Deep Ritz Method."""

    kind: Literal["drm"] = "drm"
    model: AnyModelConfig = field(default_factory=MLPModelConfig)
    A: float | jnp.ndarray | Callable[[jnp.ndarray], jnp.ndarray] = 1.0
    c: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0
    f: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0
    g: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0
    boundary_weight: float = 1.0
