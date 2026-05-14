"""FOSLS configuration."""

from dataclasses import dataclass, field
from typing import Callable, Literal

import jax.numpy as jnp

from ...models import AnyModelConfig, MLPModelConfig
from ..base import AlgorithmConfig


@dataclass(frozen=True)
class FOSLSConfig(AlgorithmConfig):
    """Configuration for the first-order system least-squares method."""
    kind: Literal["fosls"] = "fosls"
    model: AnyModelConfig = field(default_factory=MLPModelConfig)
    f: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0
    g: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0
