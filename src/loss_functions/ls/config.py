"""Least-Squares configuration."""

from dataclasses import dataclass, field
from typing import Callable, Literal

import jax.numpy as jnp

from ...models import AnyModelConfig, NeuralNetModelConfig
from ..base import AlgorithmConfig


@dataclass(frozen=True)
class LSConfig(AlgorithmConfig):
    """Configuration for Least-Squares algorithm.

    Combines model architecture and PDE parameters into a single configuration.
    """
    kind: Literal["ls"] = "ls"
    model: AnyModelConfig = field(default_factory=NeuralNetModelConfig)
    f: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0
    g: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0
    v0: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0
    sigma0: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0
    v_boundary: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0
