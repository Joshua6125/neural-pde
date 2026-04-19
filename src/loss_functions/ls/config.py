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
    f: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    g: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    v0: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    sigma0: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    v_boundary: Callable[[jnp.ndarray], jnp.ndarray] | None = None
