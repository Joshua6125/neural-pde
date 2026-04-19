"""PINN configuration."""

from dataclasses import dataclass, field
from typing import Callable, Literal

import jax.numpy as jnp

from ...models import AnyModelConfig, NeuralNetModelConfig
from ..base import AlgorithmConfig


@dataclass(frozen=True)
class PINNConfig(AlgorithmConfig):
    """Configuration for Physics-Informed Neural Network algorithm.

    Combines model architecture and PDE parameters into a single configuration.
    """
    kind: Literal["pinn"] = "pinn"
    model: AnyModelConfig = field(default_factory=NeuralNetModelConfig)
    c: float | Callable[[jnp.ndarray], jnp.ndarray] = 1.0
    f: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    u0: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    ut0: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    ic_weight: float = 1.0
    bc_weight: float = 1.0
