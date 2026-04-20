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
    f: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0
    u0: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0
    ut0: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0
    ic_weight: float = 1.0
    bc_weight: float = 1.0
