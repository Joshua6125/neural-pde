"""Variational PINN configuration."""

from dataclasses import dataclass, field
from typing import Callable, Literal

import jax.numpy as jnp

from ...models import AnyModelConfig, MLPConfig
from ..base import AlgorithmConfig


@dataclass(frozen=True)
class vPINNConfig(AlgorithmConfig):
    """Configuration for Variational PINN algorithm."""
    kind: Literal["vpinn"] = "vpinn"
    model: AnyModelConfig = field(default_factory=MLPConfig)
    c: float | Callable[[jnp.ndarray], jnp.ndarray] = 1.0
    f: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0
    u0: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0
    ut0: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0
    ic_weight: float = 1.0
    bc_weight: float = 1.0
    n_test_functions: int = 400
    domain_min: jnp.ndarray | None = None
    domain_max: jnp.ndarray | None = None
