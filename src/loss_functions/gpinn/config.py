"""gPINN configuration."""

from dataclasses import dataclass, field
from typing import Callable, Literal

import jax.numpy as jnp

from ...models import AnyModelConfig, MLPModelConfig
from ..base import AlgorithmConfig


@dataclass(frozen=True)
class gPINNConfig(AlgorithmConfig):
    """Configuration for gradient-enhanced PINN (gPINN).

    Adds weights for penalising the gradient of the PDE residual and the
    solution gradient.
    """
    kind: Literal["gpinn"] = "gpinn"
    model: AnyModelConfig = field(default_factory=MLPModelConfig)
    c: float | Callable[[jnp.ndarray], jnp.ndarray] = 1.0
    f: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0
    u0: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0
    ut0: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0
    ic_weight: float = 1.0
    bc_weight: float = 1.0
    residual_grad_weight: float = 0.0
    solution_grad_weight: float = 0.0
