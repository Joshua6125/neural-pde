from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Literal, TypeAlias
import jax.numpy as jnp

@dataclass(frozen=True)
class PINNLossConfig:
    kind: Literal["pinn"] = "pinn"
    c: float | Callable[[jnp.ndarray], jnp.ndarray] = 1.0
    f: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    u0: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    ut0: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    ic_weight: float = 1.0
    bc_weight: float = 1.0


@dataclass(frozen=True)
class LSLossConfig:
    kind: Literal["ls"] = "ls"
    f: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    g: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    v0: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    sigma0: Callable[[jnp.ndarray], jnp.ndarray] | None = None


AnyLossConfig: TypeAlias = PINNLossConfig | LSLossConfig


class LossBase(ABC):
    """Base class for loss functions."""

    @abstractmethod
    def loss_interior(self, x_interior: jnp.ndarray) -> jnp.ndarray:
        """Compute the interior loss given interior points."""
        pass

    @abstractmethod
    def loss_boundary(self, x_boundary: jnp.ndarray, normal_vector: jnp.ndarray) -> jnp.ndarray:
        """Compute the boundary loss given boundary points and their normal vectors."""
        pass

    def loss_functions(self) -> tuple[
            Callable[[jnp.ndarray], jnp.ndarray],
            Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
        ]:
        return self.loss_interior, self.loss_boundary
