from abc import ABC, abstractmethod
from typing import Callable
import jax.numpy as jnp

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
