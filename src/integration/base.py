from abc import ABC, abstractmethod
from typing import Callable
import jax.numpy as jnp

class NDCubeIntegration(ABC):
    """Base class for n-dimensional cube integration methods."""

    @abstractmethod
    def integrate_interior(self, func: Callable[[jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
        """Integrate a function over the interior domain."""
        pass

    @abstractmethod
    def integrate_boundary(self, func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
        """Integrate a function over the boundary (receives both points and normal vectors)."""
        pass

    def integrate(
            self,
            interior_func: Callable[[jnp.ndarray], jnp.ndarray],
            boundary_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Main integration method. Returns (total_loss, interior_loss, boundary_loss)."""
        loss_interior = self.integrate_interior(interior_func)
        loss_boundary = self.integrate_boundary(boundary_func)
        total_loss = loss_interior + loss_boundary
        return total_loss, loss_interior, loss_boundary
