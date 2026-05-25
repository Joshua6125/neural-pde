from abc import ABC, abstractmethod
from typing import Callable, Any
import jax.numpy as jnp
import jax

class NDCubeIntegration(ABC):
    """Base class for n-dimensional cube integration methods."""

    @abstractmethod
    def integrate_interior(self, func: Callable[[jnp.ndarray], Any]) -> Any:
        """Integrate a function over the interior domain."""
        pass

    @abstractmethod
    def integrate_boundary(
            self,
            func: Callable[[jnp.ndarray, jnp.ndarray], Any]
        ) -> Any:
        """Integrate a function over the boundary (receives both points and normal vectors)."""
        pass

    def integrate(
            self,
            interior_func: Callable[[jnp.ndarray], Any],
            boundary_func: Callable[[jnp.ndarray, jnp.ndarray], Any],
            rng_key: jax.Array | None = None,
        ) -> tuple[Any, Any]:
        """Main integration method. Returns (interior_loss, boundary_loss).
        The reduction into a single total scalar is left to the TrainingMethod."""
        loss_interior = self.integrate_interior(interior_func)
        loss_boundary = self.integrate_boundary(boundary_func)
        return loss_interior, loss_boundary
