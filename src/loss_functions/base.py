"""Base classes and interfaces for training algorithms."""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp

from ..train import TrainingMethod


@dataclass(frozen=True)
class AlgorithmConfig:
    """Base class for algorithm configurations.

    All algorithm configs must inherit from this and define a `kind` field
    to enable polymorphic instantiation.
    """
    kind: str


class Loss:
    """Base class for loss functions.

    Each algorithm provides loss functions that compute interior and boundary
    residuals for integration.
    """

    @staticmethod
    def _constant_function(value: float | jnp.ndarray) -> Callable[[jnp.ndarray], jnp.ndarray]:
        constant = jnp.asarray(value)

        def constant_fn(_: jnp.ndarray, constant: jnp.ndarray = constant) -> jnp.ndarray:
            return constant

        return constant_fn

    @abstractmethod
    def loss_interior(self, x_interior: jnp.ndarray) -> jnp.ndarray:
        """Compute loss at interior points."""
        pass

    @abstractmethod
    def loss_boundary(self, x_boundary: jnp.ndarray, normal_vector: jnp.ndarray) -> jnp.ndarray:
        """Compute loss at boundary points."""
        pass

    def loss_functions(self) -> tuple[
        Callable[[jnp.ndarray], jnp.ndarray],
        Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    ]:
        """Return tuple of (interior_loss_fn, boundary_loss_fn)."""
        return self.loss_interior, self.loss_boundary

Algorithm = TrainingMethod
