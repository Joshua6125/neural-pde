"""Base classes for training infrastructure."""

from abc import ABC, abstractmethod
from typing import Any, Callable

import jax
import jax.numpy as jnp


class TrainingMethod(ABC):
    """Base interface for training algorithms.

    A training method specifies how to:
    - Initialize model parameters
    - Create loss functions for a given set of parameters

    This interface is implemented by all algorithms (PINN, LS, etc.)
    """

    @abstractmethod
    def init_params(self, rng_key: jax.Array, sample_input: jnp.ndarray) -> Any:
        """Initialize model parameters for this method.

        Should validate that the model output contract matches expectations.

        Parameters
        ----------
        rng_key : jax.Array
            PRNG key for reproducible random initialisation
        sample_input : jnp.ndarray
            Sample input for determining model dimensions

        Returns
        -------
        Any
            Initialized parameters (typically a PyTree)
        """
        ...

    @abstractmethod
    def loss_functions(
            self,
            params: Any
        ) -> tuple[
            Callable[[jnp.ndarray], jnp.ndarray],
            Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
        ]:
        """Return loss function callables for current parameters.

        The returned tuple should contain:
        - interior_loss: f(x_interior) -> scalar loss
        - boundary_loss: f(x_boundary, normal_vector) -> scalar loss

        Parameters
        ----------
        params : Any
            Model parameters

        Returns
        -------
        tuple[Callable, Callable]
            (interior_loss_fn, boundary_loss_fn)
        """
        ...
