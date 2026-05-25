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

    This interface is implemented by all algorithms (PINN, SLS, etc.)
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

    def aggregate_loss(self, interior: Any, boundary: Any) -> jnp.ndarray:
        """Aggregate interior and boundary integration results into a single scalar loss.

        Default implementation assumes interior and boundary are PyTrees of scalar totals,
        and simply sums them up. Can be overridden by specific methods (e.g. vPINN).
        """

        def tree_sum(tree: Any) -> jnp.ndarray:
            leaves = jax.tree_util.tree_leaves(tree)
            if not leaves:
                return jnp.array(0.0)
            return sum(jnp.sum(leaf) for leaf in leaves) # type: ignore

        return tree_sum(interior) + tree_sum(boundary)
