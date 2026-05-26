"""vPINN training algorithm."""

from typing import Any
import jax
import jax.numpy as jnp

from ...models import BuiltModelProtocol
from ...train import TrainingMethod
from .loss import vPINNLoss
from .config import vPINNConfig


class vPINN(TrainingMethod):
    """Variational Physics-Informed Neural Network algorithm.

    This method evaluates integrals of the weak/projected residual first,
    and aggregate_loss performs the squaring operation after integration.
    """

    def __init__(
        self,
        model: BuiltModelProtocol,
        config: vPINNConfig,
    ):
        self.model = model
        self.config = config

    def init_params(self, rng_key: jax.Array, sample_input: jnp.ndarray):
        """Initialize model parameters and validate output contract."""
        params = self.model.init(rng_key, sample_input)

        outputs = self.model.apply(params, sample_input)
        if not isinstance(outputs, dict) or "u" not in outputs:
            raise ValueError("vPINN model must return dict with 'u' key (scalar)")

        if jnp.asarray(outputs["u"]).reshape(-1).shape[0] != 1:
            raise ValueError("vPINN model 'u' output must be scalar")

        return params

    def loss_functions(self, params):
        """Create loss functions for current parameters."""
        def u_apply(x: jnp.ndarray) -> jnp.ndarray:
            return self.model.apply(params, x)["u"]

        loss = vPINNLoss(
            u_model=u_apply,
            c=self.config.c,
            f=self.config.f,
            u0=self.config.u0,
            ut0=self.config.ut0,
            ic_weight=self.config.ic_weight,
            bc_weight=self.config.bc_weight,
            n_test_functions=self.config.n_test_functions,
        )
        return loss.loss_functions()

    def aggregate_loss(self, interior: Any, boundary: Any) -> jnp.ndarray:
        """For vPINN, the interior loss is a vector of integral evaluations.
        We square and sum them here to form the final variance objective."""
        # interior is expected to be a PyTree containing the evaluated integrals.
        # boundary is expected to evaluate to scalars directly (squared residuals).

        def square_and_sum(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(x ** 2)

        interior_loss_total = jax.tree_util.tree_reduce(
            lambda x, y: x + y,
            jax.tree_util.tree_map(square_and_sum, interior),
            0.0
        )

        def tree_sum(tree: Any) -> jnp.ndarray:
            leaves = jax.tree_util.tree_leaves(tree)
            if not leaves:
                return jnp.array(0.0)
            return sum(jnp.sum(leaf) for leaf in leaves) # type: ignore

        boundary_loss_total = tree_sum(boundary)

        return interior_loss_total + boundary_loss_total
    