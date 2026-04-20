"""PINN training algorithm."""

import jax
import jax.numpy as jnp

from ...models import BuiltModelProtocol
from ...train import TrainingMethod
from .loss import PINNLoss
from .config import PINNConfig


class PINN(TrainingMethod):
    """Physics-Informed Neural Network training algorithm."""

    def __init__(
        self,
        model: BuiltModelProtocol,
        config: PINNConfig,
    ):
        self.model = model
        self.config = config

    def init_params(self, rng_key: jax.Array, sample_input: jnp.ndarray):
        """Initialize model parameters and validate output contract."""
        params = self.model.init(rng_key, sample_input)

        # Validate model output structure
        outputs = self.model.apply(params, sample_input)
        if not isinstance(outputs, dict) or "u" not in outputs:
            raise ValueError("PINN model must return dict with 'u' key (scalar)")

        if jnp.asarray(outputs["u"]).reshape(-1).shape[0] != 1:
            raise ValueError("PINN model 'u' output must be scalar")

        return params

    def loss_functions(self, params):
        """Create loss functions for current parameters."""
        def u_apply(x: jnp.ndarray) -> jnp.ndarray:
            return self.model.apply(params, x)["u"]

        loss = PINNLoss(
            u_model=u_apply,
            c=self.config.c,
            f=self.config.f,
            u0=self.config.u0,
            ut0=self.config.ut0,
            ic_weight=self.config.ic_weight,
            bc_weight=self.config.bc_weight,
        )
        return loss.loss_functions()
