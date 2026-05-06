"""gPINN training algorithm."""

import jax
import jax.numpy as jnp

from ...models import BuiltModelProtocol
from ...train import TrainingMethod
from .loss import gPINNLoss
from .config import gPINNConfig


class gPINN(TrainingMethod):
    """Gradient-enhanced PINN training algorithm wrapper."""

    def __init__(self, model: BuiltModelProtocol, config: gPINNConfig):
        self.model = model
        self.config = config

    def init_params(self, rng_key: jax.Array, sample_input: jnp.ndarray):
        params = self.model.init(rng_key, sample_input)

        outputs = self.model.apply(params, sample_input)
        if not isinstance(outputs, dict) or "u" not in outputs:
            raise ValueError("gPINN model must return dict with 'u' key (scalar)")

        if jnp.asarray(outputs["u"]).reshape(-1).shape[0] != 1:
            raise ValueError("gPINN model 'u' output must be scalar")

        return params

    def loss_functions(self, params):
        def u_apply(x: jnp.ndarray) -> jnp.ndarray:
            return self.model.apply(params, x)["u"]

        loss = gPINNLoss(
            u_model=u_apply,
            c=self.config.c,
            f=self.config.f,
            u0=self.config.u0,
            ut0=self.config.ut0,
            ic_weight=self.config.ic_weight,
            bc_weight=self.config.bc_weight,
            residual_grad_weight=self.config.residual_grad_weight,
        )
        return loss.loss_functions()
