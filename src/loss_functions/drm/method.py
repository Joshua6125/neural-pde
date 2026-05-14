"""Deep Ritz Method training algorithm."""

import jax
import jax.numpy as jnp

from ...models import BuiltModelProtocol
from ...train import TrainingMethod
from .config import DRMConfig
from .loss import DRMLoss


class DRM(TrainingMethod):
    """Deep Ritz Method training algorithm."""

    def __init__(
        self,
        model: BuiltModelProtocol,
        config: DRMConfig,
    ):
        self.model = model
        self.config = config

    def init_params(self, rng_key: jax.Array, sample_input: jnp.ndarray):
        """Initialize model parameters and validate output contract."""
        params = self.model.init(rng_key, sample_input)

        outputs = self.model.apply(params, sample_input)
        if not isinstance(outputs, dict) or "u" not in outputs:
            raise ValueError("DRM model must return dict with 'u' key (scalar)")

        if jnp.asarray(outputs["u"]).reshape(-1).shape[0] != 1:
            raise ValueError("DRM model 'u' output must be scalar")

        return params

    def loss_functions(self, params):
        """Create loss functions for current parameters."""

        def u_apply(x: jnp.ndarray) -> jnp.ndarray:
            return self.model.apply(params, x)["u"]

        loss = DRMLoss(
            u_model=u_apply,
            A=self.config.A,
            c=self.config.c,
            f=self.config.f,
            g=self.config.g,
            boundary_weight=self.config.boundary_weight,
        )
        return loss.loss_functions()
