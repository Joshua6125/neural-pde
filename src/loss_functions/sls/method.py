"""Least-Squares training algorithm."""

import jax
import jax.numpy as jnp

from ...models import BuiltModelProtocol
from ...train import TrainingMethod
from .loss import SLSLoss
from .config import SLSConfig


class SLS(TrainingMethod):
    """Least-Squares training algorithm."""

    def __init__(
        self,
        model: BuiltModelProtocol,
        config: SLSConfig,
    ):
        self.model = model
        self.config = config

    def init_params(self, rng_key: jax.Array, sample_input: jnp.ndarray):
        """Initialize model parameters and validate output contract."""
        params = self.model.init(rng_key, sample_input)

        # Validate model output structure
        outputs = self.model.apply(params, sample_input)
        if not isinstance(outputs, dict):
            raise ValueError("SLS model must return dict with 'v' and 'sigma' keys")
        if "v" not in outputs or "sigma" not in outputs:
            raise ValueError("SLS model must return dict with 'v' and 'sigma' keys")

        v_sample = outputs["v"]
        sigma_sample = outputs["sigma"]

        if jnp.asarray(v_sample).reshape(-1).shape[0] != 1:
            raise ValueError("SLS model 'v' output must be scalar")

        expected_sigma_dim = max(sample_input.shape[-1] - 1, 1)
        if jnp.asarray(sigma_sample).reshape(-1).shape[0] != expected_sigma_dim:
            raise ValueError(
                f"SLS model 'sigma' must output {expected_sigma_dim} values for input shape "
                f"{tuple(sample_input.shape)}"
            )

        return params

    def loss_functions(self, params):
        """Create loss functions for current parameters."""
        def sls_apply(x: jnp.ndarray) -> dict[str, jnp.ndarray]:
            return self.model.apply(params, x)

        def v_apply(x: jnp.ndarray) -> jnp.ndarray:
            return sls_apply(x)["v"]

        def sigma_apply(x: jnp.ndarray) -> jnp.ndarray:
            return sls_apply(x)["sigma"]

        loss = SLSLoss(
            v_model=v_apply,
            sigma_model=sigma_apply,
            f=self.config.f,
            g=self.config.g,
            v0=self.config.v0,
            sigma0=self.config.sigma0,
            v_boundary=self.config.v_boundary,
        )
        return loss.loss_functions()
