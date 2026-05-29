"""Least-Squares training algorithm."""

import jax
import jax.numpy as jnp

from ...models import BuiltModelProtocol
from ...train import TrainingMethod
from .loss import FOSLSLoss
from .config import FOSLSConfig


class FOSLS(TrainingMethod):
    """Least-Squares training algorithm."""

    def __init__(
        self,
        model: BuiltModelProtocol,
        config: FOSLSConfig,
    ):
        self.model = model
        self.config = config

    def init_params(self, rng_key: jax.Array, sample_input: jnp.ndarray):
        """Initialize model parameters and validate output contract."""
        params = self.model.init(rng_key, sample_input)

        # Validate model output structure
        outputs = self.model.apply(params, sample_input)
        if not isinstance(outputs, dict):
            raise ValueError("FOSLS model must return dict with 'v' and 'sigma' keys")
        if "v" not in outputs or "sigma" not in outputs:
            raise ValueError("FOSLS model must return dict with 'v' and 'sigma' keys")

        v_sample = outputs["v"]
        sigma_sample = outputs["sigma"]

        if jnp.asarray(v_sample).reshape(-1).shape[0] != 1:
            raise ValueError("FOSLS model 'v' output must be scalar")

        expected_sigma_dim = max(sample_input.shape[-1] - 1, 1)
        if jnp.asarray(sigma_sample).reshape(-1).shape[0] != expected_sigma_dim:
            raise ValueError(
                f"FOSLS model 'sigma' must output {expected_sigma_dim} values for input shape "
                f"{tuple(sample_input.shape)}"
            )

        return params

    def loss_functions(self, params):
        """Create loss functions for current parameters."""
        def fosls_apply(x: jnp.ndarray) -> jnp.ndarray:
            out = self.model.apply(params, x)
            return jnp.concatenate([jnp.atleast_1d(out["v"]), jnp.atleast_1d(out["sigma"])], axis=-1)

        loss = FOSLSLoss(
            model=fosls_apply,
            f=self.config.f,
            g=self.config.g,
            v0=self.config.v0,
            sigma0=self.config.sigma0,
            v_boundary=self.config.v_boundary,
            ic_weight=self.config.ic_weight,
        )
        return loss.loss_functions()
