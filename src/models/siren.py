from typing import Mapping
import math

import jax
import jax.numpy as jnp
import flax.linen as nn


class SIREN(nn.Module):
    """SIREN: sinusoidal representation network (Sitzmann et al., 2020).

    This implementation follows the initialisation scheme recommended in the
    paper: the first layer is initialised uniformly in [-1/in, 1/in], while
    subsequent layers and final heads use uniform initialisation with bound
    sqrt(6 / fan_in) / w0 (fan_in = input dimension to the layer).

    The module returns a `dict[str, jnp.ndarray]` of named output heads to
    match the rest of the codebase.
    """

    hidden_dim: int
    num_layers: int
    output_heads: Mapping[str, int]
    w0: float = 30.0  # omega_0 for the first layer
    w0_hidden: float = 1.0  # omega_0 used for hidden/output layers (usually 1.0)

    def _first_kernel_init(self):
        def init(key, shape, dtype=jnp.float32):
            # shape: (in_features, out_features)
            in_features = int(shape[0])
            bound = 1.0 / in_features
            return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)

        return init

    def _hidden_kernel_init(self, w0: float):
        def init(key, shape, dtype=jnp.float32):
            in_features = int(shape[0])
            bound = math.sqrt(6.0 / in_features) / float(w0)
            return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)

        return init

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> dict[str, jnp.ndarray]:
        was_unbatched = x.ndim == 1
        x_in = x[None, :] if was_unbatched else x

        # normalize coordinates from [0,1] to [-1,1]
        h = 2.0 * x_in - 1.0

        # first layer
        h = nn.Dense(self.hidden_dim, kernel_init=self._first_kernel_init(), name="siren_dense_0")(h)
        h = jnp.sin(self.w0 * h)

        # hidden layers
        for i in range(1, self.num_layers):
            h = nn.Dense(
                self.hidden_dim,
                kernel_init=self._hidden_kernel_init(self.w0_hidden),
                name=f"siren_dense_{i}",
            )(h)
            h = jnp.sin(self.w0_hidden * h)

        # output heads: initialize similarly to hidden layers (paper suggests small init)
        outputs: dict[str, jnp.ndarray] = {}
        for name, dim in sorted(self.output_heads.items()):
            outputs[name] = nn.Dense(
                dim,
                kernel_init=self._hidden_kernel_init(self.w0_hidden),
                name=name,
            )(h)

        if was_unbatched:
            return {name: value[0] for name, value in outputs.items()}
        return outputs
