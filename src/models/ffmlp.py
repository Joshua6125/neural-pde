from typing import Mapping

import flax.linen as nn
import jax
import jax.numpy as jnp


class FFMLP(nn.Module):
    hidden_dim: int
    num_layers: int
    output_heads: Mapping[str, int]

    # Fourier feature parameters
    num_fourier_features: int = 128
    fourier_scale: float = 10.0

    @nn.compact
    def __call__(self, x) -> dict[str, jnp.ndarray]:

        x = jnp.asarray(x)

        input_dim = x.shape[-1]

        B = self.variable(
            "constants",
            "fourier_matrix",
            lambda: self.fourier_scale
            * jax.random.normal(
                self.make_rng("params"),
                (input_dim, self.num_fourier_features),
            ),
        ).value

        proj = 2.0 * jnp.pi * (x @ B)

        h = jnp.concatenate(
            [jnp.sin(proj), jnp.cos(proj)],
            axis=-1,
        )

        for _ in range(self.num_layers):
            h = jnp.tanh(nn.Dense(self.hidden_dim)(h))

        return {
            name: nn.Dense(dim, name=name)(h)
            for name, dim in sorted(self.output_heads.items())
        }
