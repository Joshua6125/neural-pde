from typing import Mapping

import flax.linen as nn
import jax.numpy as jnp


class MLP(nn.Module):
    """Simple fully-connected network used across methods.

    Parameters
    ----------
    hidden_dim : int
        Width of each hidden layer.
    num_layers : int
        Number of hidden layers.
    output_heads : Mapping[str, int]
        Named output heads.
    """

    hidden_dim: int
    num_layers: int
    output_heads: Mapping[str, int]
    constrained_heads: list[str]

    @nn.compact
    def __call__(self, x) -> dict[str, jnp.ndarray]:
        h = x
        for _ in range(self.num_layers):
            h = jnp.tanh(nn.Dense(self.hidden_dim)(h))

        output = {
            name: nn.Dense(dim, name=name)(h)
            for name, dim in sorted(self.output_heads.items())
        }

        for head in output.keys():
            if head in self.constrained_heads:
                p = 2.0
                eps = 1e-12
                spatial_coords = x[..., 1:]

                a_left = jnp.clip(spatial_coords, eps, 1.0)
                a_right = jnp.clip(1.0 - spatial_coords, eps, 1.0)

                boundary_func = jnp.sum(a_left ** (-p) + a_right ** (-p), axis=-1, keepdims=True) ** (-1.0 / p)

                output[head] = boundary_func * output[head]

        return output
