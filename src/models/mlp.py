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

    @nn.compact
    def __call__(self, x) -> dict[str, jnp.ndarray]:
        h = x
        for _ in range(self.num_layers):
            h = jnp.tanh(nn.Dense(self.hidden_dim)(h))

        return {
            name: nn.Dense(dim, name=name)(h)
            for name, dim in self.output_heads.items()
        }
