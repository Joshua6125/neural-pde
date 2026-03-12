import flax.linen as nn
import jax.numpy as jnp

class NeuralNet(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def setup(self):
        self.layers = [nn.Dense(self.hidden_dim) for _ in range(self.num_layers)]
        self.output_layer = nn.Dense(2)

    def __call__(self, x):
        h = x
        for layer in self.layers:
            h = jnp.tanh(layer(h))
        return self.output_layer(h)
