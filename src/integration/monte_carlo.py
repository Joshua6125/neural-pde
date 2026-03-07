from typing import Callable
import jax.numpy as jnp
from .base import NDCubeIntegration

class MonteCarloIntegration(NDCubeIntegration):
    def __init__(self, domain, num_samples=10000, adaptive=False):
        pass

    def integrate_interior(self, func: Callable[[jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
        pass

    def integrate_boundary(self, func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
        pass
