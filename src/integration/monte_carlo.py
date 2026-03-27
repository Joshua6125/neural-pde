from typing import Callable
from .base import NDCubeIntegration
from .config import MonteCarloConfig

import jax
import jax.numpy as jnp
import jax.random as jr


class MonteCarloIntegration(NDCubeIntegration):
    '''
    Monte Carlo integration for n-D cube integration.

    Complexity: O(samples) evaluations.

    Boundary: Assumes Dirichlet BC (u=0 on boundary).
    '''

    def __init__(self, config: MonteCarloConfig):
        config.validate()

        self.dim = config.dim
        self.interior_samples = config.monte_carlo_interior_samples
        self.boundary_samples = config.monte_carlo_boundary_samples
        self.x_min = config.x_min
        self.x_max = config.x_max
        self.seed = config.monte_carlo_seed

        if self.seed:
            self._fallback_key = jr.PRNGKey(self.seed)

        self.volume = (self.x_max - self.x_min) ** self.dim
        self.face_area = (self.x_max - self.x_min) ** (self.dim - 1)

    def _sample_interior(self, key: jax.Array) -> tuple[jnp.ndarray, jax.Array]:
        """Generate random samples uniformly in the domain interior."""
        key, subkey = jr.split(key)

        # Sample uniform in [0, 1)^dim
        samples = jr.uniform(subkey, shape=(self.interior_samples, self.dim))

        # Transform to [x_min, x_max]^dim
        points = self.x_min + samples * (self.x_max - self.x_min)
        return points, key

    def _setup_boundary_samples(self, key: jax.Array) -> tuple[dict, jax.Array]:
        """Generate random samples on all boundary faces."""
        face_points = []
        face_normals = []

        for axis in range(self.dim):
            for boundary_value in [self.x_min, self.x_max]:
                key, subkey = jr.split(key)

                # Sample random points on the (dim-1)-dimensional face
                # We need dim-1 free dimensions
                samples = jr.uniform(subkey, shape=(self.boundary_samples, self.dim - 1))
                samples = self.x_min + samples * (self.x_max - self.x_min)

                # Insert the fixed boundary coordinate at the correct axis
                pts = jnp.insert(samples, axis, boundary_value, axis=1)

                # Compute outward-pointing normal for this face
                normal = jnp.zeros(self.dim)
                normal = normal.at[axis].set(1.0 if boundary_value == self.x_max else -1.0)
                normals = jnp.tile(normal, (self.boundary_samples, 1))

                face_points.append(pts)
                face_normals.append(normals)

        return {
            "points": jnp.concatenate(face_points),
            "normals": jnp.concatenate(face_normals),
        }, key

    def integrate_interior(
            self,
            func: Callable[[jnp.ndarray], jnp.ndarray]
        ) -> jnp.ndarray:
        """Integrate over interior using Monte Carlo sampling."""
        points_interior, key = self._sample_interior(self._fallback_key)
        self._fallback_key = key

        # Evaluate function at random samples
        func_values = func(points_interior)
        # Monte Carlo: volume * (1/n) * sum(f)
        integral = (self.volume / self.interior_samples) * jnp.sum(func_values)
        return integral

    def integrate_boundary(
            self, func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
        ) -> jnp.ndarray:
        """Integrate over boundary using Monte Carlo sampling."""
        boundary_data, key = self._setup_boundary_samples(self._fallback_key)
        self._fallback_key = key

        func_values = func(
            boundary_data["points"],
            boundary_data["normals"]
        )
        # Monte Carlo on boundary: area * (1/n) * sum(f)
        integral = (self.face_area / self.boundary_samples) * jnp.sum(func_values)
        return integral

    def integrate_with_key(
            self,
            interior_func: Callable[[jnp.ndarray], jnp.ndarray],
            boundary_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
            rng_key: jax.Array,
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jax.Array]:
        """Integrate with explicit RNG threading for reproducible resampling."""
        points_interior, next_key = self._sample_interior(rng_key)
        boundary_data, next_key = self._setup_boundary_samples(next_key)

        interior_values = interior_func(points_interior)
        boundary_values = boundary_func(boundary_data["points"], boundary_data["normals"])

        interior_loss = (self.volume / self.interior_samples) * jnp.sum(interior_values)
        boundary_loss = (self.face_area / self.boundary_samples) * jnp.sum(boundary_values)
        total_loss = interior_loss + boundary_loss
        return total_loss, interior_loss, boundary_loss, next_key
