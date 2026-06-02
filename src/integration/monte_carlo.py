from typing import Callable, Any
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

        self.spatial_dim = config.spatial_dim
        self.dim = self.spatial_dim + 1
        self.interior_samples = config.interior_samples
        self.boundary_samples = config.boundary_samples
        self.t_min = config.t_min
        self.t_max = config.t_max
        self.x_min = config.x_min
        self.x_max = config.x_max

        self.domain_min = jnp.array([config.t_min] + [config.x_min] * config.spatial_dim)
        self.domain_max = jnp.array([config.t_max] + [config.x_max] * config.spatial_dim)

        spatial_volume = (self.x_max - self.x_min) ** self.spatial_dim
        self.volume = (self.t_max - self.t_min) * spatial_volume

        self.time_face_area = spatial_volume
        self.spatial_face_area = (self.t_max - self.t_min) * ((self.x_max - self.x_min) ** (self.spatial_dim - 1))

    def _sample_interior(self) -> jnp.ndarray:
        """Generate random samples uniformly in the domain interior."""
        self.key, subkey = jr.split(self.key)

        # Sample uniform in [0, 1)^dim
        samples = jr.uniform(subkey, shape=(self.interior_samples, self.dim))

        # Transform to [domain_min, domain_max]
        points = self.domain_min + samples * (self.domain_max - self.domain_min)
        return points

    def _setup_boundary_samples(self) -> dict:
        """Generate random samples on all boundary faces."""
        face_points = []
        face_normals = []
        face_weights = []

        for axis in range(self.dim):
            bound_min = self.domain_min[axis]
            bound_max = self.domain_max[axis]
            area = self.time_face_area if axis == 0 else self.spatial_face_area
            weight_per_sample = area / self.boundary_samples
            
            for is_max, boundary_value in [(False, bound_min), (True, bound_max)]:
                self.key, subkey = jr.split(self.key)

                # Sample random points on the (dim-1)-dimensional face
                samples = jr.uniform(subkey, shape=(self.boundary_samples, self.dim - 1))
                
                # Transform free dimensions
                free_min = jnp.concatenate([self.domain_min[:axis], self.domain_min[axis+1:]])
                free_max = jnp.concatenate([self.domain_max[:axis], self.domain_max[axis+1:]])
                samples = free_min + samples * (free_max - free_min)

                # Insert the fixed boundary coordinate at the correct axis
                pts = jnp.insert(samples, axis, boundary_value, axis=1)

                # Compute outward-pointing normal for this face
                normal = jnp.zeros(self.dim)
                normal = normal.at[axis].set(1.0 if is_max else -1.0)
                normals = jnp.tile(normal, (self.boundary_samples, 1))

                face_points.append(pts)
                face_normals.append(normals)
                face_weights.append(jnp.full(self.boundary_samples, weight_per_sample))

        return {
            "points": jnp.concatenate(face_points),
            "normals": jnp.concatenate(face_normals),
            "weights": jnp.concatenate(face_weights)
        }

    def integrate_interior(
            self,
            func: Callable[[jnp.ndarray], Any],
        ) -> Any:
        """Integrate over interior using Monte Carlo sampling."""

        points_interior = self._sample_interior()

        # Evaluate function at random samples
        func_values = func(points_interior)
        # Monte Carlo: volume * (1/n) * sum(f)
        factor = self.volume / self.interior_samples
        integral = jax.tree_util.tree_map(lambda x: factor * jnp.sum(x, axis=0), func_values)
        return integral

    def integrate_boundary(
            self,
            func: Callable[[jnp.ndarray, jnp.ndarray], Any]
        ) -> Any:
        """Integrate over boundary using Monte Carlo sampling."""
        boundary_data = self._setup_boundary_samples()

        func_values = func(
            boundary_data["points"],
            boundary_data["normals"]
        )
        weights = boundary_data["weights"]
        
        # Monte Carlo on boundary: sum(area_i / n_i * f)
        # We need to tensor multiply the weights correctly for possibly multi-dimensional output
        integral = jax.tree_util.tree_map(
            lambda x: jnp.tensordot(weights, x, axes=([0], [0])), 
            func_values
        )
        return integral

    def integrate(
            self,
            interior_func: Callable[[jnp.ndarray], Any],
            boundary_func: Callable[[jnp.ndarray, jnp.ndarray], Any],
            rng_key: jax.Array | None = jax.random.PRNGKey(42),
        ) -> tuple[Any, Any]:
        """Integrate with explicit RNG threading for reproducible resampling."""
        if rng_key is None:
            raise ValueError("rng_key may not be None in Monte Carlo Integration.")

        self.key = rng_key

        interior_loss = self.integrate_interior(interior_func)
        boundary_loss = self.integrate_boundary(boundary_func)

        return interior_loss, boundary_loss
