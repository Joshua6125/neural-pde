from typing import Callable, Any
from .base import NDCubeIntegration
from .config import QuadratureConfig

from numpy.polynomial.legendre import leggauss

import sys
import jax
import jax.numpy as jnp

class QuadratureIntegration(NDCubeIntegration):
    '''
    Gauss-Legendre quadrature for n-D cube integration.

    Complexity: O(degree^dim) points and evaluations.
    Warning: Exponential scaling in dimension - use <= 3D or low degree.

    Boundary: Assumes Dirichlet BC (u=0 on boundary).
    '''

    def __init__(self, config: QuadratureConfig):
        config.validate()

        self.degree = config.degree
        self.adaptive = config.adaptive_integration
        if self.adaptive:
            # TODO: Implement adaptive quadrature
            print("Warning: Adaptive quadrature not implemented yet. Ignoring adaptive flag.", file=sys.stderr)

        self.spatial_dim = config.spatial_dim
        self.dim = self.spatial_dim + 1
        if self.dim > 3:
            print(f"Warning: {self.dim}-dimensional quadrature with degree {self.degree} "
                  f"creates {self.degree**self.dim} points.", file=sys.stderr)

        self.t_min = config.t_min
        self.t_max = config.t_max
        self.x_min = config.x_min
        self.x_max = config.x_max

        # Time and space domains explicitly separated
        self.domain_min = jnp.array([config.t_min] + [config.x_min] * config.spatial_dim)
        self.domain_max = jnp.array([config.t_max] + [config.x_max] * config.spatial_dim)

        # Set up sampling grids for interior and boundary
        self.points_interior, self.weights_interior = self._setup_quadrature_grids()
        self.boundary_faces = self._setup_boundary_grids()

    def _setup_quadrature_grids(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        # Generate gauss-legendre sample points on [-1, 1]
        lg_samp = leggauss(self.degree)
        p = jnp.array(lg_samp[0])
        w = jnp.array(lg_samp[1])

        # Transform for each axis based on its dedicated min/max bounds
        points_mesh_axis = []
        weight_mesh_axis = []
        for d in range(self.dim):
            center = (self.domain_max[d] + self.domain_min[d]) / 2.0
            half_width = (self.domain_max[d] - self.domain_min[d]) / 2.0
            
            points_mesh_axis.append(half_width * p + center)
            weight_mesh_axis.append(w * half_width)

        # Create sample mesh
        points_mesh = jnp.meshgrid(*points_mesh_axis, indexing='ij')
        points = jnp.stack(points_mesh, axis=-1).reshape(-1, self.dim)

        # Create weight mesh
        weight_mesh = jnp.meshgrid(*weight_mesh_axis, indexing='ij')
        weights = jnp.stack(weight_mesh, axis=-1)
        weights = jnp.prod(weights, axis=-1).reshape(-1)

        return points, weights

    def integrate_interior(
            self,
            func: Callable[[jnp.ndarray], Any]
        ) -> Any:
        """Integrate over interior using quadrature rule."""
        # Evaluate function at quadrature points
        func_values = func(self.points_interior)

        # Compute weighted sum (quadrature formula)
        # func_values can be a pytree of arrays. We multiply by weights alongside axis 0
        # using tensordot to support multi-dimensional outputs gracefully.
        integral = jax.tree_util.tree_map(
            lambda x: jnp.tensordot(self.weights_interior, x, axes=([0], [0])),
            func_values
        )
        return integral

    def _setup_boundary_grids(self) -> dict[str, jnp.ndarray]:
        """Generate quadrature points and weights on all boundary faces."""

        if self.dim == 1:
            points = jnp.array([[self.domain_min[0]], [self.domain_max[0]]])
            normals = jnp.array([[-1.0], [ 1.0]])
            weights = jnp.ones(2)
            return {
                "points": points,
                "normals": normals,
                "weights": weights,
            }

        # 1D Gauss nodes on [-1,1]
        p_1d, w_1d = leggauss(self.degree)
        p_1d = jnp.array(p_1d)
        w_1d = jnp.array(w_1d)

        # Scale nodes for each specific dimension
        p_1d_scaled = []
        w_1d_scaled = []
        for d in range(self.dim):
            center = (self.domain_max[d] + self.domain_min[d]) / 2.0
            half_width = (self.domain_max[d] - self.domain_min[d]) / 2.0
            p_1d_scaled.append(half_width * p_1d + center)
            w_1d_scaled.append(half_width * w_1d)

        face_points = []
        face_normals = []
        face_weights = []

        for axis in range(self.dim):
            for is_max, boundary_value in [(False, self.domain_min[axis]), (True, self.domain_max[axis])]:

                # Select transformed 1D points/weights for all free axes
                free_p_axes = [p_1d_scaled[d] for d in range(self.dim) if d != axis]
                free_w_axes = [w_1d_scaled[d] for d in range(self.dim) if d != axis]

                # tensor grid for the free coordinates
                mesh = jnp.meshgrid(*free_p_axes, indexing="ij")
                free_points = jnp.stack(mesh, axis=-1).reshape(-1, self.dim - 1)

                # insert fixed coordinate
                pts = jnp.insert(free_points, axis, boundary_value, axis=1)

                # compute weights for tensor grid
                weight_mesh = jnp.meshgrid(*free_w_axes, indexing="ij")
                w = jnp.stack(weight_mesh, axis=-1)
                w = jnp.prod(w, axis=-1).reshape(-1)

                # outward normal
                normal = jnp.zeros(self.dim)
                normal = normal.at[axis].set(
                    1.0 if is_max else -1.0
                )
                normals = jnp.tile(normal, (pts.shape[0], 1))

                face_points.append(pts)
                face_normals.append(normals)
                face_weights.append(w)

        return {
            "points": jnp.concatenate(face_points),
            "normals": jnp.concatenate(face_normals),
            "weights": jnp.concatenate(face_weights),
        }

    def integrate_boundary(
            self,
            func: Callable[[jnp.ndarray, jnp.ndarray], Any]
        ) -> Any:
        """Integrate function over the cube boundary."""
        func_values = func(
            self.boundary_faces["points"],
            self.boundary_faces["normals"]
        )

        integral = jax.tree_util.tree_map(
            lambda x: jnp.tensordot(self.boundary_faces["weights"], x, axes=([0], [0])),
            func_values
        )
        return integral
