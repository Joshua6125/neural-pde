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
        self.grid_size = config.grid_size
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

    def _segmented_1d_rule(self, a: float, b: float) -> tuple[jnp.ndarray, jnp.ndarray]:
        if self.grid_size < 1:
            raise ValueError("grid_size must be >= 1")

        # Gauss-Legendre nodes/weights on [-1, 1]
        p, w = leggauss(self.degree)
        p = jnp.asarray(p)
        w = jnp.asarray(w)

        edges = jnp.linspace(a, b, self.grid_size + 1)

        all_points = []
        all_weights = []

        for left, right in zip(edges[:-1], edges[1:]):
            center = (left + right) / 2.0
            half_width = (right - left) / 2.0

            # Map [-1, 1] -> [left, right]
            seg_points = half_width * p + center
            seg_weights = half_width * w

            all_points.append(seg_points)
            all_weights.append(seg_weights)

        return jnp.concatenate(all_points), jnp.concatenate(all_weights)

    def _setup_quadrature_grids(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Generate tensor-product Gauss quadrature on the full domain, but with
        each axis split into `grid_size` segments.
        """
        axis_points = []
        axis_weights = []

        for d in range(self.dim):
            pts_d, wts_d = self._segmented_1d_rule(
                float(self.domain_min[d]),
                float(self.domain_max[d]),
            )
            axis_points.append(pts_d)
            axis_weights.append(wts_d)

        # Tensor-product grid over all axes
        points_mesh = jnp.meshgrid(*axis_points, indexing="ij")
        points = jnp.stack(points_mesh, axis=-1).reshape(-1, self.dim)

        weight_mesh = jnp.meshgrid(*axis_weights, indexing="ij")
        weights = jnp.prod(jnp.stack(weight_mesh, axis=-1), axis=-1).reshape(-1)

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
        """
        Generate quadrature points and weights on all boundary faces, using the
        same segmented Gauss rule on the free axes.
        """
        if self.dim == 1:
            # Boundary of an interval is two points.
            points = jnp.array([[self.domain_min[0]], [self.domain_max[0]]])
            normals = jnp.array([[-1.0], [1.0]])
            weights = jnp.ones(2)
            return {
                "points": points,
                "normals": normals,
                "weights": weights,
            }

        face_points = []
        face_normals = []
        face_weights = []

        for axis in range(self.dim):
            # Quadrature on all free axes, each split into grid_size segments
            free_axes = [d for d in range(self.dim) if d != axis]

            free_points_axes = []
            free_weights_axes = []
            for d in free_axes:
                pts_d, wts_d = self._segmented_1d_rule(
                    float(self.domain_min[d]),
                    float(self.domain_max[d]),
                )
                free_points_axes.append(pts_d)
                free_weights_axes.append(wts_d)

            # Tensor product over the free coordinates
            free_mesh = jnp.meshgrid(*free_points_axes, indexing="ij")
            free_points = jnp.stack(free_mesh, axis=-1).reshape(-1, self.dim - 1)

            weight_mesh = jnp.meshgrid(*free_weights_axes, indexing="ij")
            w = jnp.prod(jnp.stack(weight_mesh, axis=-1), axis=-1).reshape(-1)

            for is_max, boundary_value in [
                (False, self.domain_min[axis]),
                (True, self.domain_max[axis]),
            ]:
                pts = jnp.insert(free_points, axis, boundary_value, axis=1)

                normal = jnp.zeros(self.dim)
                normal = normal.at[axis].set(1.0 if is_max else -1.0)
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
