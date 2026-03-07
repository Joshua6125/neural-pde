from typing import Callable
from .base import NDCubeIntegration
from ..config import Config

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

    def __init__(
        self,
        config: Config
    ):
        assert config.dim > 0, "dim must be positive"
        assert config.gauss_legendre_degree > 0, "degree must be positive"
        assert config.x_min < config.x_max, "x_min must be < x_max"

        self.degree = config.gauss_legendre_degree
        self.adaptive = config.adaptive_integration
        if self.adaptive:
            # TODO: Implement adaptive quadrature
            print("Warning: Adaptive quadrature not implemented yet. Ignoring adaptive flag.", file=sys.stderr)

        self.dim = config.dim
        if self.dim > 3:
            print(f"Warning: {self.dim}-dimensional quadrature with degree {self.degree} "
                  f"creates {self.degree**self.dim} points.", file=sys.stderr)

        self.x_min = config.x_min
        self.x_max = config.x_max

        # Set up sampling grids for interior and boundary
        self.points_interior, self.weights_interior = self._setup_quadrature_grids()
        self.boundary_faces = self._setup_boundary_grids()

    def _setup_quadrature_grids(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        # Generate gauss-legendre sample points on [-1, 1]
        lg_samp = leggauss(self.degree)
        p = jnp.array(lg_samp[0])
        w = jnp.array(lg_samp[1])

        # Transform sample interval
        center = (self.x_max + self.x_min) / 2.0
        half_width = (self.x_max - self.x_min) / 2.0
        p_transformed = half_width * p + center

        # Transform weight accordingly as well
        w_transformed = w * half_width

        # Create sample mesh
        points_mesh_axis = [p_transformed] * self.dim
        points_mesh = jnp.meshgrid(*points_mesh_axis, indexing='ij')
        points = jnp.stack(points_mesh, axis=-1).reshape(-1, self.dim)

        # Create weight mesh
        weight_mesh_axis = [w_transformed] * self.dim
        weight_mesh = jnp.meshgrid(*weight_mesh_axis, indexing='ij')
        weights = jnp.stack(weight_mesh, axis=-1)
        weights = jnp.prod(weights, axis=-1).reshape(-1)

        return points, weights

    @staticmethod
    @jax.jit
    def _integrate_interior(func, points, weights):
        # Evaluate function at quadrature points
        func_values = func(points)

        # Compute weighted sum
        integral = jnp.sum(weights * func_values)
        return integral

    def integrate_interior(
        self,
        func: Callable[[jnp.ndarray], jnp.ndarray]
    ) -> jnp.ndarray:
        """Integrate over interior using quadrature rule."""
        return self._integrate_interior(func, self.points_interior, self.weights_interior)

    def _generate_face_quadrature(self, dim_axis: int, boundary_value: float) -> jnp.ndarray:
        """Generate quadrature points on a specific boundary face."""
        # Create a grid for the (dim-1) dimensions that are not fixed
        free_axes = [i for i in range(self.dim) if i != dim_axis]
        free_points_mesh_axis = [self.points_interior[:, i] for i in free_axes]
        free_points_mesh = jnp.meshgrid(*free_points_mesh_axis)
        free_points = jnp.stack(free_points_mesh, axis=-1).reshape(-1, self.dim - 1)

        # Insert the fixed boundary value into the correct position
        face_points = jnp.insert(free_points, dim_axis, boundary_value, axis=-1)
        return face_points

    def _compute_face_normals(self, dim_axis: int, boundary_value: float) -> jnp.ndarray:
        """Compute normal vectors for a specific boundary face."""
        normal_vector = jnp.zeros(self.dim)
        normal_vector = normal_vector.at[dim_axis].set(1.0 if boundary_value == self.x_max else -1.0)
        return jnp.tile(normal_vector, (self.degree ** (self.dim - 1), 1))

    def _compute_face_area(self) -> float:
        """Compute the area of a boundary face."""
        return (self.x_max - self.x_min) ** (self.dim - 1)

    def _setup_boundary_grids(self) -> dict[str, jnp.ndarray]:
        """Generate quadrature points and weights on all boundary faces."""

        # 1D Gauss nodes on [-1,1]
        p_1d, w_1d = leggauss(self.degree)
        p_1d = jnp.array(p_1d)
        w_1d = jnp.array(w_1d)

        center = (self.x_max + self.x_min) / 2.0
        half_width = (self.x_max - self.x_min) / 2.0

        p_1d = half_width * p_1d + center
        w_1d = half_width * w_1d

        face_points = []
        face_normals = []
        face_weights = []

        for axis in range(self.dim):
            for boundary_value in [self.x_min, self.x_max]:

                # tensor grid for the free coordinates
                mesh_axes = [p_1d] * (self.dim - 1)
                mesh = jnp.meshgrid(*mesh_axes, indexing="ij")

                free_points = jnp.stack(mesh, axis=-1).reshape(-1, self.dim - 1)

                # insert fixed coordinate
                pts = jnp.insert(free_points, axis, boundary_value, axis=1)

                # compute weights for tensor grid
                weight_axes = [w_1d] * (self.dim - 1)
                weight_mesh = jnp.meshgrid(*weight_axes, indexing="ij")

                w = jnp.stack(weight_mesh, axis=-1)
                w = jnp.prod(w, axis=-1).reshape(-1)

                # outward normal
                normal = jnp.zeros(self.dim)
                normal = normal.at[axis].set(
                    1.0 if boundary_value == self.x_max else -1.0
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

    # TODO: Should consider passing along a traced params object to avoid
    #       re-tracing the function for each face.
    @staticmethod
    @jax.jit
    def _integrate_boundary(func, points, normals, weights):
        func_values = func(points, normals)
        return jnp.sum(weights * func_values)

    def integrate_boundary(
        self,
        func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    ) -> jnp.ndarray:
        """Integrate function over the cube boundary."""
        return self._integrate_boundary(
            func,
            self.boundary_faces["points"],
            self.boundary_faces["normals"],
            self.boundary_faces["weights"]
        )
