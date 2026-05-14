"""Deep Ritz Method loss function."""

from typing import Callable

import jax
import jax.numpy as jnp

from ..base import Loss


class DRMLoss(Loss):
    """Deep Ritz Method loss for energy minimisation."""

    def __init__(
        self,
        u_model: Callable[[jnp.ndarray], jnp.ndarray],
        A: float | jnp.ndarray | Callable[[jnp.ndarray], jnp.ndarray] = 1.0,
        c: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
        f: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
        g: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
        boundary_weight: float = 1.0,
    ):
        self.u_model = u_model
        self.A = A
        self.c = c
        self.f = f
        self.g = g
        self.boundary_weight = boundary_weight

        self._A_fn = A if callable(A) else self._constant_function(A)
        self._c_fn = c if callable(c) else self._constant_function(c)
        self._f_fn = f if callable(f) else self._constant_function(f)
        self._g_fn = g if callable(g) else self._constant_function(g)

        self._vmapped_interior = jax.vmap(self._interior_energy)
        self._vmapped_boundary = jax.vmap(self._boundary_penalty)

    def _u(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.u_model(x).squeeze()

    def _coerce_matrix(self, value: jnp.ndarray, dimension: int) -> jnp.ndarray:
        matrix = jnp.asarray(value)

        if matrix.ndim == 0:
            return matrix * jnp.eye(dimension, dtype=matrix.dtype)

        if matrix.ndim == 1:
            if not matrix.shape[0] == dimension:
                raise ValueError("A vector-valued A must match the input dimension.")
            return jnp.diag(matrix)

        if not matrix.shape == (dimension, dimension):
            raise ValueError("A should be scalar, vector, or square matrix valued.")

        return matrix

    def _interior_energy(self, x: jnp.ndarray) -> jnp.ndarray:
        u_val = self._u(x)
        grad_u = jax.grad(self._u)(x)

        A_val = self._coerce_matrix(self._A_fn(x), int(x.shape[0]))

        c_val = self._c_fn(x)
        if not jnp.ndim(c_val) == 0:
            raise ValueError("c should be scalar or return scalar type.")

        f_val = self._f_fn(x)
        if not jnp.ndim(f_val) == 0:
            raise ValueError("f should be scalar or return scalar type.")

        quadratic_term = grad_u @ A_val @ grad_u
        potential_term = c_val * (u_val ** 2)
        source_term = f_val * u_val

        return 0.5 * (quadratic_term + potential_term) - source_term

    def loss_interior(self, x_interior: jnp.ndarray) -> jnp.ndarray:
        """Energy density at interior points."""
        return self._vmapped_interior(x_interior)

    def _boundary_penalty(self, x: jnp.ndarray) -> jnp.ndarray:
        u_val = self._u(x)

        g_val = self._g_fn(x)
        if jnp.ndim(g_val) != 0:
            raise ValueError("g should be scalar or return scalar type.")

        return self.boundary_weight * (u_val - g_val) ** 2

    def loss_boundary(self, x_boundary: jnp.ndarray, normal_vector: jnp.ndarray) -> jnp.ndarray:
        """Boundary penalty density."""
        _ = normal_vector
        return self._vmapped_boundary(x_boundary)
