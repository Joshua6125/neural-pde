"""Least-squares loss function for the first-order system."""

from typing import Callable

import jax
import jax.numpy as jnp

from ..base import Loss


class FOSLSLoss(Loss):
    """First-order system least-squares loss.

    The interior functional is

        ||∂_t v - div σ - f||_2^2 + ||∂_t σ - ∇v - g||_2^2.

    Boundary contributions are intentionally omitted here; this class
    represents the precursor interior functional used by SLS.
    """

    def __init__(
        self,
        v_model: Callable[[jnp.ndarray], jnp.ndarray],
        sigma_model: Callable[[jnp.ndarray], jnp.ndarray],
        f: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
        g: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
    ):
        self.v_model = v_model
        self.sigma_model = sigma_model
        self._f_fn = f if callable(f) else self._constant_function(f)
        self._g_fn = g if callable(g) else self._constant_function(g)

        self._vmapped_interior_residual = jax.vmap(self._interior_residual)

    def _v(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(self.v_model(x)).squeeze()

    def _sigma(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(self.sigma_model(x)).reshape(-1)

    @staticmethod
    def _as_vector(value: jnp.ndarray, reference: jnp.ndarray, name: str) -> jnp.ndarray:
        if jnp.ndim(value) == 0:
            return value * jnp.ones_like(reference)
        if jnp.shape(value) != jnp.shape(reference):
            raise ValueError(f"{name} should be scalar or return the right shape.")
        return value

    def _interior_residual(self, x: jnp.ndarray) -> jnp.ndarray:
        """Pointwise squared FOSLS residual."""
        v_grad = jax.grad(self._v)(x)
        dt_v = v_grad[0]
        grad_v = v_grad[1:]

        J_sigma = jax.jacobian(self._sigma)(x)
        dt_sigma = J_sigma[:, 0]
        div_sigma = jnp.trace(J_sigma[:, 1:])

        f = self._f_fn(x)
        if jnp.ndim(f) != 0:
            raise ValueError("f should be scalar or return scalar type.")

        g = self._as_vector(self._g_fn(x), grad_v, "g")

        res_v = dt_v - div_sigma - f
        res_sigma = dt_sigma - grad_v - g
        return jnp.square(res_v) + jnp.sum(jnp.square(res_sigma))

    def loss_interior(self, x_interior: jnp.ndarray) -> jnp.ndarray:
        return self._vmapped_interior_residual(x_interior)

    def loss_boundary(self, x_boundary: jnp.ndarray, normal_vector: jnp.ndarray) -> jnp.ndarray:
        del normal_vector
        return jnp.zeros(x_boundary.shape[0], dtype=x_boundary.dtype)
