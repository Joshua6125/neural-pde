"""Least-Squares loss function."""

from typing import Callable

import jax
import jax.numpy as jnp

from ..base import Loss


class SLSLoss(Loss):
    """Least-squares loss for the first-order acoustic wave system.

    Points x have shape [d+1] with x[0] = t (time) and x[1:] spatial.

    Parameters
    ----------
    v_model : Callable
        Network v(x) -> scalar.
    sigma_model : Callable
        Network sigma(x) -> [d].
    f : Callable or None
        Source term for v equation.
    g : Callable or None
        Source term for sigma equation.
    v0 : Callable or None
        Initial condition v(0,-) = v0.
    sigma0 : Callable or None
        Initial condition sigma(0,-) = sigma0.
    """

    def __init__(
        self,
        v_model: Callable[[jnp.ndarray], jnp.ndarray],
        sigma_model: Callable[[jnp.ndarray], jnp.ndarray],
        f: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
        g: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
        v0: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
        sigma0: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
        v_boundary: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
    ):
        self.v_model = v_model
        self.sigma_model = sigma_model
        self.f = f
        self.g = g
        self.v0 = v0
        self.sigma0 = sigma0
        self.v_boundary = v_boundary
        if not self.v_boundary == 0.0:
            print("WARNING: SLS formulation is only proven for Dirichlet boundary conditions")

        self._f_fn = f if callable(f) else self._constant_function(f)
        self._g_fn = g if callable(g) else self._constant_function(g)
        self._v0_fn = v0 if callable(v0) else self._constant_function(v0)
        self._sigma0_fn = sigma0 if callable(sigma0) else self._constant_function(sigma0)
        self._v_boundary_fn = (
            v_boundary if callable(v_boundary) else self._constant_function(v_boundary)
        ) if v_boundary is not None else None

        self._vmapped_interior_residual = jax.vmap(self._interior_residual)
        self._vmapped_ic_residual = jax.vmap(self._ic_residual)
        self._vmapped_spatial_bc_residual = jax.vmap(self._spatial_bc_residual)

    def _v(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.v_model(x).squeeze()

    def _sigma(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.sigma_model(x).reshape(-1)

    def _interior_residual(self, x: jnp.ndarray) -> jnp.ndarray:
        """Sum of squared residuals of both equations."""
        v_grad = jax.grad(self._v)(x)
        dt_v = v_grad[0]
        grad_v = v_grad[1:]

        J_sigma = jax.jacobian(self._sigma)(x)
        dt_sigma = J_sigma[:, 0]
        div_sigma = jnp.trace(J_sigma[:, 1:])

        f = self._f_fn(x)
        if jnp.ndim(f) != 0:
            raise ValueError("f should be scalar or return scalar type.")

        g = self._g_fn(x)
        if jnp.ndim(g) == 0:
            g = g * jnp.ones_like(grad_v)
        if not jnp.shape(g) == jnp.shape(grad_v):
            raise ValueError("g should be or return the right shape.")

        res_v = dt_v - div_sigma - f
        res_sigma = dt_sigma - grad_v - g

        return res_v ** 2 + jnp.sum(res_sigma ** 2)

    def loss_interior(self, x_interior: jnp.ndarray) -> jnp.ndarray:
        """Interior residuals."""
        return self._vmapped_interior_residual(x_interior)

    def _ic_residual(self, x: jnp.ndarray) -> jnp.ndarray:
        """IC residuals at t=t_min."""
        v_val = self._v(x)
        sigma_val = self._sigma(x)

        v0_val = self._v0_fn(x)
        if jnp.ndim(v0_val) != 0:
            raise ValueError("v0 should be scalar or return scalar type.")

        sigma0_val = self._sigma0_fn(x)
        if jnp.ndim(sigma0_val) == 0:
            sigma0_val = sigma0_val * jnp.ones_like(sigma_val)
        if not jnp.shape(sigma0_val) == jnp.shape(sigma_val):
            raise ValueError("sigma0 should be or return the right shape.")

        return (v_val - v0_val) ** 2 + jnp.sum((sigma_val - sigma0_val) ** 2)

    def _spatial_bc_residual(self, x: jnp.ndarray) -> jnp.ndarray:
        """Spatial boundary residual for the velocity field."""
        if self.v_boundary is None:
            return jnp.zeros(())

        v_val = self._v(x)
        v_boundary_val = self._v_boundary_fn(x) if self._v_boundary_fn is not None else 0.0
        if jnp.ndim(v_boundary_val) != 0:
            raise ValueError("v_boundary should be scalar or return scalar type.")

        return (v_val - v_boundary_val) ** 2

    def loss_boundary(self, x_boundary: jnp.ndarray, normal_vector: jnp.ndarray) -> jnp.ndarray:
        """IC loss at t=t_min and optional spatial Dirichlet loss for v."""
        is_ic = normal_vector[:, 0] < 0
        is_spatial_bc = normal_vector[:, 0] == 0

        return jnp.where(
            is_ic,
            self._vmapped_ic_residual(x_boundary),
            jnp.where(is_spatial_bc, self._vmapped_spatial_bc_residual(x_boundary), 0.0),
        )
