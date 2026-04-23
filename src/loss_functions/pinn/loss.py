"""PINN loss function."""

from typing import Callable

import jax
import jax.numpy as jnp

from ..base import Loss


class PINNLoss(Loss):
    """PINN loss for the wave equation u_tt - c^2 Delta u = f.

    Points x have shape [d+1] with x[0] = t (time) and x[1:] spatial.
    Assumes homogeneous Dirichlet BCs on the spatial boundary.

    Parameters
    ----------
    u_model : Callable
        Neural network u(x) -> scalar.
    c : float or Callable
        Wave speed c(x,t).
    f : float or Callable
        Forcing term f(x,t).
    u0 : float or Callable
        Initial displacement u(0,-) = u0.
    ut0 : float or Callable
        Initial velocity ∂_t u(0,-) = ut0.
    ic_weight : float
        Weight for initial condition loss.
    bc_weight : float
        Weight for boundary condition loss.
    """

    def __init__(
        self,
        u_model: Callable[[jnp.ndarray], jnp.ndarray],
        c: float | Callable[[jnp.ndarray], jnp.ndarray] = 1.0,
        f: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
        u0: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
        ut0: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
        ic_weight: float = 1.0,
        bc_weight: float = 1.0,
    ):
        self.u_model = u_model
        self.c = c
        self.f = f
        self.u0 = u0
        self.ut0 = ut0
        self.ic_weight = ic_weight
        self.bc_weight = bc_weight

        self._c_fn = c if callable(c) else self._constant_function(c)
        self._f_fn = f if callable(f) else self._constant_function(f)
        self._u0_fn = u0 if callable(u0) else self._constant_function(u0)
        self._ut0_fn = ut0 if callable(ut0) else self._constant_function(ut0)

        self._vmapped_pde_residual = jax.vmap(self._pde_residual)
        self._vmapped_ic_residual = jax.vmap(self._ic_residual)
        self._vmapped_spatial_bc_residual = jax.vmap(self._spatial_bc_residual)

    def _u(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.u_model(x).squeeze()

    def _pde_residual(self, x: jnp.ndarray) -> jnp.ndarray:
        """PDE residual (u_tt - c^2 Delta u - f)^2"""
        H = jax.hessian(self._u)(x)
        u_tt = H[0, 0]
        laplacian_u = jnp.trace(H[1:, 1:])

        c = self._c_fn(x)
        if jnp.ndim(c) != 0:
            raise ValueError("c should be scalar or return scalar type.")

        f = self._f_fn(x)
        if jnp.ndim(f) != 0:
            raise ValueError("f should be scalar or return scalar type.")

        return (u_tt - c**2 * laplacian_u - f) ** 2

    def loss_interior(self, x_interior: jnp.ndarray) -> jnp.ndarray:
        """PDE residual squared at interior points."""
        return self._vmapped_pde_residual(x_interior)

    def _ic_residual(self, x: jnp.ndarray) -> jnp.ndarray:
        """IC residuals at t=t_min."""
        u_val = self._u(x)
        ut_val = jax.grad(self._u)(x)[0]

        u0_val = self._u0_fn(x)
        if jnp.ndim(u0_val) != 0:
            raise ValueError("u0 should be scalar or return scalar type.")

        ut0_val = self._ut0_fn(x)
        if jnp.ndim(ut0_val) != 0:
            raise ValueError("ut0 should be scalar or return scalar type.")

        return self.ic_weight * ((u_val - u0_val) ** 2 + (ut_val - ut0_val) ** 2)

    def _spatial_bc_residual(self, x: jnp.ndarray) -> jnp.ndarray:
        """Homogeneous Dirichlet BC residual at spatial boundary."""
        return self.bc_weight * self._u(x) ** 2

    def loss_boundary(self, x_boundary: jnp.ndarray, normal_vector: jnp.ndarray) -> jnp.ndarray:
        """IC loss at t=t_min and Dirichlet BC loss on spatial boundaries."""
        is_ic = normal_vector[:, 0] < 0
        is_spatial_bc = normal_vector[:, 0] == 0

        return jnp.where(
            is_ic,
            self._vmapped_ic_residual(x_boundary),
            jnp.where(is_spatial_bc, self._vmapped_spatial_bc_residual(x_boundary), 0.0),
        )
