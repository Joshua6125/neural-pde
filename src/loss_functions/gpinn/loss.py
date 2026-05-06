"""gPINN loss function (gradient-enhanced PINN)."""

from typing import Callable

import jax
import jax.numpy as jnp

from ..base import Loss


class gPINNLoss(Loss):
    """gPINN loss for the wave equation with gradient-enhanced penalties.

    Interior loss per point is:
        R(x)^2 + residual_grad_weight * ||∇R(x)||^2
        + solution_grad_weight * ||∇u(x)||^2
    where R(x) = u_tt - c^2 Δ u - f.
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
        residual_grad_weight: float = 0.0,
        solution_grad_weight: float = 0.0,
    ):
        self.u_model = u_model
        self.c = c
        self.f = f
        self.u0 = u0
        self.ut0 = ut0
        self.ic_weight = ic_weight
        self.bc_weight = bc_weight
        self.residual_grad_weight = residual_grad_weight
        self.solution_grad_weight = solution_grad_weight

        self._c_fn = c if callable(c) else self._constant_function(c)
        self._f_fn = f if callable(f) else self._constant_function(f)
        self._u0_fn = u0 if callable(u0) else self._constant_function(u0)
        self._ut0_fn = ut0 if callable(ut0) else self._constant_function(ut0)

        self._vmapped_pde_residual = jax.vmap(self._pde_residual)
        self._vmapped_ic_residual = jax.vmap(self._ic_residual)
        self._vmapped_spatial_bc_residual = jax.vmap(self._spatial_bc_residual)

    def _u(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.u_model(x).squeeze()

    def _residual_scalar(self, x: jnp.ndarray) -> jnp.ndarray:
        """Return PDE residual R(x) = u_tt - c^2 Δu - f (not squared)."""
        H = jax.hessian(self._u)(x)
        u_tt = H[0, 0]
        laplacian_u = jnp.trace(H[1:, 1:])

        c = self._c_fn(x)
        if jnp.ndim(c) != 0:
            raise ValueError("c should be scalar or return scalar type.")

        f = self._f_fn(x)
        if jnp.ndim(f) != 0:
            raise ValueError("f should be scalar or return scalar type.")

        return u_tt - c**2 * laplacian_u - f

    def _pde_residual(self, x: jnp.ndarray) -> jnp.ndarray:
        """Interior loss combining residual and gradient penalties."""
        R = self._residual_scalar(x)
        res_sq = R ** 2

        # Gradient of residual (w.r.t. t and spatial coords). Prefer spatial-only
        # penalty to reduce cost — take full gradient but sum spatial components.
        grad_R = jax.grad(self._residual_scalar)(x)
        # drop time-component (index 0) and take spatial part
        grad_R_spatial = grad_R[1:]
        grad_R_norm_sq = jnp.sum(grad_R_spatial ** 2)

        # Gradient of solution u: penalise spatial gradient magnitude if requested
        grad_u = jax.grad(self._u)(x)
        grad_u_spatial = grad_u[1:]
        grad_u_norm_sq = jnp.sum(grad_u_spatial ** 2)

        return (
            res_sq
            + self.residual_grad_weight * grad_R_norm_sq
            + self.solution_grad_weight * grad_u_norm_sq
        )

    def loss_interior(self, x_interior: jnp.ndarray) -> jnp.ndarray:
        return self._vmapped_pde_residual(x_interior)

    def _ic_residual(self, x: jnp.ndarray) -> jnp.ndarray:
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
        return self.bc_weight * self._u(x) ** 2

    def loss_boundary(self, x_boundary: jnp.ndarray, normal_vector: jnp.ndarray) -> jnp.ndarray:
        is_ic = normal_vector[:, 0] < 0
        is_spatial_bc = normal_vector[:, 0] == 0

        return jnp.where(
            is_ic,
            self._vmapped_ic_residual(x_boundary),
            jnp.where(is_spatial_bc, self._vmapped_spatial_bc_residual(x_boundary), 0.0),
        )
