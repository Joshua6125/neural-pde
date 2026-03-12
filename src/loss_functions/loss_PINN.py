from typing import Callable

import jax
import jax.numpy as jnp

from .loss_base import LossBase


class PINNLoss(LossBase):
    """PINN loss for the wave equation u_tt - c^2 Delta u = f on Q = J x Omega.

    Points x have shape [d+1] with x[0] = t (time) and x[1:] spatial.
    Assumes homogeneous Dirichlet BCs on the spatial boundary.

    Parameters
    ----------
    u_model : Callable
        Neural network u(x) -> scalar. For JAX/Flax, pass ``partial(model.apply, params)``.
    c : float or Callable
        Wave speed c(x,t), either a constant or a point-wise callable.
    f : Callable or None
        Forcing term f(x,t). Defaults to zero.
    u0 : Callable or None
        Initial displacement u(0,-) = u0. Defaults to zero.
    ut0 : Callable or None
        Initial velocity delta_t u(0,-) = ut0. Defaults to zero.
    ic_weight : float
        Weighting factor for the initial condition terms.
    bc_weight : float
        Weighting factor for the spatial Dirichlet BC terms.
    """

    def __init__(
        self,
        u_model: Callable[[jnp.ndarray], jnp.ndarray],
        c: float | Callable[[jnp.ndarray], jnp.ndarray],
        f: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
        u0: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
        ut0: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
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

    def _u(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.u_model(x).squeeze()

    def _pde_residual(self, x: jnp.ndarray) -> jnp.ndarray:
        """Pointwise PDE residual (u_tt - c^2 Delta u - f)^2 at a single point."""
        H = jax.hessian(self._u)(x)
        u_tt = H[0, 0]
        laplacian_u = jnp.trace(H[1:, 1:])
        c = self.c(x) if callable(self.c) else self.c
        f = self.f(x) if self.f is not None else 0.0
        return (u_tt - c**2 * laplacian_u - f) ** 2

    def loss_interior(self, x_interior: jnp.ndarray) -> jnp.ndarray:
        """PDE residual squared at interior space-time points."""
        return jax.vmap(self._pde_residual)(x_interior)

    def _ic_residual(self, x: jnp.ndarray) -> jnp.ndarray:
        """IC residuals (displacement + velocity) at a single point at t=t_min."""
        u_val = self._u(x)
        ut_val = jax.grad(self._u)(x)[0]
        u0_val = self.u0(x) if self.u0 is not None else 0.0
        ut0_val = self.ut0(x) if self.ut0 is not None else 0.0
        return self.ic_weight * ((u_val - u0_val) ** 2 + (ut_val - ut0_val) ** 2)

    def _spatial_bc_residual(self, x: jnp.ndarray) -> jnp.ndarray:
        """Homogeneous Dirichlet BC residual at a single spatial boundary point."""
        return self.bc_weight * self._u(x) ** 2

    def loss_boundary(self, x_boundary: jnp.ndarray, normal_vector: jnp.ndarray) -> jnp.ndarray:
        """IC loss at t=t_min and Dirichlet BC loss on spatial boundaries."""
        is_ic = normal_vector[:, 0] < 0
        is_spatial_bc = normal_vector[:, 0] == 0

        ic_loss = jnp.where(is_ic, jax.vmap(self._ic_residual)(x_boundary), 0.0)
        bc_loss = jnp.where(is_spatial_bc, jax.vmap(self._spatial_bc_residual)(x_boundary), 0.0)
        return ic_loss + bc_loss
    