from typing import Callable

import jax
import jax.numpy as jnp

from .loss_base import LossBase


class LossLS(LossBase):
    """Least-squares loss for the first-order acoustic wave system.

    TODO: Should add references somehow
    Minimises the functional G as defined in FGK23

    Points x have shape [d+1] with x[0] = t (time) and x[1:] spatial.

    Parameters
    ----------
    v_model : Callable
        Network v(x) -> scalar. For JAX/Flax, pass ``partial(model.apply, params)``.
    sigma_model : Callable
        Network sigma(x) -> [d]. Must output a vector of spatial dimension d.
    f : Callable or None
        Source term f(x,t) for the v equation. Defaults to zero.
    g : Callable or None
        Source term g(x,t) for the sigma equation. Defaults to zero.
    v0 : Callable or None
        Initial condition v(0,-) = v_theta. Defaults to zero.
    sigma0 : Callable or None
        Initial condition sigma(0,-) = sigma_theta. Defaults to zero.
    """

    def __init__(
        self,
        v_model: Callable[[jnp.ndarray], jnp.ndarray],
        sigma_model: Callable[[jnp.ndarray], jnp.ndarray],
        f: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
        g: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
        v0: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
        sigma0: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    ):
        self.v_model = v_model
        self.sigma_model = sigma_model
        self.f = f
        self.g = g
        self.v0 = v0
        self.sigma0 = sigma0

    def _v(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.v_model(x).squeeze()

    def _sigma(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.sigma_model(x).reshape(-1)

    def _interior_residual(self, x: jnp.ndarray) -> jnp.ndarray:
        """Sum of squared residuals of both equations"""
        v_grad = jax.grad(self._v)(x)
        dt_v = v_grad[0]
        grad_v = v_grad[1:]

        J_sigma = jax.jacobian(self._sigma)(x)
        dt_sigma = J_sigma[:, 0]
        div_sigma = jnp.trace(J_sigma[:, 1:])

        if self.f is not None:
            f = self.f(x)
        else:
            f = 0.0

        if self.g is not None:
            g = self.g(x)
        else:
            g = jnp.zeros_like(grad_v)

        res_v = dt_v - div_sigma - f
        res_sigma = dt_sigma - grad_v - g

        return res_v ** 2 + jnp.sum(res_sigma ** 2)

    def loss_interior(self, x_interior: jnp.ndarray) -> jnp.ndarray:
        """Interior residuals"""
        return jax.vmap(self._interior_residual)(x_interior)

    def _ic_residual(self, x: jnp.ndarray) -> jnp.ndarray:
        """IC residuals at a single point at t=t_min."""
        v_val = self._v(x)
        sigma_val = self._sigma(x)

        if self.v0 is not None:
            v0_val = self.v0(x)
        else:
            v0_val = 0.0

        if self.sigma0 is not None:
            sigma0_val = self.sigma0(x)
        else:
            sigma0_val = jnp.zeros_like(sigma_val)

        return (v_val - v0_val) ** 2 + jnp.sum((sigma_val - sigma0_val) ** 2)

    def loss_boundary(self, x_boundary: jnp.ndarray, normal_vector: jnp.ndarray) -> jnp.ndarray:
        """IC loss at t=t_min. Other faces contribute zero."""
        is_ic = normal_vector[:, 0] < 0
        return jnp.where(is_ic, jax.vmap(self._ic_residual)(x_boundary), 0.0)
