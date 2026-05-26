"""vPINN loss function."""

from typing import Callable

import jax
import jax.numpy as jnp

from ..base import Loss


class vPINNLoss(Loss):
    """vPINN loss projecting the residual onto multiple test functions.

    This implementation uses the Petrov-Galerkin integration trick: it evaluates
    the strong residual against a set of orthogonal test functions (Fourier basis),
    deferring the squaring operation until after the integration step.
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
        n_test_functions: int = 10,
    ):
        self.u_model = u_model
        self.c = c
        self.f = f
        self.u0 = u0
        self.ut0 = ut0
        self.ic_weight = ic_weight
        self.bc_weight = bc_weight
        self.n_test_functions = n_test_functions

        self._c_fn = c if callable(c) else self._constant_function(c)
        self._f_fn = f if callable(f) else self._constant_function(f)
        self._u0_fn = u0 if callable(u0) else self._constant_function(u0)
        self._ut0_fn = ut0 if callable(ut0) else self._constant_function(ut0)

        # Precompute frequencies for test functions
        self._freqs = jnp.arange(1, self.n_test_functions + 1) * jnp.pi

        self._vmapped_pde_residual = jax.vmap(self._pde_residual)
        self._vmapped_ic_residual = jax.vmap(self._ic_residual)
        self._vmapped_spatial_bc_residual = jax.vmap(self._spatial_bc_residual)

    def _u(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.u_model(x).squeeze()

    def _pde_residual(self, x: jnp.ndarray) -> jnp.ndarray:
        """Returns the strong PDE residual projected onto test functions.
        Output shape: (n_test_functions,)
        """
        H = jax.hessian(self._u)(x)
        u_tt = H[0, 0]
        laplacian_u = jnp.trace(H[1:, 1:])

        c = self._c_fn(x)
        f = self._f_fn(x)

        residual = u_tt - c**2 * laplacian_u - f

        # Evaluate test functions at x.
        # We use a simple Fourier basis on the sum of coordinates for demonstration.
        x_sum = jnp.sum(x)
        test_vals = jnp.sin(self._freqs * x_sum)

        # Do not square here! Return the integrand R(x) * v_k(x)
        return residual * test_vals

    def loss_interior(self, x_interior: jnp.ndarray) -> jnp.ndarray:
        """PDE residual multiplied by test functions at interior points.
        Output shape: (N_points, n_test_functions)
        """
        return self._vmapped_pde_residual(x_interior)

    def _ic_residual(self, x: jnp.ndarray) -> jnp.ndarray:
        u_val = self._u(x)
        ut_val = jax.grad(self._u)(x)[0]
        return self.ic_weight * ((u_val - self._u0_fn(x)) ** 2 + (ut_val - self._ut0_fn(x)) ** 2)

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
