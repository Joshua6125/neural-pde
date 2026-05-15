from typing import Mapping
from .mlp import MLP

import diffrax
import flax.linen as nn
import jax.numpy as jnp


class XNODEVectorField(nn.Module):
    """Vector field network N_vec for the XNODE model."""
    hidden_dim: int
    num_layers: int

    @nn.compact
    def __call__(self, t: float, h: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        # t is a scalar during Diffrax integration; expand to concatenate
        t_arr = jnp.atleast_1d(t)

        # The vector field explicitly depends on the spatial variable x [cite: 237]
        z = jnp.concatenate([h, t_arr, x], axis=-1)

        for _ in range(self.num_layers):
            z = jnp.tanh(nn.Dense(self.hidden_dim)(z))

        # The derivative dimension must match the hidden state dimension
        return nn.Dense(self.hidden_dim)(z)


class XNODE(nn.Module):
    """XNODE model mapping spatial coordinates to continuous temporal paths."""
    hidden_dim: int
    num_layers: int
    output_heads: Mapping[str, int]
    t_max: float = 1.0
    controller_rtol: float = 1e-6
    controller_atol: float = 1e-7
    solver: diffrax.AbstractSolver = diffrax.Tsit5()

    def setup(self):
        # Maps the PDE initial condition to the hidden state h(0)
        self.n_init = MLP(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            output_heads={'h0': self.hidden_dim}
        )

        # Computes dh/dt
        self.n_vec = XNODEVectorField(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        )

    @nn.compact
    def __call__(self, x: jnp.ndarray, u0_x: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """
        Parameters
        ----------
        x : jnp.ndarray
            Input: [t, x_1, x_2, ...]
        u0_x : jnp.ndarray
            The PDE's initial condition evaluated at x (referred to as h(x) in the paper).
        t_eval : float
            The target time to integrate the ODE toward.
        """
        t = x[0]
        xs = x[1:]
        h0 = self.n_init(u0_x)['h0']
        _ = self.n_vec(0.0, h0, xs)

        # Define the pure vector field function for Diffrax
        def vf(t, y, args):
            return self.n_vec(t, y, args)

        term = diffrax.ODETerm(vf)

        sol = diffrax.diffeqsolve(
            term,
            solver=self.solver,
            t0=0.0,
            t1=t,
            dt0=None,
            stepsize_controller=diffrax.PIDController(
                rtol=self.controller_rtol,
                atol=self.controller_atol
            ),
            y0=h0,
            args=xs,
            saveat=diffrax.SaveAt(t1=True),
            adjoint=diffrax.DirectAdjoint(),
        )

        assert sol.ys, "solution should exist"

        h_t = sol.ys[0]

        return {
            name: nn.Dense(dim, name=name)(h_t)
            for name, dim in self.output_heads.items()
        }
