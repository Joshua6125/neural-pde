import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Mapping
import diffrax

class LatentField(nn.Module):
    hidden_dim: int
    num_layers: int

    @nn.compact
    def __call__(self, t: float, z: jnp.ndarray, args) -> jnp.ndarray:
        h = z
        # Inject t into the hidden state
        t_arr = jnp.array([t])
        h = jnp.concatenate([h, t_arr])

        for _ in range(self.num_layers):
            h = jnp.tanh(nn.Dense(self.hidden_dim)(h))

        # Output must be the same shape as z
        dz_dt = nn.Dense(z.shape[-1])(h)
        return dz_dt

class Decoder(nn.Module):
    output_heads: Mapping[str, int]
    hidden_dim: int
    num_layers: int

    @nn.compact
    def __call__(self, z_t: jnp.ndarray, x_space: jnp.ndarray) -> dict[str, jnp.ndarray]:
        # Concatenate latent state and spatial coordinates

        assert x_space.ndim == 1
        assert z_t.ndim == 1

        h = jnp.concatenate([z_t, x_space], axis=-1)

        for _ in range(self.num_layers):
            h = jnp.tanh(nn.Dense(self.hidden_dim)(h))

        return {
            name: nn.Dense(dim, name=name)(h)
            for name, dim in sorted(self.output_heads.items())
        }

class GlobalNODE(nn.Module):
    output_heads: Mapping[str, int]
    z_dim: int
    latent_hidden_dim: int
    latent_num_layers: int
    decoder_hidden_dim: int
    decoder_num_layers: int
    T: float = 1.0  # Max time
    dt: float = 0.01

    # @nn.compact
    # def __call__(self, x: jnp.ndarray) -> dict[str, jnp.ndarray]:
    #     t = x[0]
    #     x_space = x[1:]

    #     # Learnable initial state for the global latent trajectory
    #     z0 = self.param('z0', nn.initializers.zeros, (self.z_dim,))

    #     # Vector field
    #     vector_field = LatentField(
    #         hidden_dim=self.latent_hidden_dim,
    #         num_layers=self.latent_num_layers
    #     )

    #     # Initialize variables by executing it once outside the solver loop
    #     _ = vector_field(0.0, z0, None)

    #     def f(t, y, args):
    #         return vector_field(t, y, args)

    #     term = diffrax.ODETerm(f)

    #     # Solver and step size controller
    #     solver = diffrax.Heun()
    #     stepsize_controller = diffrax.ConstantStepSize()

    #     # Solve ODE globally over [0, T]
    #     sol = diffrax.diffeqsolve(
    #         term,
    #         solver,
    #         t0=0.0,
    #         t1=self.T,
    #         dt0=self.dt,
    #         y0=z0,
    #         stepsize_controller=stepsize_controller,
    #         saveat=diffrax.SaveAt(dense=True),
    #         adjoint=diffrax.RecursiveCheckpointAdjoint()
    #     )

    #     # Evaluate the dense interpolation at the queried time t
    #     z_t = sol.evaluate(t)

    #     jax.debug.print("x shape = {}", x.shape)
    #     jax.debug.print("t shape = {}", jnp.shape(t))
    #     jax.debug.print("x_space shape = {}", x_space.shape)
    #     jax.debug.print("z_t shape = {}", z_t.shape)

    #     # Decode spatially
    #     decoder = Decoder(
    #         output_heads=self.output_heads,
    #         hidden_dim=self.decoder_hidden_dim,
    #         num_layers=self.decoder_num_layers
    #     )
    #     return decoder(z_t, x_space)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> dict[str, jnp.ndarray]:
        t = x[0]
        x_space = x[1:]
        z0 = self.param("z0", nn.initializers.zeros, (self.z_dim,))

        vector_field = LatentField(
            hidden_dim=self.latent_hidden_dim,
            num_layers=self.latent_num_layers
        )

        decoder = Decoder(
            output_heads=self.output_heads,
            hidden_dim=self.decoder_hidden_dim,
            num_layers=self.decoder_num_layers
        )

        def f(t, y, args):
            return vector_field(t, y, args)

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(f),
            solver=diffrax.Heun(),
            t0=0.0,
            t1=self.T,
            dt0=self.dt,
            y0=z0,
            stepsize_controller=diffrax.ConstantStepSize(),
            saveat=diffrax.SaveAt(dense=True),
            adjoint=diffrax.RecursiveCheckpointAdjoint(),
        )

        z_t = sol.evaluate(t)

        jax.debug.print("x shape = {}", x.shape)
        jax.debug.print("t shape = {}", jnp.shape(t))
        jax.debug.print("x_space shape = {}", x_space.shape)
        jax.debug.print("z_t shape = {}", z_t.shape)
        
        return decoder(z_t, x_space)
