from dataclasses import dataclass
from typing import Callable, Literal
import jax.numpy as jnp

@dataclass
class Config:
    """Configuration for integration and PDE solving.

    Attributes
    ----------
    dim : int
        Spatial dimension of the domain.
    x_min, x_max : float
        Domain bounds in each dimension.
    integration_method : str
        Integration method: 'quadrature' or 'monte_carlo'.
    gauss_legendre_degree : int
        Degree of Gauss-Legendre quadrature (points = degree^dim).
    monte_carlo_interior_samples : int
        Number of samples for Monte Carlo interior integration.
    monte_carlo_boundary_samples : int
        Number of samples for Monte Carlo boundary integration.
    monte_carlo_seed : int
        Random seed for reproducibility.
    adaptive_integration : bool
        Enable adaptive refinement (not yet implemented).
    """
    dim: int = 2
    x_min: float = 0.0
    x_max: float = 1.0
    integration_method: Literal["quadrature", "monte_carlo"] = "quadrature"
    gauss_legendre_degree: int = 100
    monte_carlo_interior_samples: int = 10000
    monte_carlo_boundary_samples: int = 1000
    monte_carlo_seed: int = 42
    adaptive_integration: bool = False


@dataclass(frozen=True)
class PINNLossConfig:
    kind: Literal["pinn"] = "pinn"
    c: float | Callable[[jnp.ndarray], jnp.ndarray] = 1.0
    f: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    u0: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    ut0: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    ic_weight: float = 1.0
    bc_weight: float = 1.0


@dataclass(frozen=True)
class LSLossConfig:
    kind: Literal["ls"] = "ls"
    f: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    g: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    v0: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    sigma0: Callable[[jnp.ndarray], jnp.ndarray] | None = None
