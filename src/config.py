from dataclasses import dataclass

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
        Integration method: 'gauss_legendre' or 'monte_carlo'.
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
    integration_method: str = 'gauss_legendre'
    gauss_legendre_degree: int = 100
    monte_carlo_interior_samples: int = 10000
    monte_carlo_boundary_samples: int = 1000
    monte_carlo_seed: int = 42
    adaptive_integration: bool = False
