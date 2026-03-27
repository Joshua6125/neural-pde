from dataclasses import dataclass
from typing import Literal, TypeAlias


@dataclass(frozen=True)
class IntegrationConfigBase:
    """Shared integration-domain configuration."""

    dim: int = 2
    x_min: float = 0.0
    x_max: float = 1.0

    def validate_domain(self) -> None:
        assert self.dim > 0, "dim must be strictly positive"
        assert self.x_min < self.x_max, "x_min must be < x_max"


@dataclass(frozen=True)
class QuadratureConfig(IntegrationConfigBase):
    """Configuration for Gauss-Legendre quadrature integration."""

    integration_method: Literal["quadrature"] = "quadrature"
    gauss_legendre_degree: int = 100
    adaptive_integration: bool = False

    def validate(self) -> None:
        self.validate_domain()
        assert self.gauss_legendre_degree > 0, "degree must be strictly positive"


@dataclass(frozen=True)
class MonteCarloConfig(IntegrationConfigBase):
    """Configuration for Monte Carlo integration."""

    integration_method: Literal["monte_carlo"] = "monte_carlo"
    monte_carlo_interior_samples: int = 10000
    monte_carlo_boundary_samples: int = 1000
    monte_carlo_seed: int = 42

    def validate(self) -> None:
        self.validate_domain()
        assert self.monte_carlo_interior_samples > 0, "interior_samples must be strictly positive"
        assert self.monte_carlo_boundary_samples > 0, "boundary_samples must be strictly positive"


IntegrationConfig: TypeAlias = QuadratureConfig | MonteCarloConfig