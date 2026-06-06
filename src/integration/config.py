from dataclasses import dataclass
from typing import Literal, TypeAlias


@dataclass(frozen=True)
class IntegrationConfigBase:
    """Shared integration-domain configuration."""
    integration_method: str = "base"
    t_min: float = 0.0
    t_max: float = 1.0
    spatial_dim: int = 1
    x_min: float = 0.0
    x_max: float = 1.0

    def validate_domain(self) -> None:
        assert self.spatial_dim > 0, "dim must be strictly positive"
        assert self.x_min < self.x_max, "x_min must be < x_max"


@dataclass(frozen=True)
class QuadratureConfig(IntegrationConfigBase):
    """Configuration for Gauss-Legendre quadrature integration."""

    integration_method: Literal["quadrature"] = "quadrature"
    degree: int = 5
    grid_size: int = 1000
    adaptive_integration: bool = False

    def validate(self) -> None:
        self.validate_domain()
        assert self.degree > 0, "degree must be strictly positive"


@dataclass(frozen=True)
class MonteCarloConfig(IntegrationConfigBase):
    """Configuration for Monte Carlo integration."""
    integration_method: Literal["monte_carlo"] = "monte_carlo"
    interior_samples: int = 10000
    boundary_samples: int = 1000

    def validate(self) -> None:
        self.validate_domain()
        assert self.interior_samples > 0, "interior_samples must be strictly positive"
        assert self.boundary_samples > 0, "boundary_samples must be strictly positive"


AnyIntegrationConfig: TypeAlias = QuadratureConfig | MonteCarloConfig