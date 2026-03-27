from .base import NDCubeIntegration
from .quadrature import QuadratureIntegration
from .monte_carlo import MonteCarloIntegration
from .config import IntegrationConfig, MonteCarloConfig, QuadratureConfig

def get_integrator(config: IntegrationConfig) -> NDCubeIntegration:
    """Factory function for choosing integration method.

    Parameters
    ----------
    config : IntegrationConfig
        Typed integration configuration (quadrature or monte carlo).

    Returns
    -------
    NDCubeIntegration
        Integrator instance (QuadratureIntegration or MonteCarloIntegration).
    """
    if isinstance(config, QuadratureConfig):
        return QuadratureIntegration(config)
    if isinstance(config, MonteCarloConfig):
        return MonteCarloIntegration(config)

    raise ValueError("Unknown integration config type.")

__all__ = [
    "IntegrationConfig",
    "QuadratureConfig",
    "MonteCarloConfig",
    "NDCubeIntegration",
    "QuadratureIntegration",
    "MonteCarloIntegration",
    "get_integrator",
]
