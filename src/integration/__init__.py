from .base import NDCubeIntegration
from .quadrature import QuadratureIntegration
from .monte_carlo import MonteCarloIntegration
from ..config import Config

def get_integrator(config: Config) -> NDCubeIntegration:
    """Factory function for choosing integration method.

    Parameters
    ----------
    config : Config
        Configuration object specifying integration method and parameters.

    Returns
    -------
    NDCubeIntegration
        Integrator instance (QuadratureIntegration or MonteCarloIntegration).
    """
    method = config.integration_method.lower()

    if method == 'quadrature':
        return QuadratureIntegration(config)
    elif method == 'monte_carlo':
        return MonteCarloIntegration(config)
    else:
        raise ValueError(
            f"Unknown integration method: '{config.integration_method}'. "
            f"Must be 'quadrature' or 'monte_carlo'."
        )

__all__ = ["NDCubeIntegration", "QuadratureIntegration", "MonteCarloIntegration", "get_integrator"]
