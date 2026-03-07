from .base import NDCubeIntegration
from .quadrature import QuadratureIntegration
from .monte_carlo import MonteCarloIntegration

def get_integrator(method: str, domain, **kwargs) -> NDCubeIntegration:
    """Factory function for choosing integration method."""
    if method == "quadrature":
        return QuadratureIntegration(domain, **kwargs)
    elif method == "monte_carlo":
        return MonteCarloIntegration(domain, **kwargs)
    else:
        raise ValueError(f"Unknown integration method: {method}")

__all__ = ["NDCubeIntegration", "QuadratureIntegration", "MonteCarloIntegration", "get_integrator"]