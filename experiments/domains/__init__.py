"""
Domain plugins for experiments.

Each domain plugin implements problem-specific functionality:
- Analytical solutions
- Test data generation
- Domain-specific visualisation

Available domains can be discovered via registry.
"""

from .base import DomainPlugin
from .simple_wave_equation import SimpleWaveEquationDomain

__all__ = ["DomainPlugin", "WaveEquationDomain"]

# Registry: maps domain names to classes
DOMAIN_REGISTRY = {
    "wave_equation": SimpleWaveEquationDomain,
}


def get_domain(name: str) -> type[DomainPlugin]:
    """Get a domain plugin class by name.

    Args:
        name: Domain name (e.g., "wave_equation")

    Returns:
        Domain plugin class

    Raises:
        ValueError: If domain name not found
    """
    if name not in DOMAIN_REGISTRY:
        available = ", ".join(DOMAIN_REGISTRY.keys())
        raise ValueError(
            f"Unknown domain '{name}'. Available: {available}"
        )
    return DOMAIN_REGISTRY[name]


def list_domains() -> list[str]:
    """List all available domain names."""
    return list(DOMAIN_REGISTRY.keys())


def register_domain(name: str, plugin_class: type[DomainPlugin]) -> None:
    """Register a new domain plugin.

    Args:
        name: Domain name for lookup
        plugin_class: Class implementing DomainPlugin interface
    """
    # TODO: Should have a look at this pylance warning
    DOMAIN_REGISTRY[name] = plugin_class # type: ignore
