"""Shared pytest fixtures and configuration for tests.

Provides:
    - Common Config fixtures for different integration methods
    - Test data fixtures
    - Utility functions for test validation
"""

import pytest
import jax.numpy as jnp
from src.config import Config


# ------ Config Fixtures for Quadrature ------


@pytest.fixture
def config_quadrature_1d():
    """1D Gauss-Legendre quadrature configuration."""
    return Config(
        dim=1,
        x_min=0.0,
        x_max=1.0,
        integration_method='quadrature',
        gauss_legendre_degree=20,
        adaptive_integration=False,
    )


@pytest.fixture
def config_quadrature_2d():
    """2D Gauss-Legendre quadrature configuration."""
    return Config(
        dim=2,
        x_min=0.0,
        x_max=1.0,
        integration_method='quadrature',
        gauss_legendre_degree=15,
        adaptive_integration=False,
    )


@pytest.fixture
def config_quadrature_3d():
    """3D Gauss-Legendre quadrature configuration."""
    return Config(
        dim=3,
        x_min=0.0,
        x_max=1.0,
        integration_method='quadrature',
        gauss_legendre_degree=8,
        adaptive_integration=False,
    )


# ------ Config Fixtures for Monte Carlo ------


@pytest.fixture
def config_monte_carlo_1d():
    """1D Monte Carlo configuration."""
    return Config(
        dim=1,
        x_min=0.0,
        x_max=1.0,
        integration_method='monte_carlo',
        monte_carlo_interior_samples=10000,
        monte_carlo_boundary_samples=1000,
        monte_carlo_seed=42,
    )


@pytest.fixture
def config_monte_carlo_2d():
    """2D Monte Carlo configuration."""
    return Config(
        dim=2,
        x_min=0.0,
        x_max=1.0,
        integration_method='monte_carlo',
        monte_carlo_interior_samples=50000,
        monte_carlo_boundary_samples=5000,
        monte_carlo_seed=42,
    )


@pytest.fixture
def config_monte_carlo_3d():
    """3D Monte Carlo configuration."""
    return Config(
        dim=3,
        x_min=0.0,
        x_max=1.0,
        integration_method='monte_carlo',
        monte_carlo_interior_samples=100000,
        monte_carlo_boundary_samples=10000,
        monte_carlo_seed=42,
    )


# ------ Custom Config Fixtures ------


@pytest.fixture
def config_quadrature_custom(request):
    """Create custom quadrature config via pytest parametrize."""
    params = getattr(request, 'param', {})
    return Config(
        dim=params.get('dim', 1),
        x_min=params.get('x_min', 0.0),
        x_max=params.get('x_max', 1.0),
        integration_method='quadrature',
        gauss_legendre_degree=params.get('degree', 20),
        adaptive_integration=params.get('adaptive', False),
    )


@pytest.fixture
def config_monte_carlo_custom(request):
    """Create custom Monte Carlo config via pytest parametrize."""
    params = getattr(request, 'param', {})
    return Config(
        dim=params.get('dim', 1),
        x_min=params.get('x_min', 0.0),
        x_max=params.get('x_max', 1.0),
        integration_method='monte_carlo',
        monte_carlo_interior_samples=params.get('interior_samples', 10000),
        monte_carlo_boundary_samples=params.get('boundary_samples', 1000),
        monte_carlo_seed=params.get('seed', 42),
    )


# ------ Test Function Fixtures (Known Integrals) ------


@pytest.fixture
def test_functions_1d():
    """Dictionary of 1D test functions with known analytical integrals over [0,1]."""
    return {
        'constant': {
            'func': lambda x: jnp.ones_like(x[:, 0]),
            'integral': 1.0,
            'tolerance': 1e-10,
            'description': 'f(x) = 1',
        },
        'linear': {
            'func': lambda x: x[:, 0],
            'integral': 0.5,
            'tolerance': 1e-10,
            'description': 'f(x) = x',
        },
        'quadratic': {
            'func': lambda x: x[:, 0] ** 2,
            'integral': 1.0 / 3.0,
            'tolerance': 1e-10,
            'description': 'f(x) = x^2',
        },
        'sine': {
            'func': lambda x: jnp.sin(jnp.pi * x[:, 0]),
            'integral': 2.0 / jnp.pi,
            'tolerance': 1e-8,
            'description': 'f(x) = sin(pi x)',
        },
        'exponential': {
            'func': lambda x: jnp.exp(x[:, 0]),
            'integral': jnp.e - 1.0,
            'tolerance': 1e-8,
            'description': 'f(x) = exp(x)',
        },
    }


@pytest.fixture
def test_functions_2d():
    """Dictionary of 2D test functions with known analytical integrals over [0,1]²."""
    return {
        'constant': {
            'func': lambda x: jnp.ones(x.shape[0]),
            'integral': 1.0,
            'tolerance': 1e-10,
            'description': 'f(x,y) = 1',
        },
        'separable': {
            'func': lambda x: x[:, 0] * x[:, 1],
            'integral': 0.25,
            'tolerance': 1e-10,
            'description': 'f(x,y) = xy',
        },
        'product_sine': {
            'func': lambda x: jnp.sin(jnp.pi * x[:, 0]) * jnp.sin(jnp.pi * x[:, 1]),
            'integral': (2.0 / jnp.pi) ** 2,
            'tolerance': 1e-8,
            'description': 'f(x,y) = sin(pi x)sin(pi y)',
        },
    }


# ------ Pytest Configuration ------


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration method test"
    )
    config.addinivalue_line(
        "markers",
        "quadrature: mark test as quadrature-specific"
    )
    config.addinivalue_line(
        "markers",
        "monte_carlo: mark test as Monte Carlo-specific"
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow (dimension scaling tests)"
    )
