"""Pytest fixtures for SLS module tests.

Provides mock models, callable fixtures, and test data points.
"""

import pytest
import jax
import jax.numpy as jnp

from typing import Any

from src.models import NeuralNetModelConfig


# ============================================================================
# MOCK NEURAL NETWORK MODELS
# ============================================================================

class MockSLSModelValid:
    """Mock SLS model with a valid output contract."""

    def init(self, rng_key: jax.Array, x: jnp.ndarray) -> Any:
        return {"dummy": 0}

    def apply(self, params: Any, x: jnp.ndarray) -> dict[str, jnp.ndarray]:
        spatial_dim = max(int(x.shape[0]) - 1, 1)
        return {
            "v": jnp.asarray(x[0]),
            "sigma": x[1:] / spatial_dim,
        }


class MockSLSModelVectorV:
    """Mock SLS model with vector-shaped v output of length 1."""

    def init(self, rng_key: jax.Array, x: jnp.ndarray) -> Any:
        return {"dummy": 0}

    def apply(self, params: Any, x: jnp.ndarray) -> dict[str, jnp.ndarray]:
        spatial_dim = max(int(x.shape[0]) - 1, 1)
        return {
            "v": jnp.asarray([x[0]]),
            "sigma": x[1:] / spatial_dim,
        }


class MockSLSModelNotDict:
    """Mock SLS model returning a non-dict output."""

    def init(self, rng_key: jax.Array, x: jnp.ndarray) -> Any:
        return {"dummy": 0}

    def apply(self, params, x):
        return jnp.asarray([x[0], x[0]])


class MockSLSModelMissingV:
    """Mock SLS model missing the v output."""

    def init(self, rng_key: jax.Array, x: jnp.ndarray) -> Any:
        return {"dummy": 0}

    def apply(self, params: Any, x: jnp.ndarray) -> dict[str, jnp.ndarray]:
        spatial_dim = max(int(x.shape[0]) - 1, 1)
        return {"sigma": x[1:] / spatial_dim}


class MockSLSModelMissingSigma:
    """Mock SLS model missing the sigma output."""

    def init(self, rng_key: jax.Array, x: jnp.ndarray) -> Any:
        return {"dummy": 0}

    def apply(self, params: Any, x: jnp.ndarray) -> dict[str, jnp.ndarray]:
        return {"v": jnp.asarray(x[0])}


class MockSLSModelBadVShape:
    """Mock SLS model with invalid v output shape."""

    def init(self, rng_key: jax.Array, x: jnp.ndarray) -> Any:
        return {"dummy": 0}

    def apply(self, params: Any, x: jnp.ndarray) -> dict[str, jnp.ndarray]:
        spatial_dim = max(int(x.shape[0]) - 1, 1)
        return {
            "v": jnp.asarray([x[0], x[0]]),
            "sigma": x[1:] / spatial_dim,
        }


class MockSLSModelBadSigmaShape:
    """Mock SLS model with invalid sigma output shape for 2D input."""

    def init(self, rng_key: jax.Array, x: jnp.ndarray) -> Any:
        return {"dummy": 0}

    def apply(self, params: Any, x: jnp.ndarray) -> dict[str, jnp.ndarray]:
        return {
            "v": jnp.asarray(x[0]),
            "sigma": jnp.asarray([x[1], x[1]]),
        }


@pytest.fixture
def mock_model_valid():
    """Mock SLS model with valid scalar v and vector sigma outputs."""
    return MockSLSModelValid()


@pytest.fixture
def mock_model_vector_v():
    """Mock SLS model with v output shaped as length-1 vector."""
    return MockSLSModelVectorV()


@pytest.fixture
def mock_model_not_dict():
    """Mock SLS model returning a non-dict output."""
    return MockSLSModelNotDict()


@pytest.fixture
def mock_model_missing_v():
    """Mock SLS model missing the v key."""
    return MockSLSModelMissingV()


@pytest.fixture
def mock_model_missing_sigma():
    """Mock SLS model missing the sigma key."""
    return MockSLSModelMissingSigma()


@pytest.fixture
def mock_model_bad_v_shape():
    """Mock SLS model with invalid v output shape."""
    return MockSLSModelBadVShape()


@pytest.fixture
def mock_model_bad_sigma_shape():
    """Mock SLS model with invalid sigma output shape for 2D input."""
    return MockSLSModelBadSigmaShape()


# ============================================================================
# CALLABLE FIXTURES (f, g, v0, sigma0, v_boundary)
# ============================================================================

@pytest.fixture
def callable_f_zero():
    """Source term for v equation: f(x) = 0."""
    return lambda x: 0.0


@pytest.fixture
def callable_f_constant():
    """Source term for v equation: f(x) = 1."""
    return lambda x: 1.0


@pytest.fixture
def callable_g_zero():
    """Source term for sigma equation: g(x) = 0."""
    return lambda x: jnp.zeros_like(x[1:])


@pytest.fixture
def callable_g_constant():
    """Source term for sigma equation: g(x) = ones."""
    return lambda x: jnp.ones_like(x[1:])


@pytest.fixture
def callable_v0_zero():
    """Initial velocity field: v0(x) = 0."""
    return lambda x: 0.0


@pytest.fixture
def callable_v0_linear():
    """Initial velocity field: v0(x) = x0."""
    return lambda x: jnp.asarray(x[1])


@pytest.fixture
def callable_sigma0_zero():
    """Initial sigma field: sigma0(x) = 0."""
    return lambda x: jnp.zeros_like(x[1:])


@pytest.fixture
def callable_sigma0_linear():
    """Initial sigma field matching the valid mock model."""
    return lambda x: x[1:] / max(int(x.shape[0]) - 1, 1)


@pytest.fixture
def callable_v_boundary_zero():
    """Boundary velocity field: v_boundary(x) = 0."""
    return lambda x: 0.0


@pytest.fixture
def callable_v_boundary_linear():
    """Boundary velocity field: v_boundary(x) = x[0]."""
    return lambda x: jnp.asarray(x[0])


# ============================================================================
# TEST DATA POINTS
# ============================================================================

@pytest.fixture
def sample_input_2d():
    """2D total input sample: [t, x]."""
    return jnp.array([0.5, 0.25])


@pytest.fixture
def sample_input_3d():
    """3D total input sample: [t, x, y]."""
    return jnp.array([0.5, 0.25, 0.75])


@pytest.fixture
def interior_points_2d():
    """Interior test points for 1 spatial dimension."""
    return jnp.array([
        [0.2, 0.1],
        [0.5, 0.3],
        [0.7, 0.8],
    ])


@pytest.fixture
def interior_points_3d():
    """Interior test points for 2 spatial dimensions."""
    return jnp.array([
        [0.2, 0.1, 0.3],
        [0.5, 0.4, 0.6],
        [0.7, 0.8, 0.2],
    ])


@pytest.fixture
def boundary_ic_points_2d():
    """Boundary points marked as initial condition points."""
    points = jnp.array([
        [0.0, 0.25],
        [0.0, 0.75],
    ])
    normals = jnp.array([
        [-1.0, 0.0],
        [-1.0, 0.0],
    ])
    return points, normals


@pytest.fixture
def boundary_bc_points_2d():
    """Boundary points marked as spatial boundary points."""
    points = jnp.array([
        [0.2, 0.0],
        [0.7, 1.0],
    ])
    normals = jnp.array([
        [0.0, 1.0],
        [0.0, -1.0],
    ])
    return points, normals


@pytest.fixture
def boundary_exterior_points_2d():
    """Boundary points that should be ignored by loss_boundary."""
    points = jnp.array([
        [0.5, 0.5],
    ])
    normals = jnp.array([
        [1.0, 0.0],
    ])
    return points, normals


# ============================================================================
# CONFIG FIXTURES
# ============================================================================

@pytest.fixture
def sls_config_default():
    """Default SLSConfig instance."""
    from src.loss_functions import SLSConfig

    return SLSConfig()


@pytest.fixture
def sls_config_with_sources(callable_f_zero, callable_g_zero):
    """SLSConfig with zero source terms."""
    from src.loss_functions import SLSConfig

    return SLSConfig(f=callable_f_zero, g=callable_g_zero)


@pytest.fixture
def sls_config_with_boundary(callable_v_boundary_linear):
    """SLSConfig with a boundary velocity condition."""
    from src.loss_functions import SLSConfig

    return SLSConfig(v_boundary=callable_v_boundary_linear)


@pytest.fixture
def neuralnet_model_config():
    """Reusable NeuralNetModelConfig for SLSConfig tests."""
    return NeuralNetModelConfig(
        hidden_dim=8,
        num_layers=2,
        output_heads={"v": 1, "sigma": 1},
    )


@pytest.fixture
def rng_key():
    """Deterministic PRNG key for SLS tests."""
    return jax.random.PRNGKey(42)
