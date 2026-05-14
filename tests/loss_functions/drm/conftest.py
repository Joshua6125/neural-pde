"""Pytest fixtures for DRM module tests."""

from typing import Any

import jax
import jax.numpy as jnp
import pytest


class MockDRMModelValid:
    """Mock DRM model with a valid scalar output contract."""

    def init(self, rng_key: jax.Array, x: jnp.ndarray) -> Any:
        return {"dummy": 0}

    def apply(self, params: Any, x: jnp.ndarray) -> dict[str, jnp.ndarray]:
        return {"u": jnp.asarray(x[0] + 2.0 * x[1])}


class MockDRMModelNotDict:
    """Mock DRM model returning a non-dict output."""

    def init(self, rng_key: jax.Array, x: jnp.ndarray) -> Any:
        return {"dummy": 0}

    def apply(self, params: Any, x: jnp.ndarray):
        return jnp.asarray([x[0]])


class MockDRMModelMissingU:
    """Mock DRM model missing the u output."""

    def init(self, rng_key: jax.Array, x: jnp.ndarray) -> Any:
        return {"dummy": 0}

    def apply(self, params: Any, x: jnp.ndarray) -> dict[str, jnp.ndarray]:
        return {"v": jnp.asarray(x[0])}


class MockDRMModelBadShape:
    """Mock DRM model with invalid u output shape."""

    def init(self, rng_key: jax.Array, x: jnp.ndarray) -> Any:
        return {"dummy": 0}

    def apply(self, params: Any, x: jnp.ndarray) -> dict[str, jnp.ndarray]:
        return {"u": jnp.asarray([x[0], x[1]])}


@pytest.fixture
def mock_model_valid():
    return MockDRMModelValid()


@pytest.fixture
def mock_model_not_dict():
    return MockDRMModelNotDict()


@pytest.fixture
def mock_model_missing_u():
    return MockDRMModelMissingU()


@pytest.fixture
def mock_model_bad_shape():
    return MockDRMModelBadShape()


@pytest.fixture
def sample_input_2d():
    return jnp.array([0.5, 0.25])


@pytest.fixture
def interior_points_2d():
    return jnp.array([
        [0.2, 0.1],
        [0.5, 0.3],
        [0.7, 0.8],
    ])


@pytest.fixture
def boundary_points_2d():
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
def callable_A_identity():
    return lambda x: 1.0


@pytest.fixture
def callable_A_matrix():
    return lambda x: jnp.eye(x.shape[0])


@pytest.fixture
def callable_g_zero():
    return lambda x: 0.0


@pytest.fixture
def callable_f_zero():
    return lambda x: 0.0
