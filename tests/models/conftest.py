"""Shared fixtures for model-interface tests."""

import pytest
import jax
import jax.numpy as jnp


@pytest.fixture
def rng_key():
    """Deterministic key used for init/apply tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def sample_1d_input():
    """Batched 1D input points with shape (batch, dim=1)."""
    return jnp.array([[0.1], [0.5], [0.7], [0.9]])


@pytest.fixture
def sample_2d_input():
    """Batched 2D input points with shape (batch, dim=2)."""
    return jnp.array([[0.1, 0.2], [0.5, 0.5], [0.7, 0.3], [0.9, 0.1]])


@pytest.fixture
def default_single_head():
    """Single-head output contract used by most tests."""
    return {"output": 1}


@pytest.fixture
def multi_output_heads():
    """Multi-head output contract used for interface validation."""
    return {"u": 1, "p": 2, "sigma": 3}
