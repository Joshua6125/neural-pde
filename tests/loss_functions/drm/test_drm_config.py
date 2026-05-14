"""Tests for DRMConfig."""

import jax.numpy as jnp
import pytest

from src.loss_functions import DRMConfig
from src.models import AnyModelConfig


pytestmark = pytest.mark.DRM


class TestDRMConfigInstantiation:
    def test_default_instantiation(self):
        config = DRMConfig()
        assert config is not None
        assert config.kind == "drm"
        assert isinstance(config.model, AnyModelConfig)

    def test_scalar_defaults(self):
        config = DRMConfig()
        assert config.A == 1.0
        assert config.c == 0.0
        assert config.f == 0.0
        assert config.g == 0.0
        assert config.boundary_weight == 1.0

    def test_callable_parameters(self):
        def A_fn(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.eye(x.shape[0])

        def c_fn(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.asarray(2.0)

        config = DRMConfig(A=A_fn, c=c_fn)
        assert callable(config.A)
        assert callable(config.c)
