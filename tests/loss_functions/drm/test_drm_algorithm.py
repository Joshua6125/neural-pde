"""Tests for DRM algorithm (TrainingMethod interface)."""

import inspect

import jax.numpy as jnp
import pytest

from src.loss_functions import DRM, DRMConfig
from src.train.base import TrainingMethod


pytestmark = pytest.mark.DRM


class TestDRMInitialisation:
    def test_drm_init_with_model_and_config(self, mock_model_valid):
        config = DRMConfig()
        algorithm = DRM(model=mock_model_valid, config=config)
        assert algorithm.model is mock_model_valid
        assert algorithm.config is config

    def test_drm_is_training_method(self, mock_model_valid):
        algorithm = DRM(model=mock_model_valid, config=DRMConfig())
        assert isinstance(algorithm, TrainingMethod)


class TestDRMInitParams:
    def test_init_params_returns_params(self, mock_model_valid, sample_input_2d):
        algorithm = DRM(model=mock_model_valid, config=DRMConfig())
        params = algorithm.init_params(jnp.array([0, 1]), sample_input_2d)
        assert params is not None

    def test_init_params_rejects_non_dict_output(self, mock_model_not_dict, sample_input_2d):
        algorithm = DRM(model=mock_model_not_dict, config=DRMConfig())
        with pytest.raises(ValueError, match="must return dict"):
            algorithm.init_params(jnp.array([0, 1]), sample_input_2d)

    def test_init_params_rejects_missing_u(self, mock_model_missing_u, sample_input_2d):
        algorithm = DRM(model=mock_model_missing_u, config=DRMConfig())
        with pytest.raises(ValueError, match="'u' key"):
            algorithm.init_params(jnp.array([0, 1]), sample_input_2d)

    def test_init_params_rejects_bad_shape(self, mock_model_bad_shape, sample_input_2d):
        algorithm = DRM(model=mock_model_bad_shape, config=DRMConfig())
        with pytest.raises(ValueError, match="must be scalar"):
            algorithm.init_params(jnp.array([0, 1]), sample_input_2d)


class TestDRMLossFunctions:
    def test_loss_functions_returns_tuple(self, mock_model_valid, sample_input_2d):
        algorithm = DRM(model=mock_model_valid, config=DRMConfig())
        params = algorithm.init_params(jnp.array([0, 1]), sample_input_2d)
        result = algorithm.loss_functions(params)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_loss_functions_returns_callables(self, mock_model_valid, sample_input_2d):
        algorithm = DRM(model=mock_model_valid, config=DRMConfig())
        params = algorithm.init_params(jnp.array([0, 1]), sample_input_2d)
        interior_loss_fn, boundary_loss_fn = algorithm.loss_functions(params)
        assert callable(interior_loss_fn)
        assert callable(boundary_loss_fn)

    def test_loss_functions_interior_is_finite(self, mock_model_valid, sample_input_2d):
        algorithm = DRM(model=mock_model_valid, config=DRMConfig())
        params = algorithm.init_params(jnp.array([0, 1]), sample_input_2d)
        interior_loss_fn, _ = algorithm.loss_functions(params)
        result = interior_loss_fn(jnp.array([[0.5, 0.25]]))
        assert jnp.isfinite(result[0])


class TestDRMAlgorithmInterface:
    def test_init_params_signature(self, mock_model_valid):
        algorithm = DRM(model=mock_model_valid, config=DRMConfig())
        signature = inspect.signature(algorithm.init_params)
        assert "rng_key" in signature.parameters
        assert "sample_input" in signature.parameters

    def test_loss_functions_signature(self, mock_model_valid):
        algorithm = DRM(model=mock_model_valid, config=DRMConfig())
        signature = inspect.signature(algorithm.loss_functions)
        assert "params" in signature.parameters
