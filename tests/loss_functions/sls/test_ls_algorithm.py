"""Tests for SLS algorithm (TrainingMethod interface)."""

import inspect

import jax.numpy as jnp
import pytest

from src.loss_functions import SLS, SLSConfig
from src.train.base import TrainingMethod


pytestmark = pytest.mark.SLS


class TestSLSInitialisation:
    """Test SLS algorithm initialisation."""

    def test_sls_init_with_model_and_config(self, mock_model_valid, sls_config_default):
        """SLS can be initialized with model and config."""
        algorithm = SLS(model=mock_model_valid, config=sls_config_default)
        assert algorithm.model is mock_model_valid
        assert algorithm.config is sls_config_default

    def test_sls_is_training_method(self, mock_model_valid, sls_config_default):
        """SLS implements the TrainingMethod interface."""
        algorithm = SLS(model=mock_model_valid, config=sls_config_default)
        assert isinstance(algorithm, TrainingMethod)


class TestSLSInitParams:
    """Test init_params method (parameter initialisation)."""

    def test_init_params_returns_params(self, mock_model_valid, sls_config_default, rng_key, sample_input_2d):
        """init_params returns model parameters."""
        algorithm = SLS(model=mock_model_valid, config=sls_config_default)
        params = algorithm.init_params(rng_key, sample_input_2d)
        assert params is not None

    def test_init_params_accepts_length_one_v_output(self, mock_model_vector_v, sls_config_default, rng_key, sample_input_2d):
        """A length-one vector v output is accepted."""
        algorithm = SLS(model=mock_model_vector_v, config=sls_config_default)
        params = algorithm.init_params(rng_key, sample_input_2d)
        assert params is not None

    def test_init_params_with_2d_input_validates_sigma_shape(self, mock_model_valid, sls_config_default, rng_key, sample_input_2d):
        """Sigma dimension must match input_dim - 1 for 2D input."""
        algorithm = SLS(model=mock_model_valid, config=sls_config_default)
        params = algorithm.init_params(rng_key, sample_input_2d)
        assert params is not None

    def test_init_params_with_3d_input_validates_sigma_shape(self, mock_model_valid, sls_config_default, rng_key, sample_input_3d):
        """Sigma dimension must match input_dim - 1 for 3D input."""
        algorithm = SLS(model=mock_model_valid, config=sls_config_default)
        params = algorithm.init_params(rng_key, sample_input_3d)
        assert params is not None


class TestSLSInitParamsValidation:
    """Test init_params validation of model output contract."""

    def test_init_params_rejects_non_dict_output(self, mock_model_not_dict, sls_config_default, rng_key, sample_input_2d):
        """init_params rejects non-dict model outputs."""
        algorithm = SLS(model=mock_model_not_dict, config=sls_config_default)
        with pytest.raises(ValueError, match="must return dict"):
            algorithm.init_params(rng_key, sample_input_2d)

    def test_init_params_rejects_missing_v(self, mock_model_missing_v, sls_config_default, rng_key, sample_input_2d):
        """init_params rejects outputs without v."""
        algorithm = SLS(model=mock_model_missing_v, config=sls_config_default)
        with pytest.raises(ValueError, match="'v' and 'sigma' keys"):
            algorithm.init_params(rng_key, sample_input_2d)

    def test_init_params_rejects_missing_sigma(self, mock_model_missing_sigma, sls_config_default, rng_key, sample_input_2d):
        """init_params rejects outputs without sigma."""
        algorithm = SLS(model=mock_model_missing_sigma, config=sls_config_default)
        with pytest.raises(ValueError, match="'v' and 'sigma' keys"):
            algorithm.init_params(rng_key, sample_input_2d)

    def test_init_params_rejects_bad_v_shape(self, mock_model_bad_v_shape, sls_config_default, rng_key, sample_input_2d):
        """init_params rejects v outputs that are not scalar-like."""
        algorithm = SLS(model=mock_model_bad_v_shape, config=sls_config_default)
        with pytest.raises(ValueError, match="'v' output must be scalar"):
            algorithm.init_params(rng_key, sample_input_2d)

    def test_init_params_rejects_bad_sigma_shape(self, mock_model_bad_sigma_shape, sls_config_default, rng_key, sample_input_2d):
        """init_params rejects sigma outputs with the wrong dimension."""
        algorithm = SLS(model=mock_model_bad_sigma_shape, config=sls_config_default)
        with pytest.raises(ValueError, match="'sigma' must output"):
            algorithm.init_params(rng_key, sample_input_2d)


class TestSLSLossFunctions:
    """Test loss_functions method."""

    def test_loss_functions_returns_tuple(self, mock_model_valid, sls_config_default, rng_key, sample_input_2d):
        """loss_functions returns tuple of 2 callables."""
        algorithm = SLS(model=mock_model_valid, config=sls_config_default)
        params = algorithm.init_params(rng_key, sample_input_2d)
        result = algorithm.loss_functions(params)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_loss_functions_returns_callables(self, mock_model_valid, sls_config_default, rng_key, sample_input_2d):
        """Returned loss functions are callable."""
        algorithm = SLS(model=mock_model_valid, config=sls_config_default)
        params = algorithm.init_params(rng_key, sample_input_2d)
        interior_loss_fn, boundary_loss_fn = algorithm.loss_functions(params)
        assert callable(interior_loss_fn)
        assert callable(boundary_loss_fn)

    def test_loss_functions_interior_loss_callable(self, mock_model_valid, sls_config_default, rng_key, sample_input_2d):
        """Interior loss function is callable."""
        algorithm = SLS(model=mock_model_valid, config=sls_config_default)
        params = algorithm.init_params(rng_key, sample_input_2d)
        interior_loss_fn, _ = algorithm.loss_functions(params)
        result = interior_loss_fn(jnp.array([[0.5, 0.25]]))
        assert jnp.isfinite(result[0])

    def test_loss_functions_boundary_loss_callable(self, mock_model_valid, sls_config_default, rng_key, sample_input_2d):
        """Boundary loss function is callable."""
        algorithm = SLS(model=mock_model_valid, config=sls_config_default)
        params = algorithm.init_params(rng_key, sample_input_2d)
        _, boundary_loss_fn = algorithm.loss_functions(params)
        result = boundary_loss_fn(jnp.array([[0.5, 0.25]]), jnp.array([[1.0, 0.0]]))
        assert jnp.isfinite(result[0])


class TestSLSLossFunctionsWithConfig:
    """Test loss_functions creates SLSLoss with correct config parameters."""

    def test_loss_functions_uses_source_terms(self, mock_model_valid, sls_config_with_sources, rng_key, sample_input_2d):
        """Source terms from config flow into the loss functions."""
        algorithm = SLS(model=mock_model_valid, config=sls_config_with_sources)
        params = algorithm.init_params(rng_key, sample_input_2d)
        interior_loss_fn, _ = algorithm.loss_functions(params)
        result = interior_loss_fn(jnp.array([[0.5, 0.25]]))
        assert jnp.isfinite(result[0])

    def test_loss_functions_uses_boundary_condition(self, mock_model_valid, sls_config_with_boundary, rng_key, sample_input_2d):
        """Boundary callable from config is used by the boundary loss."""
        algorithm = SLS(model=mock_model_valid, config=sls_config_with_boundary)
        params = algorithm.init_params(rng_key, sample_input_2d)
        _, boundary_loss_fn = algorithm.loss_functions(params)
        result = boundary_loss_fn(jnp.array([[0.2, 0.0]]), jnp.array([[0.0, 1.0]]))
        assert jnp.isfinite(result[0])

    def test_loss_functions_matches_exact_zero_solution(self, mock_model_valid, callable_v0_zero, callable_sigma0_linear, callable_v_boundary_linear, rng_key, sample_input_2d):
        """Loss closures can represent an exact zero-residual configuration."""
        config = SLSConfig(
            f=0.0,
            g=0.0,
            v0=callable_v0_zero,
            sigma0=callable_sigma0_linear,
            v_boundary=callable_v_boundary_linear,
        )
        algorithm = SLS(model=mock_model_valid, config=config)
        params = algorithm.init_params(rng_key, sample_input_2d)
        interior_loss_fn, boundary_loss_fn = algorithm.loss_functions(params)

        interior = interior_loss_fn(jnp.array([[0.5, 0.25]]))
        boundary_ic = boundary_loss_fn(jnp.array([[0.0, 0.25]]), jnp.array([[-1.0, 0.0]]))
        boundary_bc = boundary_loss_fn(jnp.array([[0.2, 0.0]]), jnp.array([[0.0, 1.0]]))

        assert jnp.allclose(interior[0], 0.0, atol=1e-8)
        assert jnp.allclose(boundary_ic[0], 0.0, atol=1e-8)
        assert jnp.allclose(boundary_bc[0], 0.0, atol=1e-8)


class TestSLSWithDifferentDimensions:
    """Test SLS works with different total input dimensions."""

    def test_ls_2d_total_input(self, mock_model_valid, sls_config_default, rng_key, sample_input_2d):
        """SLS works for [t, x] inputs."""
        algorithm = SLS(model=mock_model_valid, config=sls_config_default)
        params = algorithm.init_params(rng_key, sample_input_2d)
        assert params is not None

    def test_ls_3d_total_input(self, mock_model_valid, sls_config_default, rng_key, sample_input_3d):
        """SLS works for [t, x, y] inputs."""
        algorithm = SLS(model=mock_model_valid, config=sls_config_default)
        params = algorithm.init_params(rng_key, sample_input_3d)
        assert params is not None


class TestSLSModelOutputContract:
    """Test SLS validates model output contract thoroughly."""

    def test_init_params_calls_model_apply(self, mock_model_valid, sls_config_default, rng_key, sample_input_2d):
        """init_params validates the output from model.apply."""
        algorithm = SLS(model=mock_model_valid, config=sls_config_default)
        params = algorithm.init_params(rng_key, sample_input_2d)
        assert params is not None

    def test_loss_functions_uses_bound_model_outputs(self, mock_model_valid, sls_config_default, rng_key, sample_input_2d):
        """loss_functions closes over the bound model parameters."""
        algorithm = SLS(model=mock_model_valid, config=sls_config_default)
        params = algorithm.init_params(rng_key, sample_input_2d)
        interior_loss_fn, _ = algorithm.loss_functions(params)
        first = interior_loss_fn(jnp.array([[0.5, 0.25]]))
        second = interior_loss_fn(jnp.array([[0.5, 0.25]]))
        assert jnp.allclose(first, second)


class TestSLSAlgorithmInterface:
    """Test SLS method signatures and interface compliance."""

    def test_init_params_signature(self, mock_model_valid, sls_config_default):
        """init_params exposes rng_key and sample_input parameters."""
        algorithm = SLS(model=mock_model_valid, config=sls_config_default)
        signature = inspect.signature(algorithm.init_params)
        assert "rng_key" in signature.parameters
        assert "sample_input" in signature.parameters

    def test_loss_functions_signature(self, mock_model_valid, sls_config_default):
        """loss_functions exposes params parameter."""
        algorithm = SLS(model=mock_model_valid, config=sls_config_default)
        signature = inspect.signature(algorithm.loss_functions)
        assert "params" in signature.parameters
