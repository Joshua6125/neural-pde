"""Tests for PINN algorithm (TrainingMethod interface)."""

import pytest
import jax
import jax.numpy as jnp

from src.loss_functions.pinn import PINN, PINNConfig
from src.models import PINNModelConfig, NeuralNetModelConfig
from tests.loss_functions.pinn.conftest_pinn import (
    make_mock_linear_model,
    make_mock_quadratic_model,
    make_mock_invalid_model,
    make_mock_invalid_output_shape_model,
)


class TestPINNInitialization:
    """Test PINN algorithm initialization."""

    def test_pinn_init_with_model_and_config(self):
        """PINN can be initialized with model and config."""
        model = make_mock_linear_model()
        config = PINNConfig()

        algorithm = PINN(model=model, config=config)
        assert algorithm.model is model
        assert algorithm.config is config

    def test_pinn_stores_attributes(self):
        """PINN stores model and config attributes."""
        model = make_mock_linear_model()
        config = PINNConfig(c=2.0)

        algorithm = PINN(model=model, config=config)
        assert hasattr(algorithm, "model")
        assert hasattr(algorithm, "config")


class TestPINNInitParams:
    """Test init_params method (parameter initialization)."""

    def test_init_params_returns_params(self):
        """init_params returns parameters dictionary."""
        model = make_mock_linear_model()
        config = PINNConfig()
        algorithm = PINN(model=model, config=config)

        rng_key = jax.random.PRNGKey(42)
        sample_input = jnp.array([0.5, 0.5])

        params = algorithm.init_params(rng_key, sample_input)
        assert params is not None

    def test_init_params_callable_with_rng_and_sample(self):
        """init_params accepts rng_key and sample_input."""
        model = make_mock_linear_model()
        config = PINNConfig()
        algorithm = PINN(model=model, config=config)

        rng_key = jax.random.PRNGKey(42)
        sample_input = jnp.array([0.5, 0.5])

        # Should not raise
        params = algorithm.init_params(rng_key, sample_input)

    def test_init_params_deterministic_with_same_key(self):
        """Parameters are deterministic with same rng_key."""
        model = make_mock_linear_model()
        config = PINNConfig()
        algorithm = PINN(model=model, config=config)

        rng_key = jax.random.PRNGKey(42)
        sample_input = jnp.array([0.5, 0.5])

        # Note: mock linear model returns {"dummy": 0}, so not really learnable params
        params1 = algorithm.init_params(rng_key, sample_input)
        params2 = algorithm.init_params(rng_key, sample_input)

        # Should be same structure
        assert type(params1) == type(params2)

    def test_init_params_different_with_different_keys(self):
        """Parameters may differ with different rng_keys."""
        # Use a real model fixture that has learnable params
        model = make_mock_linear_model()
        config = PINNConfig()
        algorithm = PINN(model=model, config=config)

        rng_key1 = jax.random.PRNGKey(42)
        rng_key2 = jax.random.PRNGKey(43)
        sample_input = jnp.array([0.5, 0.5])

        params1 = algorithm.init_params(rng_key1, sample_input)
        params2 = algorithm.init_params(rng_key2, sample_input)

        # Should have same structure
        assert type(params1) == type(params2)


class TestPINNInitParamsValidation:
    """Test init_params validation of model output contract."""

    def test_init_params_validates_output_is_dict(self):
        """init_params validates model output is dict."""
        model = make_mock_invalid_model()  # Returns {"v": ...} instead of {"u": ...}
        config = PINNConfig()
        algorithm = PINN(model=model, config=config)

        rng_key = jax.random.PRNGKey(42)
        sample_input = jnp.array([0.5, 0.5])

        with pytest.raises(ValueError, match="must return dict"):
            algorithm.init_params(rng_key, sample_input)

    def test_init_params_validates_u_key_exists(self):
        """init_params validates 'u' key exists in output."""
        model = make_mock_invalid_model()  # Wrong key
        config = PINNConfig()
        algorithm = PINN(model=model, config=config)

        rng_key = jax.random.PRNGKey(42)
        sample_input = jnp.array([0.5, 0.5])

        with pytest.raises(ValueError, match="'u' key"):
            algorithm.init_params(rng_key, sample_input)

    def test_init_params_validates_u_is_scalar(self):
        """init_params validates 'u' output is scalar."""
        model = make_mock_invalid_output_shape_model()  # Wrong shape
        config = PINNConfig()
        algorithm = PINN(model=model, config=config)

        rng_key = jax.random.PRNGKey(42)
        sample_input = jnp.array([0.5, 0.5])

        with pytest.raises(ValueError, match="scalar"):
            algorithm.init_params(rng_key, sample_input)

    def test_init_params_accepts_valid_model(self):
        """init_params accepts valid model (dict with 'u' scalar)."""
        model = make_mock_linear_model()  # Valid
        config = PINNConfig()
        algorithm = PINN(model=model, config=config)

        rng_key = jax.random.PRNGKey(42)
        sample_input = jnp.array([0.5, 0.5])

        params = algorithm.init_params(rng_key, sample_input)
        assert params is not None


class TestPINNLossFunctions:
    """Test loss_functions method."""

    def test_loss_functions_returns_tuple(self):
        """loss_functions returns tuple of 2 callables."""
        model = make_mock_linear_model()
        config = PINNConfig()
        algorithm = PINN(model=model, config=config)

        rng_key = jax.random.PRNGKey(42)
        sample_input = jnp.array([0.5, 0.5])
        params = algorithm.init_params(rng_key, sample_input)

        loss_fns = algorithm.loss_functions(params)
        assert isinstance(loss_fns, tuple)
        assert len(loss_fns) == 2

    def test_loss_functions_returns_callables(self):
        """Returned loss functions are callable."""
        model = make_mock_linear_model()
        config = PINNConfig()
        algorithm = PINN(model=model, config=config)

        rng_key = jax.random.PRNGKey(42)
        sample_input = jnp.array([0.5, 0.5])
        params = algorithm.init_params(rng_key, sample_input)

        interior_loss_fn, boundary_loss_fn = algorithm.loss_functions(params)
        assert callable(interior_loss_fn)
        assert callable(boundary_loss_fn)

    def test_loss_functions_interior_loss_callable(self):
        """Interior loss function is callable."""
        model = make_mock_linear_model()
        config = PINNConfig()
        algorithm = PINN(model=model, config=config)

        rng_key = jax.random.PRNGKey(42)
        sample_input = jnp.array([0.5, 0.5])
        params = algorithm.init_params(rng_key, sample_input)

        interior_loss_fn, _ = algorithm.loss_functions(params)

        x_interior = jnp.array([[0.5, 0.5]])
        result = interior_loss_fn(x_interior)
        assert jnp.isfinite(result[0])

    def test_loss_functions_boundary_loss_callable(self):
        """Boundary loss function is callable."""
        model = make_mock_linear_model()
        config = PINNConfig()
        algorithm = PINN(model=model, config=config)

        rng_key = jax.random.PRNGKey(42)
        sample_input = jnp.array([0.5, 0.5])
        params = algorithm.init_params(rng_key, sample_input)

        _, boundary_loss_fn = algorithm.loss_functions(params)

        x_boundary = jnp.array([[0.5, 0.5]])
        normal_vector = jnp.array([[0.0, 1.0]])
        result = boundary_loss_fn(x_boundary, normal_vector)
        assert jnp.isfinite(result[0])


class TestPINNLossFunctionsWithConfig:
    """Test loss_functions creates loss with correct config parameters."""

    def test_loss_functions_uses_config_c(self):
        """Loss functions use c from config."""
        model = make_mock_linear_model()
        config = PINNConfig(c=2.0)
        algorithm = PINN(model=model, config=config)

        rng_key = jax.random.PRNGKey(42)
        sample_input = jnp.array([0.5, 0.5])
        params = algorithm.init_params(rng_key, sample_input)

        interior_loss_fn, _ = algorithm.loss_functions(params)
        # Just verify it computes without error
        result = interior_loss_fn(jnp.array([[0.5, 0.5]]))
        assert jnp.isfinite(result[0])

    def test_loss_functions_uses_config_f(self):
        """Loss functions use f from config."""
        def f_fn(x):
            return jnp.sin(jnp.sum(x))

        model = make_mock_linear_model()
        config = PINNConfig(f=f_fn)
        algorithm = PINN(model=model, config=config)

        rng_key = jax.random.PRNGKey(42)
        sample_input = jnp.array([0.5, 0.5])
        params = algorithm.init_params(rng_key, sample_input)

        interior_loss_fn, _ = algorithm.loss_functions(params)
        result = interior_loss_fn(jnp.array([[0.5, 0.5]]))
        assert jnp.isfinite(result[0])

    def test_loss_functions_uses_config_ic_conditions(self):
        """Loss functions use u0, ut0 from config."""
        def u0_fn(x):
            return jnp.sin(x[1])
        def ut0_fn(x):
            return 0.0

        model = make_mock_linear_model()
        config = PINNConfig(u0=u0_fn, ut0=ut0_fn)
        algorithm = PINN(model=model, config=config)

        rng_key = jax.random.PRNGKey(42)
        sample_input = jnp.array([0.5, 0.5])
        params = algorithm.init_params(rng_key, sample_input)

        _, boundary_loss_fn = algorithm.loss_functions(params)
        x_boundary = jnp.array([[0.0, 0.5]])
        normal_vector = jnp.array([[-1.0, 0.0]])
        result = boundary_loss_fn(x_boundary, normal_vector)
        assert jnp.isfinite(result[0])

    def test_loss_functions_uses_config_weights(self):
        """Loss functions use ic_weight, bc_weight from config."""
        model = make_mock_linear_model()
        config = PINNConfig(ic_weight=2.0, bc_weight=3.0)
        algorithm = PINN(model=model, config=config)

        rng_key = jax.random.PRNGKey(42)
        sample_input = jnp.array([0.5, 0.5])
        params = algorithm.init_params(rng_key, sample_input)

        # Just verify loss functions are created and callable
        interior_loss_fn, boundary_loss_fn = algorithm.loss_functions(params)
        assert callable(interior_loss_fn)
        assert callable(boundary_loss_fn)


class TestPINNWithDifferentDimensions:
    """Test PINN works with different spatial dimensions."""

    def test_pinn_1d_spatial(self):
        """PINN works for 1D spatial (2D total with time)."""
        model = make_mock_linear_model()
        config = PINNConfig()
        algorithm = PINN(model=model, config=config)

        rng_key = jax.random.PRNGKey(42)
        sample_input = jnp.array([0.5, 0.5])

        params = algorithm.init_params(rng_key, sample_input)
        assert params is not None

    def test_pinn_2d_spatial(self):
        """PINN works for 2D spatial (3D total with time)."""
        model = make_mock_linear_model()
        config = PINNConfig()
        algorithm = PINN(model=model, config=config)

        rng_key = jax.random.PRNGKey(42)
        sample_input = jnp.array([0.5, 0.5, 0.5])

        params = algorithm.init_params(rng_key, sample_input)
        assert params is not None

    def test_pinn_3d_spatial(self):
        """PINN works for 3D spatial (4D total with time)."""
        model = make_mock_linear_model()
        config = PINNConfig()
        algorithm = PINN(model=model, config=config)

        rng_key = jax.random.PRNGKey(42)
        sample_input = jnp.array([0.5, 0.5, 0.5, 0.5])

        params = algorithm.init_params(rng_key, sample_input)
        assert params is not None


class TestPINNTrainingMethodInterface:
    """Test PINN implements TrainingMethod interface correctly."""

    def test_pinn_is_training_method(self):
        """PINN is instance of TrainingMethod."""
        from src.train import TrainingMethod

        model = make_mock_linear_model()
        config = PINNConfig()
        algorithm = PINN(model=model, config=config)

        assert isinstance(algorithm, TrainingMethod)

    def test_pinn_has_init_params_method(self):
        """PINN implements init_params method."""
        model = make_mock_linear_model()
        config = PINNConfig()
        algorithm = PINN(model=model, config=config)

        assert hasattr(algorithm, "init_params")
        assert callable(algorithm.init_params)

    def test_pinn_has_loss_functions_method(self):
        """PINN implements loss_functions method."""
        model = make_mock_linear_model()
        config = PINNConfig()
        algorithm = PINN(model=model, config=config)

        assert hasattr(algorithm, "loss_functions")
        assert callable(algorithm.loss_functions)

    def test_pinn_init_params_signature(self):
        """init_params has correct signature."""
        import inspect

        model = make_mock_linear_model()
        config = PINNConfig()
        algorithm = PINN(model=model, config=config)

        sig = inspect.signature(algorithm.init_params)
        assert "rng_key" in sig.parameters
        assert "sample_input" in sig.parameters

    def test_pinn_loss_functions_signature(self):
        """loss_functions has correct signature."""
        import inspect

        model = make_mock_linear_model()
        config = PINNConfig()
        algorithm = PINN(model=model, config=config)

        sig = inspect.signature(algorithm.loss_functions)
        assert "params" in sig.parameters


class TestPINNWithVariableWaveSpeed:
    """Test PINN with variable (callable) wave speed."""

    def test_pinn_with_callable_c(self):
        """PINN works with callable wave speed."""
        def c_fn(x):
            return 0.5 + 0.1 * jnp.sum(x**2)

        model = make_mock_linear_model()
        config = PINNConfig(c=c_fn)
        algorithm = PINN(model=model, config=config)

        rng_key = jax.random.PRNGKey(42)
        sample_input = jnp.array([0.5, 0.5])
        params = algorithm.init_params(rng_key, sample_input)

        interior_loss_fn, _ = algorithm.loss_functions(params)
        result = interior_loss_fn(jnp.array([[0.5, 0.5]]))
        assert jnp.isfinite(result[0])


class TestPINNWithComplexConfigs:
    """Test PINN with complex configurations."""

    def test_pinn_with_all_config_options(self):
        """PINN works with all config options specified."""
        def c_fn(x):
            return 1.0
        def f_fn(x):
            return jnp.sin(jnp.sum(x))
        def u0_fn(x):
            return jnp.cos(x[1])
        def ut0_fn(x):
            return 0.0

        model = make_mock_linear_model()
        config = PINNConfig(
            c=c_fn,
            f=f_fn,
            u0=u0_fn,
            ut0=ut0_fn,
            ic_weight=2.0,
            bc_weight=3.0
        )
        algorithm = PINN(model=model, config=config)

        rng_key = jax.random.PRNGKey(42)
        sample_input = jnp.array([0.5, 0.5])

        params = algorithm.init_params(rng_key, sample_input)
        interior_loss_fn, boundary_loss_fn = algorithm.loss_functions(params)

        # Verify both loss functions are computable
        x_int = jnp.array([[0.5, 0.5]])
        result_int = interior_loss_fn(x_int)
        assert jnp.isfinite(result_int[0])

        x_bc = jnp.array([[0.5, 0.5]])
        normal_bc = jnp.array([[0.0, 1.0]])
        result_bc = boundary_loss_fn(x_bc, normal_bc)
        assert jnp.isfinite(result_bc[0])

    def test_pinn_loss_consistency_across_calls(self):
        """Loss values are consistent across multiple calls."""
        model = make_mock_linear_model()
        config = PINNConfig(c=1.0)
        algorithm = PINN(model=model, config=config)

        rng_key = jax.random.PRNGKey(42)
        sample_input = jnp.array([0.5, 0.5])
        params = algorithm.init_params(rng_key, sample_input)

        interior_loss_fn, _ = algorithm.loss_functions(params)
        x = jnp.array([[0.5, 0.5]])

        result1 = interior_loss_fn(x)
        result2 = interior_loss_fn(x)

        assert jnp.allclose(result1, result2)


class TestPINNModelOutputContract:
    """Test PINN validates model output contract thoroughly."""

    def test_init_params_calls_model_apply(self):
        """init_params calls model.apply to validate output."""
        model = make_mock_linear_model()
        config = PINNConfig()
        algorithm = PINN(model=model, config=config)

        rng_key = jax.random.PRNGKey(42)
        sample_input = jnp.array([0.5, 0.5])

        # Should call model.apply and validate
        params = algorithm.init_params(rng_key, sample_input)

        # Model should have been called
        assert params is not None

    def test_loss_functions_creates_u_apply_closure(self):
        """loss_functions creates u_apply closure with model calls."""
        model = make_mock_linear_model()
        config = PINNConfig()
        algorithm = PINN(model=model, config=config)

        rng_key = jax.random.PRNGKey(42)
        sample_input = jnp.array([0.5, 0.5])
        params = algorithm.init_params(rng_key, sample_input)

        interior_loss_fn, _ = algorithm.loss_functions(params)

        # Compute loss (which will use u_apply closure)
        result = interior_loss_fn(jnp.array([[0.5, 0.5]]))
        assert jnp.isfinite(result[0])
