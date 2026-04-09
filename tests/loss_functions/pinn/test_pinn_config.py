"""Tests for PINNConfig (configuration validation)."""

import pytest
import jax.numpy as jnp

from src.loss_functions import PINNConfig
from src.models import PINNModelConfig


class TestPINNConfigInstantiation:
    """Test valid PINNConfig instantiation."""

    def test_default_instantiation(self):
        """PINNConfig can be instantiated with defaults."""
        config = PINNConfig()
        assert config is not None
        assert hasattr(config, "kind")
        assert hasattr(config, "model")
        assert hasattr(config, "c")

    def test_kind_is_pinn(self):
        """kind field is always 'pinn'."""
        config = PINNConfig()
        assert config.kind == "pinn"

    def test_model_config_present(self):
        """model field contains PINNModelConfig."""
        config = PINNConfig()
        assert isinstance(config.model, PINNModelConfig)

    def test_scalar_wave_speed(self):
        """Wave speed c can be a scalar float."""
        config = PINNConfig(c=2.5)
        assert config.c == 2.5

    def test_callable_wave_speed(self):
        """Wave speed c can be a callable."""
        def c_fn(x):
            return 1.0 + jnp.sum(x**2)
        config = PINNConfig(c=c_fn)
        assert callable(config.c)
        assert config.c is c_fn

    def test_callable_forcing_term(self):
        """Forcing term f can be a callable."""
        def f_fn(x):
            return jnp.sin(jnp.sum(x))
        config = PINNConfig(f=f_fn)
        assert callable(config.f)

    def test_callable_u0(self):
        """Initial displacement u0 can be a callable."""
        def u0_fn(x):
            return jnp.cos(x[1])
        config = PINNConfig(u0=u0_fn)
        assert callable(config.u0)

    def test_callable_ut0(self):
        """Initial velocity ut0 can be a callable."""
        def ut0_fn(x):
            return 0.5
        config = PINNConfig(ut0=ut0_fn)
        assert callable(config.ut0)

    def test_ic_weight_positive(self):
        """IC weight can be positive."""
        config = PINNConfig(ic_weight=2.0)
        assert config.ic_weight == 2.0

    def test_ic_weight_zero(self):
        """IC weight can be zero."""
        config = PINNConfig(ic_weight=0.0)
        assert config.ic_weight == 0.0

    def test_ic_weight_negative(self):
        """IC weight can be negative (edge case)."""
        config = PINNConfig(ic_weight=-1.0)
        assert config.ic_weight == -1.0

    def test_bc_weight_positive(self):
        """BC weight can be positive."""
        config = PINNConfig(bc_weight=3.0)
        assert config.bc_weight == 3.0

    def test_bc_weight_zero(self):
        """BC weight can be zero."""
        config = PINNConfig(bc_weight=0.0)
        assert config.bc_weight == 0.0


class TestPINNConfigDefaults:
    """Test default values in PINNConfig."""

    def test_c_defaults_to_one(self):
        """c defaults to 1.0."""
        config = PINNConfig()
        assert config.c == 1.0

    def test_f_defaults_to_none(self):
        """f defaults to None."""
        config = PINNConfig()
        assert config.f is None

    def test_u0_defaults_to_none(self):
        """u0 defaults to None."""
        config = PINNConfig()
        assert config.u0 is None

    def test_ut0_defaults_to_none(self):
        """ut0 defaults to None."""
        config = PINNConfig()
        assert config.ut0 is None

    def test_ic_weight_defaults_to_one(self):
        """ic_weight defaults to 1.0."""
        config = PINNConfig()
        assert config.ic_weight == 1.0

    def test_bc_weight_defaults_to_one(self):
        """bc_weight defaults to 1.0."""
        config = PINNConfig()
        assert config.bc_weight == 1.0


class TestPINNConfigTypes:
    """Test type validation."""

    def test_c_accepts_float(self):
        """c accepts float type."""
        config = PINNConfig(c=1.5)
        assert isinstance(config.c, float)

    def test_c_accepts_callable(self):
        """c accepts callable type."""
        fn = lambda x: 1.0
        config = PINNConfig(c=fn)
        assert callable(config.c)

    def test_f_accepts_none(self):
        """f accepts None value."""
        config = PINNConfig(f=None)
        assert config.f is None

    def test_f_accepts_callable(self):
        """f accepts callable type."""
        fn = lambda x: jnp.sum(x)
        config = PINNConfig(f=fn)
        assert callable(config.f)

    def test_u0_accepts_none(self):
        """u0 accepts None value."""
        config = PINNConfig(u0=None)
        assert config.u0 is None

    def test_u0_accepts_callable(self):
        """u0 accepts callable type."""
        fn = lambda x: 0.0
        config = PINNConfig(u0=fn)
        assert callable(config.u0)

    def test_ut0_accepts_none(self):
        """ut0 accepts None value."""
        config = PINNConfig(ut0=None)
        assert config.ut0 is None

    def test_ut0_accepts_callable(self):
        """ut0 accepts callable type."""
        fn = lambda x: 1.0
        config = PINNConfig(ut0=fn)
        assert callable(config.ut0)

    def test_weights_accept_float(self):
        """Weights accept float type."""
        config = PINNConfig(ic_weight=1.5, bc_weight=2.5)
        assert isinstance(config.ic_weight, float)
        assert isinstance(config.bc_weight, float)


class TestPINNConfigImmutability:
    """Test that PINNConfig is immutable (frozen dataclass)."""

    def test_frozen_prevents_attribute_mutation(self):
        """Frozen dataclass raises error on attribute assignment."""
        config = PINNConfig()
        with pytest.raises((AttributeError, Exception)):  # FrozenInstanceError
            config.c = 2.0

    def test_frozen_prevents_new_attributes(self):
        """Cannot add new attributes to frozen dataclass."""
        config = PINNConfig()
        with pytest.raises((AttributeError, Exception)):
            config.new_field = "value"


class TestPINNConfigMultipleArgs:
    """Test instantiation with multiple arguments."""

    def test_all_fields_specified(self):
        """Can specify all fields in PINNConfig."""
        def c_fn(x):
            return 1.0
        def f_fn(x):
            return 0.0
        def u0_fn(x):
            return 0.0
        def ut0_fn(x):
            return 0.0

        config = PINNConfig(
            c=c_fn,
            f=f_fn,
            u0=u0_fn,
            ut0=ut0_fn,
            ic_weight=2.0,
            bc_weight=3.0
        )

        assert config.c is c_fn
        assert config.f is f_fn
        assert config.u0 is u0_fn
        assert config.ut0 is ut0_fn
        assert config.ic_weight == 2.0
        assert config.bc_weight == 3.0

    def test_mixed_scalar_and_callable(self):
        """Can mix scalar wave speed with callable ICs."""
        def u0_fn(x):
            return jnp.sin(jnp.sum(x))

        config = PINNConfig(c=1.5, u0=u0_fn)
        assert config.c == 1.5
        assert config.u0 is u0_fn

    def test_callable_wave_speed_with_forcing(self):
        """Can use callable wave speed with forcing term."""
        def c_fn(x):
            return jnp.sqrt(1.0 + jnp.sum(x**2))
        def f_fn(x):
            return jnp.sin(x[0])

        config = PINNConfig(c=c_fn, f=f_fn)
        assert callable(config.c)
        assert callable(config.f)


class TestPINNConfigWeightRanges:
    """Test various weight ranges."""

    def test_large_ic_weight(self):
        """IC weight can be very large."""
        config = PINNConfig(ic_weight=1000.0)
        assert config.ic_weight == 1000.0

    def test_small_ic_weight(self):
        """IC weight can be very small."""
        config = PINNConfig(ic_weight=0.001)
        assert config.ic_weight == 0.001

    def test_asymmetric_weights(self):
        """IC and BC weights can be very different."""
        config = PINNConfig(ic_weight=0.1, bc_weight=100.0)
        assert config.ic_weight == 0.1
        assert config.bc_weight == 100.0

    def test_equal_weights(self):
        """IC and BC weights can be equal."""
        config = PINNConfig(ic_weight=1.5, bc_weight=1.5)
        assert config.ic_weight == config.bc_weight


class TestPINNConfigWithMultipleDimensions:
    """Test configs work with different spatial dimensions."""

    def test_1d_spatial_callable_works(self):
        """Callables work with 1D spatial + time (2D total)."""
        def u0_fn(x):
            # x has shape (2,): [t, x_spatial]
            return jnp.sin(x[1])

        config = PINNConfig(u0=u0_fn)
        # Test evaluation
        test_x = jnp.array([0.5, 0.5])
        result = config.u0(test_x)
        assert isinstance(result, jnp.ndarray)

    def test_2d_spatial_callable_works(self):
        """Callables work with 2D spatial + time (3D total)."""
        def u0_fn(x):
            # x has shape (3,): [t, x, y]
            return jnp.sin(x[1] * x[2])

        config = PINNConfig(u0=u0_fn)
        # Test evaluation
        test_x = jnp.array([0.5, 0.5, 0.5])
        result = config.u0(test_x)
        assert isinstance(result, jnp.ndarray)

    def test_3d_spatial_callable_works(self):
        """Callables work with 3D spatial + time (4D total)."""
        def u0_fn(x):
            # x has shape (4,): [t, x, y, z]
            return jnp.sum(x[1:] ** 2)

        config = PINNConfig(u0=u0_fn)
        # Test evaluation
        test_x = jnp.array([0.5, 0.5, 0.5, 0.5])
        result = config.u0(test_x)
        assert isinstance(result, jnp.ndarray)


class TestPINNConfigCallableReturnTypes:
    """Test callable parameters return expected types."""

    def test_c_callable_returns_scalar(self):
        """c callable should return scalar."""
        def c_fn(x):
            return 1.5

        config = PINNConfig(c=c_fn)
        result = config.c(jnp.array([0.5, 0.5]))
        assert isinstance(result, (float, jnp.ndarray))

    def test_f_callable_returns_scalar(self):
        """f callable should return scalar."""
        def f_fn(x):
            return jnp.sum(x)

        config = PINNConfig(f=f_fn)
        result = config.f(jnp.array([0.5, 0.5]))
        assert isinstance(result, jnp.ndarray)

    def test_u0_callable_returns_scalar(self):
        """u0 callable should return scalar."""
        def u0_fn(x):
            return jnp.sin(jnp.sum(x))

        config = PINNConfig(u0=u0_fn)
        result = config.u0(jnp.array([0.5, 0.5]))
        assert isinstance(result, jnp.ndarray)

    def test_ut0_callable_returns_scalar(self):
        """ut0 callable should return scalar."""
        def ut0_fn(x):
            return jnp.cos(x[0])

        config = PINNConfig(ut0=ut0_fn)
        result = config.ut0(jnp.array([0.5, 0.5]))
        assert isinstance(result, jnp.ndarray)
