"""Tests for SLSConfig (configuration validation)."""

import pytest

from src.loss_functions import SLSConfig
from src.models import NeuralNetModelConfig


pytestmark = pytest.mark.SLS


class TestSLSConfigInstantiation:
    """Test valid SLSConfig instantiation."""

    def test_default_instantiation(self, sls_config_default):
        """SLSConfig can be instantiated with defaults."""
        config = sls_config_default
        assert config is not None
        assert config.kind == "sls"
        assert isinstance(config.model, NeuralNetModelConfig)

    def test_kind_is_ls(self):
        """kind field is always 'sls'."""
        config = SLSConfig()
        assert config.kind == "sls"

    def test_model_config_present(self, neuralnet_model_config):
        """model field can be explicitly provided."""
        config = SLSConfig(model=neuralnet_model_config)
        assert config.model is neuralnet_model_config

    def test_callable_sources_are_stored(self, callable_f_zero, callable_g_zero):
        """Source callables are accepted and stored."""
        config = SLSConfig(f=callable_f_zero, g=callable_g_zero)
        assert config.f is callable_f_zero
        assert config.g is callable_g_zero

    def test_callable_initial_conditions_are_stored(self, callable_v0_zero, callable_sigma0_zero):
        """Initial-condition callables are accepted and stored."""
        config = SLSConfig(v0=callable_v0_zero, sigma0=callable_sigma0_zero)
        assert config.v0 is callable_v0_zero
        assert config.sigma0 is callable_sigma0_zero

    def test_boundary_callable_is_stored(self, callable_v_boundary_zero):
        """Boundary callable is accepted and stored."""
        config = SLSConfig(v_boundary=callable_v_boundary_zero)
        assert config.v_boundary is callable_v_boundary_zero


class TestSLSConfigDefaults:
    """Test default values in SLSConfig."""

    def test_f_defaults_to_zero(self):
        """f defaults to zero."""
        config = SLSConfig()
        assert config.f == 0.0

    def test_g_defaults_to_zero(self):
        """g defaults to zero."""
        config = SLSConfig()
        assert config.g == 0.0

    def test_v0_defaults_to_zero(self):
        """v0 defaults to zero."""
        config = SLSConfig()
        assert config.v0 == 0.0

    def test_sigma0_defaults_to_zero(self):
        """sigma0 defaults to zero."""
        config = SLSConfig()
        assert config.sigma0 == 0.0

    def test_v_boundary_defaults_to_zero(self):
        """v_boundary defaults to zero."""
        config = SLSConfig()
        assert config.v_boundary == 0.0


class TestSLSConfigImmutability:
    """Test SLSConfig immutability."""

    def test_frozen_prevents_attribute_mutation(self):
        """Frozen dataclass raises on mutation."""
        config = SLSConfig()
        with pytest.raises((AttributeError, Exception)):
            config.kind = "other" # type: ignore ; This is supposed to be give a warning.

    def test_frozen_prevents_new_attributes(self):
        """Cannot add new attributes to frozen dataclass."""
        config = SLSConfig()
        with pytest.raises((AttributeError, Exception)):
            config.new_field = "value" # type: ignore ; This is supposed to give a warning.


class TestSLSConfigMultipleArgs:
    """Test instantiation with multiple SLSConfig arguments."""

    def test_all_fields_specified(self, neuralnet_model_config, callable_f_zero, callable_g_zero, callable_v0_zero, callable_sigma0_zero, callable_v_boundary_zero):
        """All fields can be specified together."""
        config = SLSConfig(
            model=neuralnet_model_config,
            f=callable_f_zero,
            g=callable_g_zero,
            v0=callable_v0_zero,
            sigma0=callable_sigma0_zero,
            v_boundary=callable_v_boundary_zero,
        )

        assert config.model is neuralnet_model_config
        assert config.f is callable_f_zero
        assert config.g is callable_g_zero
        assert config.v0 is callable_v0_zero
        assert config.sigma0 is callable_sigma0_zero
        assert config.v_boundary is callable_v_boundary_zero

    def test_mixed_defaults_and_explicit_values(self, callable_f_zero):
        """Defaults and explicit values can be mixed."""
        config = SLSConfig(f=callable_f_zero)
        assert config.kind == "sls"
        assert config.f is callable_f_zero
        assert config.g == 0.0
        assert config.v0 == 0.0
        assert config.sigma0 ==0.0
        assert config.v_boundary == 0.0

    def test_custom_model_config_is_supported(self):
        """Custom model config instances are supported."""
        model_config = NeuralNetModelConfig(
            hidden_dim=16,
            num_layers=3,
            output_heads={"v": 1, "sigma": 1},
        )
        config = SLSConfig(model=model_config)
        assert config.model == model_config


class TestSLSConfigBoundaryBehaviour:
    """Test boundary-related configuration behaviour."""

    def test_boundary_callable_can_be_none(self):
        """v_boundary can be omitted for pure interior/IC training."""
        config = SLSConfig(v_boundary=0.0)
        assert config.v_boundary == 0.0

    def test_boundary_callable_can_be_defined(self, callable_v_boundary_linear):
        """v_boundary can be defined for Dirichlet boundary use cases."""
        config = SLSConfig(v_boundary=callable_v_boundary_linear)
        assert callable(config.v_boundary)
