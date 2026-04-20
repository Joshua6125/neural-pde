"""Tests for LSConfig (configuration validation)."""

import pytest

from src.loss_functions import LSConfig
from src.models import NeuralNetModelConfig


pytestmark = pytest.mark.LS


class TestLSConfigInstantiation:
    """Test valid LSConfig instantiation."""

    def test_default_instantiation(self, ls_config_default):
        """LSConfig can be instantiated with defaults."""
        config = ls_config_default
        assert config is not None
        assert config.kind == "ls"
        assert isinstance(config.model, NeuralNetModelConfig)

    def test_kind_is_ls(self):
        """kind field is always 'ls'."""
        config = LSConfig()
        assert config.kind == "ls"

    def test_model_config_present(self, neuralnet_model_config):
        """model field can be explicitly provided."""
        config = LSConfig(model=neuralnet_model_config)
        assert config.model is neuralnet_model_config

    def test_callable_sources_are_stored(self, callable_f_zero, callable_g_zero):
        """Source callables are accepted and stored."""
        config = LSConfig(f=callable_f_zero, g=callable_g_zero)
        assert config.f is callable_f_zero
        assert config.g is callable_g_zero

    def test_callable_initial_conditions_are_stored(self, callable_v0_zero, callable_sigma0_zero):
        """Initial-condition callables are accepted and stored."""
        config = LSConfig(v0=callable_v0_zero, sigma0=callable_sigma0_zero)
        assert config.v0 is callable_v0_zero
        assert config.sigma0 is callable_sigma0_zero

    def test_boundary_callable_is_stored(self, callable_v_boundary_zero):
        """Boundary callable is accepted and stored."""
        config = LSConfig(v_boundary=callable_v_boundary_zero)
        assert config.v_boundary is callable_v_boundary_zero


class TestLSConfigDefaults:
    """Test default values in LSConfig."""

    def test_f_defaults_to_none(self):
        """f defaults to None."""
        config = LSConfig()
        assert config.f is None

    def test_g_defaults_to_none(self):
        """g defaults to None."""
        config = LSConfig()
        assert config.g is None

    def test_v0_defaults_to_none(self):
        """v0 defaults to None."""
        config = LSConfig()
        assert config.v0 is None

    def test_sigma0_defaults_to_none(self):
        """sigma0 defaults to None."""
        config = LSConfig()
        assert config.sigma0 is None

    def test_v_boundary_defaults_to_none(self):
        """v_boundary defaults to None."""
        config = LSConfig()
        assert config.v_boundary is None


class TestLSConfigImmutability:
    """Test LSConfig immutability."""

    def test_frozen_prevents_attribute_mutation(self):
        """Frozen dataclass raises on mutation."""
        config = LSConfig()
        with pytest.raises((AttributeError, Exception)):
            config.kind = "other" # type: ignore ; This is supposed to be give a warning.

    def test_frozen_prevents_new_attributes(self):
        """Cannot add new attributes to frozen dataclass."""
        config = LSConfig()
        with pytest.raises((AttributeError, Exception)):
            config.new_field = "value" # type: ignore ; This is supposed to give a warning.


class TestLSConfigMultipleArgs:
    """Test instantiation with multiple LSConfig arguments."""

    def test_all_fields_specified(self, neuralnet_model_config, callable_f_zero, callable_g_zero, callable_v0_zero, callable_sigma0_zero, callable_v_boundary_zero):
        """All fields can be specified together."""
        config = LSConfig(
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
        config = LSConfig(f=callable_f_zero)
        assert config.kind == "ls"
        assert config.f is callable_f_zero
        assert config.g is None
        assert config.v0 is None
        assert config.sigma0 is None
        assert config.v_boundary is None

    def test_custom_model_config_is_supported(self):
        """Custom model config instances are supported."""
        model_config = NeuralNetModelConfig(
            hidden_dim=16,
            num_layers=3,
            output_heads={"v": 1, "sigma": 1},
        )
        config = LSConfig(model=model_config)
        assert config.model == model_config


class TestLSConfigBoundaryBehaviour:
    """Test boundary-related configuration behaviour."""

    def test_boundary_callable_can_be_none(self):
        """v_boundary can be omitted for pure interior/IC training."""
        config = LSConfig(v_boundary=None)
        assert config.v_boundary is None

    def test_boundary_callable_can_be_defined(self, callable_v_boundary_linear):
        """v_boundary can be defined for Dirichlet boundary use cases."""
        config = LSConfig(v_boundary=callable_v_boundary_linear)
        assert callable(config.v_boundary)
