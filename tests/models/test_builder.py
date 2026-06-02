"""Tests for model config validation and model factory behaviour."""

import importlib.util
import warnings

import pytest


def _require_jaxkan() -> None:
    """Skip the module if optional jaxkan dependency is unavailable."""
    if importlib.util.find_spec("jaxkan") is None:
        message = "Skipping model builder tests because optional dependency 'jaxkan' is not installed"
        warnings.warn(message, RuntimeWarning)
        pytest.skip(message, allow_module_level=True)


_require_jaxkan()

from src.models import (
    MLP,
    KAN,
    MLPConfig,
    KANConfig,
    build_model,
)


pytestmark = pytest.mark.models


class TestMLPConfigValidation:
    """Test MLPConfig.validate behaviour."""

    def test_validate_succeeds_for_valid_config(self):
        """A valid MLPConfig validates without raising."""
        cfg = MLPConfig(hidden_dim=32, num_layers=2, output_heads={"u": 1})
        cfg.validate()

    def test_validate_raises_for_non_positive_hidden_dim(self):
        """hidden_dim <= 0 raises AssertionError."""
        cfg = MLPConfig(hidden_dim=0, num_layers=2)
        with pytest.raises(AssertionError, match="hidden_dim must be strictly positive"):
            cfg.validate()

    def test_validate_raises_for_non_positive_num_layers(self):
        """num_layers <= 0 raises AssertionError."""
        cfg = MLPConfig(hidden_dim=8, num_layers=0)
        with pytest.raises(AssertionError, match="num_layers must be strictly positive"):
            cfg.validate()

    def test_validate_raises_for_empty_output_heads(self):
        """Empty output_heads raises AssertionError."""
        cfg = MLPConfig(hidden_dim=8, num_layers=1, output_heads={})
        with pytest.raises(AssertionError, match="output_heads"):
            cfg.validate()

    def test_validate_raises_for_empty_head_name(self):
        """Empty output head name raises AssertionError."""
        cfg = MLPConfig(hidden_dim=8, num_layers=1, output_heads={"": 1})
        with pytest.raises(AssertionError, match="output head names must be non-empty"):
            cfg.validate()

    def test_validate_raises_for_non_positive_head_dim(self):
        """Output head dim <= 0 raises AssertionError."""
        cfg = MLPConfig(hidden_dim=8, num_layers=1, output_heads={"u": 0})
        with pytest.raises(
            AssertionError,
            match="each output head dimension must be strictly positive",
        ):
            cfg.validate()

    def test_config_is_frozen(self):
        """Frozen dataclass prevents mutation of fields."""
        cfg = MLPConfig()
        with pytest.raises((AttributeError, Exception)):
            cfg.hidden_dim = 99  # type: ignore[misc]


class TestKANConfigValidation:
    """Test KANConfig.validate behaviour."""

    def test_validate_succeeds_for_valid_config(self):
        """A valid KANConfig validates without raising."""
        cfg = KANConfig(
            hidden_dim=32,
            num_layers=2,
            output_heads={"u": 1},
            input_dim=1,
            grid_size=5,
            degree=3,
            model_type="efficient",
        )
        cfg.validate()

    def test_validate_raises_for_non_positive_hidden_dim(self):
        """hidden_dim <= 0 raises AssertionError."""
        cfg = KANConfig(hidden_dim=0)
        with pytest.raises(AssertionError, match="hidden_dim must be strictly positive"):
            cfg.validate()

    def test_validate_raises_for_non_positive_num_layers(self):
        """num_layers <= 0 raises AssertionError."""
        cfg = KANConfig(num_layers=0)
        with pytest.raises(AssertionError, match="num_layers must be strictly positive"):
            cfg.validate()

    def test_validate_raises_for_non_positive_input_dim(self):
        """input_dim <= 0 raises AssertionError."""
        cfg = KANConfig(input_dim=0)
        with pytest.raises(AssertionError, match="input_dim must be strictly positive"):
            cfg.validate()

    def test_validate_raises_for_non_positive_grid_size(self):
        """grid_size <= 0 raises AssertionError."""
        cfg = KANConfig(grid_size=0)
        with pytest.raises(AssertionError, match="grid_size must be strictly positive"):
            cfg.validate()

    def test_validate_raises_for_non_positive_degree(self):
        """degree <= 0 raises AssertionError."""
        cfg = KANConfig(degree=0)
        with pytest.raises(AssertionError, match="degree must be strictly positive"):
            cfg.validate()

    def test_config_is_frozen(self):
        """Frozen dataclass prevents mutation of fields."""
        cfg = KANConfig()
        with pytest.raises((AttributeError, Exception)):
            cfg.grid_size = 99  # type: ignore[misc]


class TestBuildModel:
    """Test build_model dispatch and error handling."""

    def test_build_model_returns_mlp_for_mlp_config(self):
        """MLPConfig dispatches to MLP."""
        cfg = MLPConfig(hidden_dim=16, num_layers=2, output_heads={"u": 1})
        model = build_model(cfg)
        assert isinstance(model._module, MLP)

    def test_build_model_returns_kan_for_kan_config(self):
        """KANConfig dispatches to KAN."""
        cfg = KANConfig(hidden_dim=16, num_layers=2, output_heads={"u": 1}, input_dim=1)
        model = build_model(cfg)
        assert isinstance(model._module, KAN)

    def test_build_model_validates_config_before_building(self):
        """Invalid config fields raise via validate during build_model."""
        cfg = MLPConfig(hidden_dim=0)
        with pytest.raises(AssertionError, match="hidden_dim must be strictly positive"):
            build_model(cfg)

    def test_build_model_raises_for_unknown_config_type(self):
        """Unknown config object type raises ValueError."""

        class UnknownConfig:
            pass

        with pytest.raises(ValueError, match="Unknown model config type"):
            build_model(UnknownConfig())  # type: ignore[arg-type]

    def test_build_model_kan_custom_parameters(self):
        cfg = KANConfig(hidden_dim=16, num_layers=2, output_heads={"u": 1}, input_dim=4)
        model = build_model(cfg)
        assert model._module.hidden_dim == 16
        assert model._module.num_layers == 2
        assert model._module.input_dim == 4
