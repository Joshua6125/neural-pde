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
    NeuralNet,
    KANModel,
    NeuralNetModelConfig,
    KANModelConfig,
    build_model,
)


pytestmark = pytest.mark.models


class TestNeuralNetModelConfigValidation:
    """Test NeuralNetModelConfig.validate behaviour."""

    def test_validate_succeeds_for_valid_config(self):
        """A valid NeuralNetModelConfig validates without raising."""
        cfg = NeuralNetModelConfig(hidden_dim=32, num_layers=2, output_heads={"u": 1})
        cfg.validate()

    def test_validate_raises_for_non_positive_hidden_dim(self):
        """hidden_dim <= 0 raises AssertionError."""
        cfg = NeuralNetModelConfig(hidden_dim=0, num_layers=2)
        with pytest.raises(AssertionError, match="hidden_dim must be strictly positive"):
            cfg.validate()

    def test_validate_raises_for_non_positive_num_layers(self):
        """num_layers <= 0 raises AssertionError."""
        cfg = NeuralNetModelConfig(hidden_dim=8, num_layers=0)
        with pytest.raises(AssertionError, match="num_layers must be strictly positive"):
            cfg.validate()

    def test_validate_raises_for_empty_output_heads(self):
        """Empty output_heads raises AssertionError."""
        cfg = NeuralNetModelConfig(hidden_dim=8, num_layers=1, output_heads={})
        with pytest.raises(AssertionError, match="output_heads"):
            cfg.validate()

    def test_validate_raises_for_empty_head_name(self):
        """Empty output head name raises AssertionError."""
        cfg = NeuralNetModelConfig(hidden_dim=8, num_layers=1, output_heads={"": 1})
        with pytest.raises(AssertionError, match="output head names must be non-empty"):
            cfg.validate()

    def test_validate_raises_for_non_positive_head_dim(self):
        """Output head dim <= 0 raises AssertionError."""
        cfg = NeuralNetModelConfig(hidden_dim=8, num_layers=1, output_heads={"u": 0})
        with pytest.raises(
            AssertionError,
            match="each output head dimension must be strictly positive",
        ):
            cfg.validate()

    def test_config_is_frozen(self):
        """Frozen dataclass prevents mutation of fields."""
        cfg = NeuralNetModelConfig()
        with pytest.raises((AttributeError, Exception)):
            cfg.hidden_dim = 99  # type: ignore[misc]


class TestKANModelConfigValidation:
    """Test KANModelConfig.validate behaviour."""

    def test_validate_succeeds_for_valid_config(self):
        """A valid KANModelConfig validates without raising."""
        cfg = KANModelConfig(
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
        cfg = KANModelConfig(hidden_dim=0)
        with pytest.raises(AssertionError, match="hidden_dim must be strictly positive"):
            cfg.validate()

    def test_validate_raises_for_non_positive_num_layers(self):
        """num_layers <= 0 raises AssertionError."""
        cfg = KANModelConfig(num_layers=0)
        with pytest.raises(AssertionError, match="num_layers must be strictly positive"):
            cfg.validate()

    def test_validate_raises_for_non_positive_input_dim(self):
        """input_dim <= 0 raises AssertionError."""
        cfg = KANModelConfig(input_dim=0)
        with pytest.raises(AssertionError, match="input_dim must be strictly positive"):
            cfg.validate()

    def test_validate_raises_for_non_positive_grid_size(self):
        """grid_size <= 0 raises AssertionError."""
        cfg = KANModelConfig(grid_size=0)
        with pytest.raises(AssertionError, match="grid_size must be strictly positive"):
            cfg.validate()

    def test_validate_raises_for_non_positive_degree(self):
        """degree <= 0 raises AssertionError."""
        cfg = KANModelConfig(degree=0)
        with pytest.raises(AssertionError, match="degree must be strictly positive"):
            cfg.validate()

    def test_config_is_frozen(self):
        """Frozen dataclass prevents mutation of fields."""
        cfg = KANModelConfig()
        with pytest.raises((AttributeError, Exception)):
            cfg.grid_size = 99  # type: ignore[misc]


class TestBuildModel:
    """Test build_model dispatch and error handling."""

    def test_build_model_returns_neuralnet_for_neuralnet_config(self):
        """NeuralNetModelConfig dispatches to NeuralNet."""
        cfg = NeuralNetModelConfig(hidden_dim=16, num_layers=2, output_heads={"u": 1})
        model = build_model(cfg)
        assert isinstance(model._module, NeuralNet)

    def test_build_model_returns_kan_for_kan_config(self):
        """KANModelConfig dispatches to KANModel."""
        cfg = KANModelConfig(hidden_dim=16, num_layers=2, output_heads={"u": 1}, input_dim=1)
        model = build_model(cfg)
        assert isinstance(model._module, KANModel)

    def test_build_model_validates_config_before_building(self):
        """Invalid config fields raise via validate during build_model."""
        cfg = NeuralNetModelConfig(hidden_dim=0)
        with pytest.raises(AssertionError, match="hidden_dim must be strictly positive"):
            build_model(cfg)

    def test_build_model_raises_for_unknown_config_type(self):
        """Unknown config object type raises ValueError."""

        class UnknownConfig:
            pass

        with pytest.raises(ValueError, match="Unknown model config type"):
            build_model(UnknownConfig())  # type: ignore[arg-type]
