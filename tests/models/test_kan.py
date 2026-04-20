"""Tests for KANModel interface behavior."""

import importlib.util
import warnings
from typing import cast

import pytest
import jax.numpy as jnp


def _require_jaxkan() -> None:
    """Skip the module if optional jaxkan dependency is unavailable."""
    if importlib.util.find_spec("jaxkan") is None:
        message = "Skipping KAN tests because optional dependency 'jaxkan' is not installed"
        warnings.warn(message, RuntimeWarning)
        pytest.skip(message, allow_module_level=True)


_require_jaxkan()

from src.models import KANModel


pytestmark = pytest.mark.models


class TestKANInstantiation:
    """Test KANModel object construction behavior."""

    def test_instantiate_with_defaults(self):
        """KANModel can be created with default optional parameters."""
        model = KANModel(hidden_dim=16, num_layers=2, output_heads={"u": 1})
        assert model.hidden_dim == 16
        assert model.num_layers == 2
        assert model.grid_size == 5
        assert model.degree == 3
        assert model.model_type == "efficient"
        assert model.seed == 42

    def test_instantiate_with_custom_optional_parameters(self):
        """KANModel preserves explicitly provided optional parameters."""
        model = KANModel(
            hidden_dim=32,
            num_layers=3,
            output_heads={"u": 1, "p": 2},
            grid_size=7,
            degree=4,
            model_type="cheby",
            seed=11,
        )
        assert model.grid_size == 7
        assert model.degree == 4
        assert model.model_type == "cheby"
        assert model.seed == 11


class TestKANValidationErrors:
    """Test invalid-user-input error flags exposed by KANModel."""

    def test_raises_for_non_positive_hidden_dim(self, rng_key, sample_1d_input):
        """hidden_dim <= 0 raises ValueError."""
        model = KANModel(hidden_dim=0, num_layers=1, output_heads={"u": 1})
        with pytest.raises(ValueError, match="hidden_dim must be strictly positive"):
            model.init(rng_key, sample_1d_input)

    def test_raises_for_non_positive_num_layers(self, rng_key, sample_1d_input):
        """num_layers <= 0 raises ValueError."""
        model = KANModel(hidden_dim=8, num_layers=0, output_heads={"u": 1})
        with pytest.raises(ValueError, match="num_layers must be strictly positive"):
            model.init(rng_key, sample_1d_input)

    def test_raises_for_empty_output_heads(self, rng_key, sample_1d_input):
        """Empty output_heads raises ValueError."""
        model = KANModel(hidden_dim=8, num_layers=1, output_heads={})
        with pytest.raises(ValueError, match="output_heads must be non-empty"):
            model.init(rng_key, sample_1d_input)

    def test_raises_for_empty_head_name(self, rng_key, sample_1d_input):
        """Empty output head name raises ValueError."""
        model = KANModel(hidden_dim=8, num_layers=1, output_heads={"": 1})
        with pytest.raises(ValueError, match="output head names must be non-empty"):
            model.init(rng_key, sample_1d_input)

    def test_raises_for_non_positive_head_dim(self, rng_key, sample_1d_input):
        """Output head dim <= 0 raises ValueError."""
        model = KANModel(hidden_dim=8, num_layers=1, output_heads={"u": 0})
        with pytest.raises(
            ValueError,
            match="each output head dimension must be strictly positive",
        ):
            model.init(rng_key, sample_1d_input)

    def test_raises_for_unknown_model_type(self, rng_key, sample_1d_input):
        """Unknown model_type raises ValueError."""
        model = KANModel(
            hidden_dim=8,
            num_layers=1,
            output_heads={"u": 1},
            model_type="does-not-exist",
        )
        with pytest.raises(ValueError, match="Unknown model_type"):
            model.init(rng_key, sample_1d_input)


class TestKANApplyInterface:
    """Test KANModel apply output contract behavior."""

    def test_apply_returns_all_output_heads(self, rng_key, sample_2d_input):
        """apply returns exactly the declared head keys."""
        model = KANModel(hidden_dim=8, num_layers=2, output_heads={"u": 1, "p": 2})
        params = model.init(rng_key, sample_2d_input)

        output = cast(dict[str, jnp.ndarray], model.apply(params, sample_2d_input))
        assert set(output.keys()) == {"u", "p"}

    def test_apply_shapes_follow_declared_head_dims(self, rng_key, sample_2d_input):
        """Each output head has shape (batch, declared_dim)."""
        model = KANModel(hidden_dim=8, num_layers=2, output_heads={"u": 1, "p": 2})
        params = model.init(rng_key, sample_2d_input)

        output = cast(dict[str, jnp.ndarray], model.apply(params, sample_2d_input))
        assert output["u"].shape == (4, 1)
        assert output["p"].shape == (4, 2)

    def test_apply_outputs_are_finite(self, rng_key, sample_2d_input):
        """KAN outputs are finite for normal finite batched inputs."""
        model = KANModel(hidden_dim=8, num_layers=2, output_heads={"u": 1})
        params = model.init(rng_key, sample_2d_input)

        output = cast(dict[str, jnp.ndarray], model.apply(params, sample_2d_input))
        assert jnp.all(jnp.isfinite(output["u"]))

    @pytest.mark.parametrize(
        "model_type",
        ["efficient", "cheby", "original", "base", "spline", "chebyshev"],
    )
    def test_supported_model_types_initialize_and_apply(self, model_type, rng_key, sample_1d_input):
        """All supported model types run through init/apply successfully."""
        model = KANModel(
            hidden_dim=8,
            num_layers=1,
            output_heads={"u": 1},
            model_type=model_type,
        )
        params = model.init(rng_key, sample_1d_input)
        output = cast(dict[str, jnp.ndarray], model.apply(params, sample_1d_input))
        assert output["u"].shape == (4, 1)
