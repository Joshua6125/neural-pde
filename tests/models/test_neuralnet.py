"""Tests for NeuralNet model interface behavior."""

import pytest
import jax
import jax.numpy as jnp
from typing import cast

from src.models import NeuralNet


pytestmark = pytest.mark.models


class TestNeuralNetInstantiation:
    """Test NeuralNet object construction behavior."""

    def test_instantiate_with_valid_args(self, default_single_head):
        """NeuralNet can be created with explicit dimensions and heads."""
        model = NeuralNet(hidden_dim=32, num_layers=2, output_heads=default_single_head)
        assert model.hidden_dim == 32
        assert model.num_layers == 2
        assert model.output_heads == {"output": 1}

    def test_instantiate_with_multi_head(self, multi_output_heads):
        """NeuralNet preserves the provided head mapping."""
        model = NeuralNet(hidden_dim=16, num_layers=3, output_heads=multi_output_heads)
        assert model.output_heads == {"u": 1, "p": 2, "sigma": 3}


class TestNeuralNetInit:
    """Test NeuralNet init contract."""

    def test_init_returns_non_empty_pytree(self, rng_key, sample_1d_input):
        """init produces a non-empty parameter tree."""
        model = NeuralNet(hidden_dim=16, num_layers=2, output_heads={"u": 1})
        params = model.init(rng_key, sample_1d_input)

        leaves = jax.tree_util.tree_leaves(params)
        assert len(leaves) > 0

    def test_init_is_deterministic_for_same_key_and_input(self, sample_2d_input):
        """init is deterministic for equal PRNG keys and equal input shape."""
        model = NeuralNet(hidden_dim=16, num_layers=2, output_heads={"u": 1})

        key_1 = jax.random.PRNGKey(7)
        key_2 = jax.random.PRNGKey(7)
        params_1 = model.init(key_1, sample_2d_input)
        params_2 = model.init(key_2, sample_2d_input)

        leaves_1 = jax.tree_util.tree_leaves(params_1)
        leaves_2 = jax.tree_util.tree_leaves(params_2)
        assert len(leaves_1) == len(leaves_2)
        for p_1, p_2 in zip(leaves_1, leaves_2):
            assert jnp.allclose(p_1, p_2)


class TestNeuralNetApply:
    """Test NeuralNet apply output interface."""

    def test_apply_returns_dict(self, rng_key, sample_1d_input):
        """apply returns a head-name to array mapping."""
        model = NeuralNet(hidden_dim=16, num_layers=2, output_heads={"u": 1})
        params = model.init(rng_key, sample_1d_input)

        output = cast(dict[str, jnp.ndarray], model.apply(params, sample_1d_input))
        assert isinstance(output, dict)
        assert set(output.keys()) == {"u"}

    def test_apply_output_shapes_follow_head_dims(self, rng_key, sample_2d_input, multi_output_heads):
        """Each output head has shape (batch, declared_dim)."""
        model = NeuralNet(hidden_dim=16, num_layers=2, output_heads=multi_output_heads)
        params = model.init(rng_key, sample_2d_input)

        output = cast(dict[str, jnp.ndarray], model.apply(params, sample_2d_input))
        assert output["u"].shape == (4, 1)
        assert output["p"].shape == (4, 2)
        assert output["sigma"].shape == (4, 3)

    def test_apply_outputs_are_finite(self, rng_key, sample_2d_input):
        """All output values are finite for normal finite inputs."""
        model = NeuralNet(hidden_dim=16, num_layers=3, output_heads={"u": 1})
        params = model.init(rng_key, sample_2d_input)

        output = cast(dict[str, jnp.ndarray], model.apply(params, sample_2d_input))
        assert jnp.all(jnp.isfinite(output["u"]))

    @pytest.mark.parametrize("batch_size", [1, 4, 16, 64])
    def test_apply_supports_variable_batch_size(self, rng_key, batch_size):
        """apply can be evaluated on different batch sizes with fixed input dimension."""
        model = NeuralNet(hidden_dim=16, num_layers=2, output_heads={"u": 1})
        params = model.init(rng_key, jnp.ones((1, 2)))

        x = jnp.ones((batch_size, 2))
        output = cast(dict[str, jnp.ndarray], model.apply(params, x))
        assert output["u"].shape == (batch_size, 1)

    def test_apply_is_deterministic_for_same_params(self, rng_key, sample_1d_input):
        """Repeated apply calls with identical params/input are deterministic."""
        model = NeuralNet(hidden_dim=16, num_layers=2, output_heads={"u": 1})
        params = model.init(rng_key, sample_1d_input)

        output_1 = cast(dict[str, jnp.ndarray], model.apply(params, sample_1d_input))
        output_2 = cast(dict[str, jnp.ndarray], model.apply(params, sample_1d_input))
        assert jnp.allclose(output_1["u"], output_2["u"])
