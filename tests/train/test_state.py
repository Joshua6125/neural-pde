"""Tests for training state/config helpers."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import optax
import pytest

from src.train import TrainConfig, TrainState, get_optimiser


pytestmark = pytest.mark.training


class TestTrainConfigValidation:
    """Validate TrainConfig contracts and failure paths."""

    def test_validate_accepts_valid_defaults(self):
        cfg = TrainConfig()
        cfg.validate()

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("epochs", 0, "epochs must be strictly positive"),
            ("epochs", -1, "epochs must be strictly positive"),
            ("learning_rate", 0.0, "learning_rate must be strictly positive"),
            ("learning_rate", -0.1, "learning_rate must be strictly positive"),
            ("log_every", 0, "log_every must be strictly positive"),
            ("log_every", -2, "log_every must be strictly positive"),
        ],
    )
    def test_validate_raises_for_invalid_scalar_fields(self, field, value, message):
        cfg = replace(TrainConfig(), **{field: value})
        with pytest.raises(AssertionError, match=message):
            cfg.validate()

    def test_validate_accepts_none_integration_seed(self):
        cfg = TrainConfig(integration_seed=None)
        cfg.validate()

    def test_validate_accepts_zero_integration_seed(self):
        cfg = TrainConfig(integration_seed=0)
        cfg.validate()

    def test_validate_raises_for_negative_integration_seed(self):
        cfg = TrainConfig(integration_seed=-1)
        with pytest.raises(AssertionError, match="integration_seed must be non-negative"):
            cfg.validate()


class TestGetOptimiser:
    """Test optimiser factory behaviour."""

    def test_get_optimiser_returns_adam(self):
        optimiser = get_optimiser(TrainConfig(optimiser="adam", learning_rate=1e-3))
        assert isinstance(optimiser, optax.GradientTransformation)

    def test_get_optimiser_returns_adamw(self):
        optimiser = get_optimiser(TrainConfig(optimiser="adamw", learning_rate=1e-3))
        assert isinstance(optimiser, optax.GradientTransformation)

    def test_get_optimiser_returns_sgd(self):
        optimiser = get_optimiser(TrainConfig(optimiser="sgd", learning_rate=1e-3))
        assert isinstance(optimiser, optax.GradientTransformation)

    def test_get_optimiser_raises_for_unknown_optimiser(self):
        cfg = TrainConfig(optimiser="adam")
        cfg = replace(cfg, optimiser="mystery")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown optimiser"):
            get_optimiser(cfg)


class TestTrainStateApplyGradients:
    """Test TrainState gradient application semantics."""

    def test_apply_gradients_increments_step_and_updates_params(self):
        params = {"w": jnp.array(2.0)}
        optimiser = optax.sgd(learning_rate=0.1)
        state = TrainState(
            step=0,
            params=params,
            opt_state=optimiser.init(params),
            integration_key=jax.random.PRNGKey(1),
        )

        grads = {"w": jnp.array(3.0)}
        next_state = state.apply_gradients(grads, optimiser)

        assert next_state.step == 1
        assert jnp.allclose(next_state.params["w"], jnp.array(1.7))

    def test_apply_gradients_keeps_integration_key_unchanged(self):
        params = {"w": jnp.array(0.0)}
        optimiser = optax.adam(learning_rate=0.01)
        integration_key = jax.random.PRNGKey(17)
        state = TrainState(
            step=3,
            params=params,
            opt_state=optimiser.init(params),
            integration_key=integration_key,
        )

        grads = {"w": jnp.array(1.0)}
        next_state = state.apply_gradients(grads, optimiser)

        assert jnp.array_equal(next_state.integration_key, integration_key)

    def test_train_state_is_frozen(self):
        params = {"w": jnp.array(0.0)}
        optimiser = optax.sgd(learning_rate=0.1)
        state = TrainState(
            step=0,
            params=params,
            opt_state=optimiser.init(params),
            integration_key=jax.random.PRNGKey(0),
        )

        with pytest.raises((AttributeError, Exception)):
            state.step = 1  # type: ignore[misc]
