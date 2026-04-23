"""Unit tests for Trainer behaviour with deterministic test doubles."""

import jax
import jax.numpy as jnp
import pytest

from src.train import TrainConfig, Trainer
from src.train.trainer import TrainStepMetrics


pytestmark = pytest.mark.training


class TestTrainerInitialisation:
    """Test Trainer constructor and state initialisation behaviour."""

    def test_init_state_uses_train_seed_for_parameter_init(
        self,
        mock_training_method,
        deterministic_integrator,
        optimiser_adam,
        train_cfg_default,
        sample_input_vector_2d,
    ):
        trainer = Trainer(
            method=mock_training_method,
            integrator=deterministic_integrator,
            optimiser=optimiser_adam,
            train_cfg=train_cfg_default,
        )

        state_1 = trainer.init_state(sample_input_vector_2d)
        state_2 = trainer.init_state(sample_input_vector_2d)

        assert jnp.allclose(state_1.params["w"], state_2.params["w"])

    def test_init_state_uses_explicit_integration_seed_when_provided(
        self,
        mock_training_method,
        deterministic_integrator,
        optimiser_adam,
        train_cfg_with_integration_seed,
        sample_input_vector_2d,
    ):
        trainer = Trainer(
            method=mock_training_method,
            integrator=deterministic_integrator,
            optimiser=optimiser_adam,
            train_cfg=train_cfg_with_integration_seed,
        )

        state = trainer.init_state(sample_input_vector_2d)
        assert jnp.array_equal(state.integration_key, jax.random.PRNGKey(99))

    def test_constructor_raises_when_train_config_is_invalid(
        self,
        mock_training_method,
        deterministic_integrator,
        optimiser_adam,
    ):
        bad_cfg = TrainConfig(epochs=0)
        with pytest.raises(AssertionError, match="epochs must be strictly positive"):
            Trainer(
                method=mock_training_method,
                integrator=deterministic_integrator,
                optimiser=optimiser_adam,
                train_cfg=bad_cfg,
            )


class TestTrainerTrainStep:
    """Test single-step behaviour and metrics contracts."""

    def test_train_step_increments_state_step(
        self,
        mock_training_method,
        deterministic_integrator,
        optimiser_adam,
        train_cfg_default,
        sample_input_vector_2d,
    ):
        trainer = Trainer(
            method=mock_training_method,
            integrator=deterministic_integrator,
            optimiser=optimiser_adam,
            train_cfg=train_cfg_default,
        )
        state = trainer.init_state(sample_input_vector_2d)

        next_state, metrics = trainer.train_step(state)

        assert next_state.step == state.step + 1
        assert metrics.step == next_state.step

    def test_train_step_returns_metrics_dataclass(
        self,
        mock_training_method,
        deterministic_integrator,
        optimiser_adam,
        train_cfg_default,
        sample_input_vector_2d,
    ):
        trainer = Trainer(
            method=mock_training_method,
            integrator=deterministic_integrator,
            optimiser=optimiser_adam,
            train_cfg=train_cfg_default,
        )
        state = trainer.init_state(sample_input_vector_2d)

        _, metrics = trainer.train_step(state)

        assert isinstance(metrics, TrainStepMetrics)
        assert isinstance(metrics.total_loss, float)
        assert isinstance(metrics.interior_loss, float)
        assert isinstance(metrics.boundary_loss, float)

    def test_train_step_updates_integration_key_for_key_advancing_integrator(
        self,
        mock_training_method,
        deterministic_integrator,
        optimiser_adam,
        train_cfg_default,
        sample_input_vector_2d,
    ):
        trainer = Trainer(
            method=mock_training_method,
            integrator=deterministic_integrator,
            optimiser=optimiser_adam,
            train_cfg=train_cfg_default,
        )
        state = trainer.init_state(sample_input_vector_2d)

        next_state, _ = trainer.train_step(state)

        assert not jnp.array_equal(next_state.integration_key, state.integration_key)

    def test_train_step_propagates_non_finite_losses(
        self,
        mock_nan_method,
        deterministic_integrator,
        optimiser_adam,
        train_cfg_default,
        sample_input_vector_2d,
    ):
        trainer = Trainer(
            method=mock_nan_method,
            integrator=deterministic_integrator,
            optimiser=optimiser_adam,
            train_cfg=train_cfg_default,
        )
        state = trainer.init_state(sample_input_vector_2d)

        _, metrics = trainer.train_step(state)

        assert jnp.isnan(jnp.array(metrics.total_loss))


    def test_train_step_raises_for_malformed_method_loss_tuple(
        self,
        mock_malformed_method,
        deterministic_integrator,
        optimiser_adam,
        train_cfg_default,
        sample_input_vector_2d,
    ):
        trainer = Trainer(
            method=mock_malformed_method,
            integrator=deterministic_integrator,
            optimiser=optimiser_adam,
            train_cfg=train_cfg_default,
        )
        state = trainer.init_state(sample_input_vector_2d)

        with pytest.raises(ValueError):
            trainer.train_step(state)


class TestTrainerFit:
    """Test fit-loop control flow and edge cases."""

    def test_fit_raises_when_state_and_sample_input_missing(
        self,
        mock_training_method,
        deterministic_integrator,
        optimiser_adam,
        train_cfg_default,
    ):
        trainer = Trainer(
            method=mock_training_method,
            integrator=deterministic_integrator,
            optimiser=optimiser_adam,
            train_cfg=train_cfg_default,
        )

        with pytest.raises(ValueError, match="Must provide either an initial state or sample_input"):
            trainer.fit(sample_input=None, state=None)

    def test_fit_runs_with_sample_input_only(
        self,
        mock_training_method,
        deterministic_integrator,
        optimiser_adam,
        sample_input_vector_2d,
    ):
        cfg = TrainConfig(epochs=3, log_every=1, use_jit=False)
        trainer = Trainer(
            method=mock_training_method,
            integrator=deterministic_integrator,
            optimiser=optimiser_adam,
            train_cfg=cfg,
        )

        state, history = trainer.fit(sample_input=sample_input_vector_2d)

        assert state.step == 3
        assert len(history) == 3

    def test_fit_runs_with_existing_state_only(
        self,
        mock_training_method,
        deterministic_integrator,
        optimiser_adam,
        sample_input_vector_2d,
    ):
        cfg = TrainConfig(epochs=2, log_every=1, use_jit=False)
        trainer = Trainer(
            method=mock_training_method,
            integrator=deterministic_integrator,
            optimiser=optimiser_adam,
            train_cfg=cfg,
        )
        initial_state = trainer.init_state(sample_input_vector_2d)

        state, history = trainer.fit(state=initial_state)

        assert state.step == initial_state.step + 2
        assert len(history) == 2

    @pytest.mark.parametrize("epochs,log_every,expected", [(5, 1, 5), (5, 2, 3), (5, 3, 2)])
    def test_fit_history_respects_log_every(
        self,
        mock_training_method,
        deterministic_integrator,
        optimiser_adam,
        sample_input_vector_2d,
        epochs,
        log_every,
        expected,
    ):
        cfg = TrainConfig(epochs=epochs, log_every=log_every, use_jit=False)
        trainer = Trainer(
            method=mock_training_method,
            integrator=deterministic_integrator,
            optimiser=optimiser_adam,
            train_cfg=cfg,
        )

        _, history = trainer.fit(sample_input=sample_input_vector_2d)

        assert len(history) == expected

    def test_fit_invokes_callback_every_epoch(
        self,
        mock_training_method,
        deterministic_integrator,
        optimiser_adam,
        sample_input_vector_2d,
        callback_recorder,
    ):
        recorded, callback = callback_recorder
        cfg = TrainConfig(epochs=4, log_every=10, use_jit=False)
        trainer = Trainer(
            method=mock_training_method,
            integrator=deterministic_integrator,
            optimiser=optimiser_adam,
            train_cfg=cfg,
        )

        state, _ = trainer.fit(sample_input=sample_input_vector_2d, callback=callback)

        assert state.step == 4
        assert len(recorded) == 4
        assert all(isinstance(m, TrainStepMetrics) for m in recorded)

    def test_fit_propagates_callback_exceptions(
        self,
        mock_training_method,
        deterministic_integrator,
        optimiser_adam,
        sample_input_vector_2d,
    ):
        cfg = TrainConfig(epochs=3, log_every=1, use_jit=False)
        trainer = Trainer(
            method=mock_training_method,
            integrator=deterministic_integrator,
            optimiser=optimiser_adam,
            train_cfg=cfg,
        )

        def bad_callback(metric):
            del metric
            raise RuntimeError("callback failed")

        with pytest.raises(RuntimeError, match="callback failed"):
            trainer.fit(sample_input=sample_input_vector_2d, callback=bad_callback)

    def test_fit_jit_path_executes(
        self,
        mock_training_method,
        deterministic_integrator,
        optimiser_adam,
        train_cfg_short_jit,
        sample_input_vector_2d,
    ):
        trainer = Trainer(
            method=mock_training_method,
            integrator=deterministic_integrator,
            optimiser=optimiser_adam,
            train_cfg=train_cfg_short_jit,
        )

        state, history = trainer.fit(sample_input=sample_input_vector_2d)

        assert state.step == train_cfg_short_jit.epochs
        assert len(history) == train_cfg_short_jit.epochs

    def test_fit_raises_when_method_missing_init_params(
        self,
        deterministic_integrator,
        optimiser_adam,
        sample_input_vector_2d,
    ):
        class NotATrainingMethod:
            pass

        cfg = TrainConfig(epochs=1, use_jit=False)
        trainer = Trainer(
            method=NotATrainingMethod(),  # type: ignore[arg-type]
            integrator=deterministic_integrator,
            optimiser=optimiser_adam,
            train_cfg=cfg,
        )

        with pytest.raises(AttributeError):
            trainer.fit(sample_input=sample_input_vector_2d)
