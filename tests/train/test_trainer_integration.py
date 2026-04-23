"""Integration smoke tests for Trainer with real methods and integrators."""

import jax.numpy as jnp
import optax
import pytest

from src.integration import MonteCarloConfig, QuadratureConfig, get_integrator
from src.train import TrainConfig, Trainer


pytestmark = pytest.mark.training


class TestTrainerWithRealPINN:
    """Trainer integration checks with real PINN method."""

    def test_fit_runs_with_quadrature_integrator(self, real_pinn_method, sample_input_vector_2d):
        integrator = get_integrator(
            QuadratureConfig(dim=2, x_min=0.0, x_max=1.0, degree=4, adaptive_integration=False)
        )
        cfg = TrainConfig(epochs=2, learning_rate=1e-3, optimiser="adam", seed=0, log_every=1, use_jit=False)
        optimiser = optax.adam(cfg.learning_rate)

        trainer = Trainer(method=real_pinn_method, integrator=integrator, optimiser=optimiser, train_cfg=cfg)
        state, history = trainer.fit(sample_input=sample_input_vector_2d)

        assert state.step == cfg.epochs
        assert len(history) == cfg.epochs
        assert all(jnp.isfinite(jnp.asarray(metric.total_loss)) for metric in history)


class TestTrainerWithRealLS:
    """Trainer integration checks with real LS method."""

    def test_fit_runs_with_quadrature_integrator(self, real_ls_method, sample_input_vector_2d):
        integrator = get_integrator(
            QuadratureConfig(dim=2, x_min=0.0, x_max=1.0, degree=3, adaptive_integration=False)
        )
        cfg = TrainConfig(epochs=2, learning_rate=1e-3, optimiser="adamw", seed=1, log_every=1, use_jit=False)
        optimiser = optax.adamw(cfg.learning_rate)

        trainer = Trainer(method=real_ls_method, integrator=integrator, optimiser=optimiser, train_cfg=cfg)
        state, history = trainer.fit(sample_input=sample_input_vector_2d)

        assert state.step == cfg.epochs
        assert len(history) == cfg.epochs
        assert all(jnp.isfinite(jnp.asarray(metric.interior_loss)) for metric in history)

    def test_fit_runs_with_monte_carlo_integrator_and_advances_key(self, real_ls_method, sample_input_vector_2d):
        integrator = get_integrator(
            MonteCarloConfig(
                dim=2,
                x_min=0.0,
                x_max=1.0,
                monte_carlo_interior_samples=64,
                monte_carlo_boundary_samples=16,
                monte_carlo_seed=5,
            )
        )
        cfg = TrainConfig(
            epochs=2,
            learning_rate=1e-3,
            optimiser="sgd",
            seed=11,
            integration_seed=29,
            log_every=1,
            use_jit=False,
        )
        trainer = Trainer(
            method=real_ls_method,
            integrator=integrator,
            optimiser=optax.sgd(cfg.learning_rate),
            train_cfg=cfg,
        )

        state = trainer.init_state(sample_input_vector_2d)
        key_before = state.integration_key
        state, history = trainer.fit(state=state)

        assert state.step == cfg.epochs
        assert len(history) == cfg.epochs
        assert not jnp.array_equal(state.integration_key, key_before)


class TestTrainerIntegrationHistorySemantics:
    """Cross-method checks for fit history behavior."""

    @pytest.mark.parametrize("epochs,log_every,expected_history", [(4, 1, 4), (4, 2, 2)])
    def test_history_length_matches_logging_schedule(
        self,
        real_pinn_method,
        sample_input_vector_2d,
        epochs,
        log_every,
        expected_history,
    ):
        integrator = get_integrator(
            QuadratureConfig(dim=2, x_min=0.0, x_max=1.0, degree=3, adaptive_integration=False)
        )
        cfg = TrainConfig(
            epochs=epochs,
            learning_rate=5e-4,
            optimiser="adam",
            seed=0,
            log_every=log_every,
            use_jit=False,
        )
        trainer = Trainer(
            method=real_pinn_method,
            integrator=integrator,
            optimiser=optax.adam(cfg.learning_rate),
            train_cfg=cfg,
        )

        _, history = trainer.fit(sample_input=sample_input_vector_2d)

        assert len(history) == expected_history
