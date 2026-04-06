"""Training execution interface."""

from typing import Callable

import jax.numpy as jnp

from .algorithms import AlgorithmConfig, build_algorithm
from .integration import IntegrationConfig, get_integrator
from .models import AnyModelConfig, build_model
from .train import (
    TrainConfig,
    TrainState,
    TrainStepMetrics,
    Trainer,
    get_optimizer,
)


def run_training(
        algorithm_cfg: AlgorithmConfig,
        integration_cfg: IntegrationConfig,
        model_cfg: AnyModelConfig,
        train_cfg: TrainConfig,
        sample_input: jnp.ndarray | None = None,
        state: TrainState | None = None,
        callback: Callable[[TrainStepMetrics], None] | None = None,
    ) -> tuple[TrainState, list[TrainStepMetrics]]:
    """Execute training with algorithm.

    Parameters
    ----------
    algorithm_cfg : AlgorithmConfig
        Algorithm configuration (PINNConfig, LSConfig, etc.)
        Bundles model architecture and PDE parameters.
    integration_cfg : IntegrationConfig
        Integration configuration (quadrature, Monte Carlo, etc.)
    model_cfg : AnyModelConfig
        Model config (type of model used etc)
    train_cfg : TrainConfig
        Training hyperparameters (learning rate, steps, seed, etc.)
    sample_input : jnp.ndarray | None
        Sample input for model initialization.
    state : TrainState | None
        Continue from existing state.
    callback : Callable | None
        Called after each training step with metrics.

    Returns
    -------
    tuple[TrainState, list[TrainStepMetrics]]
        Final training state and metrics history
    """
    integrator = get_integrator(integration_cfg)
    model = build_model(model_cfg)
    algorithm = build_algorithm(algorithm_cfg, model)
    optimizer = get_optimizer(train_cfg)

    trainer = Trainer(
        method=algorithm,
        integrator=integrator,
        optimizer=optimizer,
        train_cfg=train_cfg,
    )

    return trainer.fit(sample_input=sample_input, state=state, callback=callback)
