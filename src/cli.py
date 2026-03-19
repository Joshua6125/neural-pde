from typing import Callable

import jax.numpy as jnp

from .config import IntegrationConfig
from .loss_functions import LSLossConfig, PINNLossConfig
from .integration import get_integrator
from .models import AnyModelConfig, build_model_bundle
from .train.methods import get_training_method
from .train import (
    TrainConfig,
    TrainState,
    TrainStepMetrics,
    Trainer,
    get_optimizer,
)


def run_training(
        integration_cfg: IntegrationConfig,
        loss_cfg: LSLossConfig | PINNLossConfig,
        train_cfg: TrainConfig,
        model_cfg: AnyModelConfig,
        sample_input: jnp.ndarray | None = None,
        state: TrainState | None = None,
        callback: Callable[[TrainStepMetrics], None] | None = None,
    ) -> tuple[TrainState, list[TrainStepMetrics]]:
    """Build dependencies and execute training.


    """
    integrator = get_integrator(integration_cfg)
    model_bundle = build_model_bundle(model_cfg)
    method = get_training_method(loss_cfg=loss_cfg, model_bundle=model_bundle)
    optimizer = get_optimizer(train_cfg)

    trainer = Trainer(
        method=method,
        integrator=integrator,
        optimizer=optimizer,
        train_cfg=train_cfg,
    )

    return trainer.fit(sample_input=sample_input, state=state, callback=callback)
