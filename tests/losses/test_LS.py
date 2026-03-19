import jax.numpy as jnp
import pytest

from src.cli import run_training
from src.config import QuadratureConfig
from src.loss_functions import LSLossConfig
from src.models import LSModelConfig, NeuralNetModelConfig, PINNModelConfig
from src.train import TrainConfig


def test_ls_training_smoke_runs_two_steps():
    integration_cfg = QuadratureConfig(dim=2, gauss_legendre_degree=3)
    loss_cfg = LSLossConfig()
    train_cfg = TrainConfig(epochs=2, use_jit=False)
    model_cfg = LSModelConfig(
        v_model=NeuralNetModelConfig(hidden_dim=8, num_layers=2, output_dim=1),
        sigma_model=NeuralNetModelConfig(hidden_dim=8, num_layers=2, output_dim=1),
    )

    state, history = run_training(
        integration_cfg=integration_cfg,
        loss_cfg=loss_cfg,
        train_cfg=train_cfg,
        model_cfg=model_cfg,
        sample_input=jnp.zeros((2,)),
    )

    assert state.step == 2
    assert len(history) == 2


def test_ls_rejects_pinn_model_config():
    integration_cfg = QuadratureConfig(dim=2, gauss_legendre_degree=3)
    loss_cfg = LSLossConfig()
    train_cfg = TrainConfig(epochs=1, use_jit=False)
    model_cfg = PINNModelConfig(
        u_model=NeuralNetModelConfig(hidden_dim=8, num_layers=2, output_dim=1)
    )

    with pytest.raises(ValueError, match="LS loss config requires LS model config/bundle"):
        run_training(
            integration_cfg=integration_cfg,
            loss_cfg=loss_cfg,
            train_cfg=train_cfg,
            model_cfg=model_cfg,
            sample_input=jnp.zeros((2,)),
        )
