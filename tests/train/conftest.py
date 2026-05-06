"""Shared fixtures for training module tests."""

from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

from src.integration import NDCubeIntegration
from src.loss_functions import SLS, SLSConfig, PINN, PINNConfig
from src.models import NeuralNetModelConfig, build_model
from src.train import TrainConfig, TrainingMethod


class MockTrainingMethod(TrainingMethod):
    """Minimal differentiable method used for Trainer unit tests."""

    def init_params(self, rng_key: jax.Array, sample_input: jnp.ndarray) -> Any:
        del rng_key, sample_input
        return {"w": jnp.array(0.0)}

    def loss_functions(self, params: Any):
        w = params["w"]

        def interior_loss(x: jnp.ndarray) -> jnp.ndarray:
            del x
            return jnp.ones((4,)) * (w - 1.0) ** 2

        def boundary_loss(x: jnp.ndarray, normal: jnp.ndarray) -> jnp.ndarray:
            del x, normal
            return jnp.ones((4,)) * 0.5 * (w + 1.0) ** 2

        return interior_loss, boundary_loss


class MockMalformedMethod(TrainingMethod):
    """Method with malformed loss_functions output for error-path tests."""

    def init_params(self, rng_key: jax.Array, sample_input: jnp.ndarray) -> Any:
        del rng_key, sample_input
        return {"w": jnp.array(0.0)}

    def loss_functions(self, params: Any): # type: ignore ; This is meant to give a warning.
        del params
        return (lambda x: x,)  # wrong arity/structure (len == 1)


class MockNaNMethod(TrainingMethod):
    """Method returning non-finite losses to test propagation behaviour."""

    def init_params(self, rng_key: jax.Array, sample_input: jnp.ndarray) -> Any:
        del rng_key, sample_input
        return {"w": jnp.array(0.0)}

    def loss_functions(self, params: Any):
        del params

        def interior_loss(x: jnp.ndarray) -> jnp.ndarray:
            del x
            return jnp.array([jnp.nan, jnp.nan])

        def boundary_loss(x: jnp.ndarray, normal: jnp.ndarray) -> jnp.ndarray:
            del x, normal
            return jnp.array([0.0, 0.0])

        return interior_loss, boundary_loss


class DeterministicTestIntegrator(NDCubeIntegration):
    """Simple deterministic integrator for Trainer unit tests."""

    def __init__(self) -> None:
        self._interior_points = jnp.array(
            [[0.2, 0.1], [0.4, 0.3], [0.6, 0.5], [0.8, 0.7]]
        )
        self._boundary_points = jnp.array(
            [[0.0, 0.2], [0.0, 0.8], [0.5, 0.0], [0.5, 1.0]]
        )
        self._boundary_normals = jnp.array(
            [[-1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
        )

    def integrate_interior(self, func):
        values = func(self._interior_points)
        return jnp.mean(values)

    def integrate_boundary(self, func):
        values = func(self._boundary_points, self._boundary_normals)
        return jnp.mean(values)


@pytest.fixture
def sample_input_vector_2d():
    """Single input point [t, x] used for model initialisation paths."""
    return jnp.array([0.5, 0.25])


@pytest.fixture
def train_cfg_default():
    """Default deterministic training configuration for unit tests."""
    return TrainConfig(
        epochs=5,
        learning_rate=optax.constant_schedule(1e-2),
        optimiser="adam",
        seed=7,
        log_every=1,
        use_jit=False
    )


@pytest.fixture
def train_cfg_with_integration_seed():
    """Training config with explicit integration seed override."""
    return TrainConfig(
        epochs=5,
        learning_rate=optax.constant_schedule(1e-2),
        optimiser="adam",
        seed=7,
        integration_seed=99,
        log_every=1,
        use_jit=False,
    )


@pytest.fixture
def train_cfg_short_jit():
    """Short run config with JIT enabled."""
    return TrainConfig(
        epochs=2,
        learning_rate=optax.constant_schedule(1e-2),
        optimiser="adamw",
        seed=3,
        log_every=1,
        use_jit=True
    )


@pytest.fixture
def optimiser_adam(train_cfg_default):
    """Adam optimiser built from default train config."""
    return optax.adam(train_cfg_default.learning_rate)


@pytest.fixture
def mock_training_method():
    """Valid method double for Trainer unit tests."""
    return MockTrainingMethod()


@pytest.fixture
def mock_malformed_method():
    """Invalid method double for Trainer error-path tests."""
    return MockMalformedMethod()


@pytest.fixture
def mock_nan_method():
    """Method returning non-finite losses."""
    return MockNaNMethod()


@pytest.fixture
def deterministic_integrator():
    """Deterministic integrator used for trainer unit behaviour tests."""
    return DeterministicTestIntegrator()


@pytest.fixture
def callback_recorder():
    """Collect callback metrics invocations for assertions."""
    recorded = []

    def _callback(metric):
        recorded.append(metric)

    return recorded, _callback


@pytest.fixture
def real_pinn_method():
    """Real PINN method with a lightweight neural network model."""
    cfg = PINNConfig(
        model=NeuralNetModelConfig(hidden_dim=8, num_layers=2, output_heads={"u": 1}),
        c=1.0,
        f=0.0,
        u0=0.0,
        ut0=0.0,
        ic_weight=1.0,
        bc_weight=1.0,
    )
    model = build_model(cfg.model)
    return PINN(model=model, config=cfg)


@pytest.fixture
def real_sls_method():
    """Real SLS method with a lightweight neural network model."""
    cfg = SLSConfig(
        model=NeuralNetModelConfig(hidden_dim=8, num_layers=2, output_heads={"v": 1, "sigma": 1}),
        f=0.0,
        g=0.0,
        v0=0.0,
        sigma0=0.0,
        v_boundary=0.0,
    )
    model = build_model(cfg.model)
    return SLS(model=model, config=cfg)
