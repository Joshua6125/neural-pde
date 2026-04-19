from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from ..integration import NDCubeIntegration
from .base import TrainingMethod
from .state import TrainConfig, TrainState


@dataclass(frozen=True)
class TrainStepMetrics:
    """Metrics tracked at each optimization step."""

    step: int
    total_loss: float
    interior_loss: float
    boundary_loss: float


class Trainer:
    """Generic training loop independent of method and integration choices."""

    def __init__(
            self,
            method: TrainingMethod,
            integrator: NDCubeIntegration,
            optimizer: optax.GradientTransformation,
            train_cfg: TrainConfig,
        ):
        self.method = method
        self.integrator = integrator
        self.optimizer = optimizer
        self.train_cfg = train_cfg
        self.train_cfg.validate()

        self._train_step_fn = self._train_step_impl
        if self.train_cfg.use_jit:
            self._train_step_fn = jax.jit(self._train_step_impl)

    def init_state(self, sample_input: jnp.ndarray) -> TrainState:
        """Initialize parameters, optimizer state, and RNG."""
        root_key = jr.PRNGKey(self.train_cfg.seed)
        root_key, init_key, derived_integration_key = jr.split(root_key, 3)
        params = self.method.init_params(init_key, sample_input)
        opt_state = self.optimizer.init(params)
        integration_key = (
            jr.PRNGKey(self.train_cfg.integration_seed)
            if self.train_cfg.integration_seed is not None
            else derived_integration_key
        )
        return TrainState(
            step=0,
            params=params,
            opt_state=opt_state,
            integration_key=integration_key,
        )

    def _loss_with_aux(
            self,
            params: Any,
            integration_key: jax.Array,
        ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jax.Array]]:
        interior_fn, boundary_fn = self.method.loss_functions(params)
        total, interior, boundary, next_key = self.integrator.integrate_with_key(
            interior_fn,
            boundary_fn,
            integration_key,
        )
        return total, (interior, boundary, next_key)

    def _train_step_impl(
            self,
            params: Any,
            opt_state: optax.OptState,
            integration_key: jax.Array,
        ) -> tuple[Any, optax.OptState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jax.Array]:
        """Core train step that can be JIT-compiled."""
        fun = lambda p: self._loss_with_aux(p, integration_key)
        value, grads = jax.value_and_grad(fun, has_aux=True)(params)
        total_loss, (interior_loss, boundary_loss, next_integration_key) = value

        updates, next_opt_state = self.optimizer.update(grads, opt_state, params)
        next_params = optax.apply_updates(params, updates)
        return (
            next_params,
            next_opt_state,
            total_loss,
            interior_loss,
            boundary_loss,
            next_integration_key,
        )

    def train_step(self, state: TrainState) -> tuple[TrainState, TrainStepMetrics]:
        """Run one optimization step and return updated state and metrics."""
        params, opt_state, total_loss, interior_loss, boundary_loss, integration_key = self._train_step_fn(
            state.params,
            state.opt_state,
            state.integration_key,
        )

        state = TrainState(
            step=state.step + 1,
            params=params,
            opt_state=opt_state,
            integration_key=integration_key,
        )

        metrics = TrainStepMetrics(
            step=state.step,
            total_loss=float(total_loss),
            interior_loss=float(interior_loss),
            boundary_loss=float(boundary_loss),
        )
        return state, metrics

    def fit(
            self,
            sample_input: jnp.ndarray | None = None,
            state: TrainState | None = None,
            callback: Callable[[TrainStepMetrics], None] | None = None,
        ) -> tuple[TrainState, list[TrainStepMetrics]]:
        """Run training for the configured number of epochs.

        Parameters
        ----------
        sample_input : jnp.ndarray | None
            Required if ``state`` is not provided. Used for model initialisation.
        state : TrainState | None
            Existing state for continuing training.
        callback : Callable[[TrainStepMetrics], None] | None
            Optional callback invoked each epoch.
        """
        if state is None and sample_input is None:
            raise ValueError("Must provide either an initial state or sample_input.")

        if state is None:
            assert sample_input is not None
            state = self.init_state(sample_input)

        history: list[TrainStepMetrics] = []
        for epoch in range(self.train_cfg.epochs):
            state, metrics = self.train_step(state)
            if epoch % self.train_cfg.log_every == 0:
                print(f"Training progress: {epoch}/{self.train_cfg.epochs}")
                history.append(metrics)

            if callback is not None:
                callback(metrics)

        return state, history
