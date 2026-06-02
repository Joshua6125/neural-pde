import inspect
import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

import time
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from ..integration import NDCubeIntegration
from .base import TrainingMethod
from .state import TrainConfig, TrainState


@dataclass(frozen=True)
class TrainStepMetrics:
    """Metrics tracked at each optimisation step."""

    step: int
    total_loss: float
    interior_loss: float
    boundary_loss: float
    training_time: float


class Trainer:
    """Generic training loop independent of method and integration choices."""

    def __init__(
            self,
            method: TrainingMethod,
            integrator: NDCubeIntegration,
            optimiser: optax.GradientTransformation,
            train_cfg: TrainConfig,
        ):
        self.method = method
        self.integrator = integrator
        self.optimiser = optimiser
        self.train_cfg = train_cfg
        self.train_cfg.validate()

        self._train_step_fn = self._train_step_impl
        if self.train_cfg.use_jit:
            self._train_step_fn = jax.jit(self._train_step_impl)

    def init_state(self, sample_input: jnp.ndarray) -> TrainState:
        """Initialize parameters, optimiser state, and RNG."""
        root_key = jr.PRNGKey(self.train_cfg.seed)
        root_key, init_key, derived_integration_key = jr.split(root_key, 3)
        params = self.method.init_params(init_key, sample_input)
        opt_state = self.optimiser.init(params)
        return TrainState(
            step=0,
            params=params,
            opt_state=opt_state,
            integration_key=derived_integration_key,
        )

    def _loss_with_aux(
            self,
            params: Any,
            integration_key: jax.Array,
        ) -> tuple[jnp.ndarray, tuple[Any, Any, jax.Array]]:
        interior_fn, boundary_fn = self.method.loss_functions(params)
        interior, boundary = self.integrator.integrate(
            interior_fn,
            boundary_fn,
            integration_key,
        )
        total = self.method.aggregate_loss(interior, boundary)
        # Split key to ensure next time we have other sample points
        next_key, _ = jr.split(integration_key)

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

        updates, next_opt_state = self.optimiser.update(grads, opt_state, params)
        next_params = optax.apply_updates(params, updates)
        return (
            next_params,
            next_opt_state,
            total_loss,
            interior_loss,
            boundary_loss,
            next_integration_key,
        )

    @staticmethod
    def _tree_sum(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        return sum(float(jnp.sum(leaf)) for leaf in leaves) if leaves else 0.0

    @staticmethod
    def _invoke_callback(
            callback: Callable[..., None],
            metrics: TrainStepMetrics,
            previous_state: TrainState,
        ) -> None:
        """Call callbacks that accept either one or two positional arguments."""
        signature = inspect.signature(callback)
        positional_params = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]

        if any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in signature.parameters.values()):
            callback(metrics, previous_state)
        elif len(positional_params) >= 2:
            callback(metrics, previous_state)
        elif len(positional_params) == 1:
            callback(metrics)
        else:
            callback()

    def _has_converged(self, loss_window: deque[float]) -> bool:
        """Check whether the rolling loss window has flattened out."""
        if not loss_window:
            return False

        losses = tuple(loss_window)
        if not all(math.isfinite(loss) for loss in losses):
            return False

        mean_loss = sum(losses) / len(losses)
        mean_abs_loss = sum(abs(loss) for loss in losses) / len(losses)
        tolerance = self.train_cfg.convergence_rel_tol * mean_abs_loss
        max_deviation = max(abs(loss - mean_loss) for loss in losses)
        # print(f"convergence: {max_deviation} - {tolerance}")
        return max_deviation <= tolerance

    def fit(
            self,
            sample_input: jnp.ndarray | None = None,
            state: TrainState | None = None,
            callback: Callable[..., None] | None = None,
        ) -> tuple[TrainState, list[TrainStepMetrics]]:
        """Run training for the configured number of epochs.

        Parameters
        ----------
        sample_input : jnp.ndarray | None
            Required if ``state`` is not provided. Used for model initialisation.
        state : TrainState | None
            Existing state for continuing training.
        callback : Callable[..., None] | None
            Optional callback invoked each epoch.
        """
        if state is None and sample_input is None:
            raise ValueError("Must provide either an initial state or sample_input.")

        if state is None:
            assert sample_input is not None
            state = self.init_state(sample_input)

        if self.train_cfg.convergence_check and self.train_cfg.convergence_window_size <= 0:
            raise ValueError("convergence_window_size must be positive when convergence_check is enabled")

        loss_window: deque[float] = deque(maxlen=self.train_cfg.convergence_window_size)

        start_time = time.time()
        history: list[TrainStepMetrics] = []
        for epoch in range(1, self.train_cfg.epochs + 1):
            train_time_start = time.time() - state.total_training_time

            previous_state = state
            params, opt_state, total_loss, interior_loss, boundary_loss, integration_key = self._train_step_fn(
                state.params,
                state.opt_state,
                state.integration_key,
            )

            total_training_time = time.time() - train_time_start if epoch > 1 else 0.0

            state = TrainState(
                step=state.step + 1,
                params=params,
                opt_state=opt_state,
                integration_key=integration_key,
                total_training_time=total_training_time,
            )

            should_log = self.train_cfg.log_every > 0 and epoch % self.train_cfg.log_every == 0
            # print(epoch, should_log)
            if should_log:
                metrics = TrainStepMetrics(
                    step=epoch,
                    total_loss=float(total_loss),
                    interior_loss=self._tree_sum(interior_loss),
                    boundary_loss=self._tree_sum(boundary_loss),
                    training_time=total_training_time,
                )

                print(f"Training progress: {epoch}/{self.train_cfg.epochs},",
                        f"{total_training_time:.2f}/{self.train_cfg.max_training_time:.2f}s",
                        f"({time.time() - start_time:.2f}s total elapsed)")

                history.append(metrics)

                if callback is not None:
                    self._invoke_callback(callback, metrics, previous_state)

            if self.train_cfg.convergence_check:
                loss_window.append(float(total_loss))
                if len(loss_window) > self.train_cfg.convergence_window_size:
                    loss_window.popleft()

                if len(loss_window) == self.train_cfg.convergence_window_size and self._has_converged(loss_window):
                    return state, history

            # Stop training if max training time has been hit.
            if state.total_training_time > self.train_cfg.max_training_time:
                return state, history

        return state, history
