from dataclasses import dataclass
from typing import Any, Literal

import jax
import optax
from optax import Schedule


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for model training.

    Attributes
    ----------
    epochs : int
        Number of optimisation steps.
    learning_rate : optax.Schedule
        Optimiser learning rate.
    optimiser : str
        Optimiser name: 'adam' or 'sgd'.
    seed : int
        Random seed used for parameter initialisation.
    integration_seed : int | None
        Optional dedicated seed for integration sampling. If ``None``,
        integration randomness is derived from ``seed``.
    log_every : int
        Logging frequency in epochs.
    use_jit : bool
        Enable JIT compilation for train_step.
    """

    epochs: int = 1000
    learning_rate: Schedule = optax.constant_schedule(1e-3)
    optimiser: Literal["adam", "adamw", "sgd"] = "adamw"
    seed: int = 0
    integration_seed: int | None = None
    log_every: int = 100
    use_jit: bool = True

    def validate(self) -> None:
        assert self.epochs > 0, "epochs must be strictly positive"
        # NOTE: Need to check if scheduler automatically checks if learning rate is valid.
        # assert self.learning_rate > 0.0, "learning_rate must be strictly positive"
        assert self.log_every > 0, "log_every must be strictly positive"
        if self.integration_seed is not None:
            assert self.integration_seed >= 0, "integration_seed must be non-negative"


@dataclass(frozen=True)
class TrainState:
    """Mutable training values represented as an immutable dataclass."""

    step: int
    params: Any
    opt_state: optax.OptState
    integration_key: jax.Array

    def apply_gradients(
            self,
            grads: Any,
            optimiser: optax.GradientTransformation
        ) -> "TrainState":
        """Apply gradients and return updated state."""
        updates, opt_state = optimiser.update(grads, self.opt_state, self.params)
        params = optax.apply_updates(self.params, updates)

        return TrainState(
            step=self.step + 1,
            params=params,
            opt_state=opt_state,
            integration_key=self.integration_key,
        )


def get_optimiser(config: TrainConfig) -> optax.GradientTransformation:
    """Factory function for choosing optimisation method."""
    config.validate()
    if config.optimiser == "adam":
        return optax.adam(config.learning_rate)
    if config.optimiser == "adamw":
        return optax.adamw(config.learning_rate)
    if config.optimiser == "sgd":
        return optax.sgd(config.learning_rate)

    raise ValueError(
        f"Unknown optimiser: '{config.optimiser}'. Must be 'adam', 'adamw' or 'sgd'."
    )
