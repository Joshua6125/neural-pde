from typing import Callable, cast
from omegaconf import DictConfig

from src.loss_functions import PINNConfig, gPINNConfig, SLSConfig, FOSLSConfig
from src.models import MLPModelConfig, KANModelConfig, AnyModelConfig
from src.train import TrainConfig
from src.integration import MonteCarloConfig, QuadratureConfig, AnyIntegrationConfig

import optax
import jax.numpy as jnp


def build_integration_config(data: DictConfig) -> AnyIntegrationConfig:
    integration_type = (
        data.get("kind")
        or "monte_carlo" # Default to Monte Carlo
    )

    specific_data = data.get(integration_type, {})
    if not isinstance(specific_data, DictConfig):
        specific_data = DictConfig({})

    # choose config class and allowed keys
    if integration_type == "monte_carlo":
        return MonteCarloConfig(
            dim=int(data.get("dim", 2)),
            x_min=float(data.get("x_min", 0.0)),
            x_max=float(data.get("x_max", 0.0)),
            monte_carlo_boundary_samples=int(specific_data.get("monte_carlo_boundary_samples", 1)),
            monte_carlo_interior_samples=int(specific_data.get("monte_carlo_interior_samples", 1)),
            monte_carlo_seed=int(specific_data.get("monte_carlo_seed", 0))
        )
    elif integration_type == "quadrature":
        return QuadratureConfig(
            dim=int(data.get("dim", 2)),
            x_min=float(data.get("x_min", 0.0)),
            x_max=float(data.get("x_max", 0.0)),
            degree=int(specific_data.get("degree", 1)),
            adaptive_integration=bool(specific_data.get("adaptive_integration", False))
        )
    else:
        raise ValueError(f"Unrecognised integration type: {integration_type}")


def build_learning_rate_schedule(spec: DictConfig) -> optax.Schedule:
    """Build an optax schedule from a scalar or schedule specification."""

    # Object is a schedule
    if callable(spec):
        return cast(optax.Schedule, spec)

    # Constant schedule
    if isinstance(spec, (int, float)):
        return optax.constant_schedule(float(spec))

    kind = str(spec.get("kind", "constant").lower())

    if kind == "constant":
        return optax.constant_schedule(float(spec.get("value", spec.get("init_value", 1e-3))))

    if kind == "exponential_decay":
        return optax.exponential_decay(
            init_value=float(spec.get("init_value", 1e-3)),
            transition_steps=int(spec.get("transition_steps", 1000)),
            decay_rate=float(spec.get("decay_rate", 0.95)),
            staircase=bool(spec.get("staircase", True)),
            end_value=spec.get("end_value"),
            transition_begin=int(spec.get("transition_begin", 0)),
        )

    if kind == "cosine_decay":
        return optax.cosine_decay_schedule(
            init_value=float(spec.get("init_value", 1e-3)),
            decay_steps=int(spec.get("decay_steps", 1000)),
            alpha=float(spec.get("alpha", 0.0)),
        )

    raise ValueError(f"Unknown learning rate schedule kind: {kind}")


def build_mlp_config(
    spec: DictConfig,
    output_heads: dict[str, int],
) -> MLPModelConfig:

    if not output_heads:
        raise ValueError("Model must have output heads.")

    return MLPModelConfig(
        output_heads=output_heads,
        hidden_dim=int(spec.get("hidden_dim", 1)),
        num_layers=int(spec.get("num_layers", 1)),
    )


def build_kan_config(
    spec: DictConfig,
    output_heads: dict[str, int],
) -> KANModelConfig:

    if not output_heads:
        raise ValueError("Model must have output heads.")

    return KANModelConfig(
        output_heads=output_heads,
        hidden_dim=int(spec.get("hidden_dim", 1)),
        num_layers=int(spec.get("num_layers", 1)),
        input_dim=int(spec.get("input_dim", 1)),
    )


def build_pinn_config(
    spec: DictConfig,
    model: AnyModelConfig,
    c: float | Callable[[jnp.ndarray], jnp.ndarray] = 1.0,
    f: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
    u0: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
    ut0: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
) -> PINNConfig:

    return PINNConfig(
        model=model,
        c=c,
        f=f,
        u0=u0,
        ut0=ut0,
        ic_weight=float(spec.get("ic_weight", 1.0)),
        bc_weight=float(spec.get("bc_weight", 1.0)),
    )


def build_gpinn_config(
    spec: DictConfig,
    model: AnyModelConfig,
    c: float | Callable[[jnp.ndarray], jnp.ndarray] = 1.0,
    f: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
    u0: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
    ut0: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
) -> gPINNConfig:
    return gPINNConfig(
        model=model,
        c=c,
        f=f,
        u0=u0,
        ut0=ut0,
        ic_weight=float(spec.get("ic_weight", 1.0)),
        bc_weight=float(spec.get("bc_weight", 1.0)),
        residual_grad_weight=float(spec.get("residual_grad_weight", 1e-2))
    )


def build_sls_config(
    spec: DictConfig,
    model: AnyModelConfig,
    f: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
    g: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
    v0: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
    sigma0: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
    v_boundary: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
) -> SLSConfig:
    return SLSConfig(
        model=model,
        f=f,
        g=g,
        v0=v0,
        sigma0=sigma0,
        v_boundary=v_boundary,
    )


def build_fosls_config(
    spec: DictConfig,
    model: AnyModelConfig,
    f: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
    g: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
    v0: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
    sigma0: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
    v_boundary: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
) -> FOSLSConfig:
    return FOSLSConfig(
        model=model,
        f=f,
        g=g,
        v0=v0,
        sigma0=sigma0,
        v_boundary=v_boundary,
        ic_weight=float(spec.get("ic_weight", 1.0)),
    )


def build_trainer_config(
    spec: DictConfig,
    learning_rate: optax.Schedule | None,
) -> TrainConfig:

    lr = learning_rate if not learning_rate is None else build_learning_rate_schedule(spec.get("learning_rate", {}))

    return TrainConfig(
        epochs=int(spec.get("epochs", 1)),
        learning_rate=lr,
        optimiser=str(spec.get("optimiser", "adamw")),
        seed=int(spec.get("seed", 42)),
        integration_seed=spec.get("integration_seed", None),
        log_every=int(spec.get("log_every", 50)),
        use_jit=bool(spec.get("use_jit", True))
    )
