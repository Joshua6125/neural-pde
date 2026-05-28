from typing import Callable, cast
from omegaconf import DictConfig

from src.loss_functions import PINNConfig, gPINNConfig, vPINNConfig, SLSConfig, FOSLSConfig, AlgorithmConfig
from src.models import MLPModelConfig, KANModelConfig, AnyModelConfig
from src.train import TrainConfig
from src.integration import MonteCarloConfig, QuadratureConfig, AnyIntegrationConfig

import optax
import jax.numpy as jnp
import jax


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
            spatial_dim=int(data.get("spatial_dim", 1)),
            x_min=float(data.get("x_min", 0.0)),
            x_max=float(data.get("x_max", 1.0)),
            t_min=float(data.get("t_min", 0.0)),
            t_max=float(data.get("t_max", 1.0)),
            boundary_samples=int(specific_data.get("boundary_samples", 1)),
            interior_samples=int(specific_data.get("interior_samples", 1)),
        )
    elif integration_type == "quadrature":
        return QuadratureConfig(
            spatial_dim=int(data.get("spatial_dim", 1)),
            x_min=float(data.get("x_min", 0.0)),
            x_max=float(data.get("x_max", 1.0)),
            t_min=float(data.get("t_min", 0.0)),
            t_max=float(data.get("t_max", 1.0)),
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


def build_model_config(
    spec: DictConfig,
    output_heads: dict[str, int],
) -> AnyModelConfig:

    kind = spec.get("name", None)

    if kind == "mlp":
        return build_mlp_config(spec, output_heads)
    if kind == "kan":
        return build_kan_config(spec, output_heads)

    raise ValueError(f"Unknown model type: {kind}")


def build_pinn_config(
    spec: DictConfig,
    model: AnyModelConfig,
    wave_functions: dict,
) -> PINNConfig:
    return PINNConfig(
        model=model,
        c=wave_functions.get("c", 1.0),
        f=wave_functions.get("f", 0.0),
        u0=wave_functions.get("u0", 0.0),
        ut0=wave_functions.get("ut0", 0.0),
        ic_weight=float(spec.get("ic_weight", 1.0)),
        bc_weight=float(spec.get("bc_weight", 1.0)),
    )


def build_gpinn_config(
    spec: DictConfig,
    model: AnyModelConfig,
    wave_functions: dict,
) -> gPINNConfig:
    return gPINNConfig(
        model=model,
        c=wave_functions.get("c", 1.0),
        f=wave_functions.get("f", 0.0),
        u0=wave_functions.get("u0", 0.0),
        ut0=wave_functions.get("ut0", 0.0),
        ic_weight=float(spec.get("ic_weight", 1.0)),
        bc_weight=float(spec.get("bc_weight", 1.0)),
        residual_grad_weight=float(spec.get("residual_grad_weight", 1e-2))
    )


def build_sls_config(
    model: AnyModelConfig,
    wave_functions: dict,
) -> SLSConfig:
    return SLSConfig(
        model=model,
        f=wave_functions.get("f", 0.0),
        g=wave_functions.get("g", 0.0),
        v0=wave_functions.get("v0", 0.0),
        sigma0=wave_functions.get("sigma0", 0.0),
        v_boundary=wave_functions.get("v_boundary", 0.0)
    )


def build_fosls_config(
    spec: DictConfig,
    model: AnyModelConfig,
    wave_functions: dict,
) -> FOSLSConfig:
    return FOSLSConfig(
        model=model,
        f=wave_functions.get("f", 0.0),
        g=wave_functions.get("g", 0.0),
        v0=wave_functions.get("v0", 0.0),
        sigma0=wave_functions.get("sigma0", 0.0),
        v_boundary=wave_functions.get("v_boundary", 0.0),
        ic_weight=float(spec.get("ic_weight", 1.0)),
    )


def build_vpinn_config(
    spec: DictConfig,
    model: AnyModelConfig,
    wave_functions: dict,
) -> vPINNConfig:
    return vPINNConfig(
        model=model,
        c=wave_functions.get("c", 1.0),
        f=wave_functions.get("f", 0.0),
        u0=wave_functions.get("u0", 0.0),
        ut0=wave_functions.get("ut0", 0.0),
        ic_weight=float(spec.get("ic_weight", 1.0)),
        bc_weight=float(spec.get("bc_weight", 1.0)),
        n_test_functions=int(spec.get("n_test_functions", 10))
    )


def build_method_config(
    data: DictConfig,
    model: AnyModelConfig,
    wave_functions: dict
) -> AlgorithmConfig:
    kind = data.get("name", None)

    if kind == "pinn":
        return build_pinn_config(data, model, wave_functions)
    if kind == "gpinn":
        return build_gpinn_config(data, model, wave_functions)
    if kind == "vpinn":
        return build_vpinn_config(data, model, wave_functions)
    if kind == "sls":
        return build_sls_config(model, wave_functions)
    if kind == "fosls":
        return build_fosls_config(data, model, wave_functions)

    raise ValueError(f"Unknown method type: {kind}")


def build_trainer_config(
    spec: DictConfig,
    learning_rate: optax.Schedule | None = None,
) -> TrainConfig:

    lr = learning_rate if not learning_rate is None else build_learning_rate_schedule(spec.get("learning_rate", {}))

    return TrainConfig(
        epochs=int(spec.get("epochs", 1)),
        learning_rate=lr,
        optimiser=str(spec.get("optimiser", "adamw")),
        seed=int(spec.get("seed", 42)),
        log_every=int(spec.get("log_every", 50)),
        use_jit=bool(spec.get("use_jit", True))
    )


def make_first_order_model(model_apply, method_kind: str):
    """
    Wraps the standard model_apply to always return the first-order vector
    representation: (v, sigma) using autodiff if it's a second-order model (e.g. PINN).
    If it's already a first-order model (SLS/FOSLS), it leaves it alone.
    """
    if method_kind in ["sls", "fosls"]:
        # output is already dict with v and sigma
        def wrapped_apply(params, t, x):
            out = model_apply(params, jnp.array([t, x]))
            return jnp.concatenate([out["v"], out["sigma"]], axis=-1)
        return wrapped_apply

    # For PINN/gPINN: model_apply returns dict with u, we need v = u_t, sigma = u_x
    def u_fn(params, t, x):
        return model_apply(params, jnp.array([t, x]))["u"][0]

    def wrapped_apply(params, t, x):
        v = jax.grad(u_fn, argnums=1)(params, t, x) # Differentiate t
        sigma = jax.grad(u_fn, argnums=2)(params, t, x) # Differentiate x
        return jnp.array([v, sigma])

    return wrapped_apply
