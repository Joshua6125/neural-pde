from typing import Callable, cast
from omegaconf import DictConfig

from src.loss_functions import PINNConfig, gPINNConfig, vPINNConfig, FOSLSConfig, AlgorithmConfig, FOSLSLoss
from src.models import MLPConfig, KANConfig, AnyModelConfig
from src.train import TrainConfig
from src.integration import MonteCarloConfig, QuadratureConfig, AnyIntegrationConfig, NDCubeIntegration

import optax
import jax.numpy as jnp
import jax
import diffrax


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
) -> MLPConfig:

    if not output_heads:
        raise ValueError("Model must have output heads.")

    return MLPConfig(
        output_heads=output_heads,
        hidden_dim=int(spec.get("hidden_dim", 1)),
        num_layers=int(spec.get("num_layers", 1)),
    )


def build_kan_config(
    spec: DictConfig,
    output_heads: dict[str, int],
) -> KANConfig:

    if not output_heads:
        raise ValueError("Model must have output heads.")

    return KANConfig(
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
        max_training_time=float(spec.get("max_training_time", 100.0)),
        learning_rate=lr,
        optimiser=str(spec.get("optimiser", "adamw")),
        seed=int(spec.get("seed", 42)),
        log_every=int(spec.get("log_every", 50)),
        use_jit=bool(spec.get("use_jit", True))
    )


def make_first_order_model(params, model_apply, method_kind: str):
    """
    Wraps the standard model_apply to always return the first-order vector (v, sigma)
    """
    if method_kind in ["fosls"]:
        # output is already dict with v and sigma
        def fosls_apply(x: jnp.ndarray) -> jnp.ndarray:
            out = model_apply(params, x)
            return jnp.concatenate([jnp.atleast_1d(out["v"]), jnp.atleast_1d(out["sigma"])], axis=-1)
        return fosls_apply

    # For other models model_apply returns dict with u, we need v = u_t, sigma = u_x
    def u_fn(x):
        return jnp.squeeze(model_apply(params, x)["u"])

    def wrapped_apply(x):
        grad = jax.grad(u_fn)(x)
        v = grad[0]
        sigma = grad[1:]
        return jnp.concatenate([jnp.atleast_1d(v), jnp.atleast_1d(sigma)], axis=-1)

    return wrapped_apply


def make_second_order_model(model_apply, method_kind: str, u0_fn=None):
    """
    Wraps the model_apply to always return the second-order variable (u).
    """
    if u0_fn is None:
        u0_fn = lambda x: jnp.zeros_like(x)

    if method_kind in ["fosls"]:
        def fosls_apply(params, x_in):
            t, x = x_in[0], x_in[1:]

            def vector_field(time, u_val, args):
                out = model_apply(params, jnp.array([time, x]))
                return jnp.atleast_1d(out["v"])

            # 1) Evaluate the initial condition at t=0 instead of t
            u0 = u0_fn((jnp.zeros_like(jnp.atleast_1d(x[0])), x))

            def return_initial(*_):
                return u0

            def run_integration(*_):
                term = diffrax.ODETerm(vector_field)
                solver = diffrax.Tsit5()
                sol = diffrax.diffeqsolve(
                    term,
                    solver,
                    t0=0.0,
                    t1=t,
                    dt0=None,
                    stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8),
                    y0=u0,
                    max_steps=1000
                )
                assert sol.ys is not None
                return sol.ys[-1]

            u_final = jax.lax.cond(t == 0.0, return_initial, run_integration)
            return jnp.atleast_1d(u_final)

        return fosls_apply

    # For PINN/gPINN/vPINN: model_apply returns dict with u directly
    def u_fn(params, x_in):
        return jnp.squeeze(model_apply(params, x_in)["u"])

    def wrapped_apply(params, x):
        u_val = u_fn(params, x)
        return jnp.atleast_1d(u_val)

    return wrapped_apply


def create_evaluation_domain(cfg: DictConfig) -> jnp.ndarray:
    """Creates a deterministic space-time grid for evaluation."""
    t_min = float(cfg.get("integration", {}).get("t_min", 0.0))
    t_max = float(cfg.get("integration", {}).get("t_max", 1.0))
    x_min = float(cfg.get("integration", {}).get("x_min", 0.0))
    x_max = float(cfg.get("integration", {}).get("x_max", 1.0))

    t_grid = jnp.linspace(t_min, t_max, cfg.plot_params.get("t_grid", 100))
    x_grid = jnp.linspace(x_min, x_max, cfg.plot_params.get("x_grid", 100))

    T, X = jnp.meshgrid(t_grid, x_grid, indexing='ij')

    return jnp.stack([T.flatten(), X.flatten()], axis=-1)


def calculate_fosls_norm(
    model_apply_fn: Callable,
    params,
    method_kind: str,
    f_fn: Callable,
    g_fn: Callable,
    v0_fn: Callable,
    sigma0_fn: Callable,
    integrator: NDCubeIntegration,
    ic_weight: float = 1.0,
) -> float:
    """
    Evaluates the FOSLS norm using the provided integrator.
    """

    first_order_fn = make_first_order_model(params, model_apply_fn, method_kind)

    loss_obj = FOSLSLoss(
        model=first_order_fn,
        f=f_fn,
        g=g_fn,
        v0=v0_fn,
        sigma0=sigma0_fn,
        v_boundary=0.0,
        ic_weight=ic_weight,
    )

    interior_loss, boundary_loss = integrator.integrate(
        interior_func=loss_obj.loss_interior,
        boundary_func=loss_obj.loss_boundary
    )

    return float(jnp.sum(interior_loss) + jnp.sum(boundary_loss))
