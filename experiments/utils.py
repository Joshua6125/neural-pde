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
    model_name: str,
    spec: DictConfig,
    output_heads: dict[str, int],
) -> AnyModelConfig:
    if model_name == "mlp":
        return build_mlp_config(spec, output_heads)
    if model_name == "kan":
        return build_kan_config(spec, output_heads)

    raise ValueError(f"Unknown model type: {model_name}")


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
    integration_data: DictConfig | None = None,
) -> vPINNConfig:
    domain_min = None
    domain_max = None

    if integration_data:
        t_min = float(integration_data.get("t_min", 0.0))
        t_max = float(integration_data.get("t_max", 1.0))
        x_min = float(integration_data.get("x_min", 0.0))
        x_max = float(integration_data.get("x_max", 1.0))
        spatial_dim = int(integration_data.get("spatial_dim", 1))

        domain_min = jnp.array([t_min] + [x_min] * spatial_dim)
        domain_max = jnp.array([t_max] + [x_max] * spatial_dim)

    return vPINNConfig(
        model=model,
        c=wave_functions.get("c", 1.0),
        f=wave_functions.get("f", 0.0),
        u0=wave_functions.get("u0", 0.0),
        ut0=wave_functions.get("ut0", 0.0),
        ic_weight=float(spec.get("ic_weight", 1.0)),
        bc_weight=float(spec.get("bc_weight", 1.0)),
        n_test_functions=int(spec.get("n_test_functions", 10)),
        domain_min=domain_min,
        domain_max=domain_max,
    )


def build_method_config(
    method_name: str,
    data: DictConfig,
    model: AnyModelConfig,
    wave_functions: dict,
    integration_data: DictConfig | None = None,
) -> AlgorithmConfig:
    if method_name == "pinn":
        return build_pinn_config(data, model, wave_functions)
    if method_name == "gpinn":
        return build_gpinn_config(data, model, wave_functions)
    if method_name == "vpinn":
        return build_vpinn_config(data, model, wave_functions, integration_data)
    if method_name == "fosls":
        return build_fosls_config(data, model, wave_functions)

    raise ValueError(f"Unknown method type: {method_name}")


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
        use_jit=bool(spec.get("use_jit", True)),
        convergence_check=bool(spec.get("convergence_check", False)),
        convergence_window_size=int(spec.get("convergence_window_size", 1000)),
        convergence_rel_tol=float(spec.get("convergence_rel_tol", 1e-3))
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
            x_in = jnp.asarray(x_in)
            t = x_in[0]
            x = jnp.atleast_1d(x_in[1])

            def vector_field(time, u_val, args):
                model_input = jnp.concatenate([jnp.atleast_1d(time), x])
                out = model_apply(params, model_input)
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
                    y0=u0
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


def calculate_true_l2_error(
    model_apply_fn,
    params,
    method_kind,
    v_sol,
    sigma_sol,
    integrator,
):
    first_order_fn = make_first_order_model(
        params,
        model_apply_fn,
        method_kind,
    )

    def interior_error(x):
        pred = first_order_fn(x)

        v_pred = jnp.ravel(jnp.asarray(pred[0]))
        sigma_pred = jnp.ravel(jnp.asarray(pred[1:]))

        v_true = jnp.ravel(jnp.asarray(v_sol(x[0], x[1:])))
        sigma_true = jnp.ravel(jnp.asarray(sigma_sol(x[0], x[1:])))

        return (
            jnp.sum((v_true - v_pred) ** 2)
            + jnp.sum((sigma_true - sigma_pred) ** 2)
        )

    interior, _ = integrator.integrate(
        interior_func=jax.vmap(interior_error),
        boundary_func=lambda x, _: jnp.zeros(x.shape[0], dtype=x.dtype),
    )

    return float(jnp.sum(interior))


def calculate_true_v_error(
    model_apply_fn,
    params,
    method_kind,
    v_sol,
    sigma_sol,
    integrator,
):
    first_order_fn = make_first_order_model(
        params,
        model_apply_fn,
        method_kind,
    )

    def error_field(x):
        pred = first_order_fn(x)

        v_pred = jnp.ravel(jnp.asarray(pred[0]))
        sigma_pred = jnp.ravel(jnp.asarray(pred[1:]))

        v_true = jnp.ravel(jnp.asarray(v_sol(x[0], x[1:])))
        sigma_true = jnp.ravel(jnp.asarray(sigma_sol(x[0], x[1:])))

        return jnp.concatenate([
            v_true - v_pred,
            sigma_true - sigma_pred,
        ])

    jac_error = jax.jacfwd(error_field)

    def interior_error(x):
        e = error_field(x)
        J = jac_error(x)

        e_v = e[0]
        e_sigma = e[1:]

        # J[row, variable]
        dt_e_v = J[0, 0]

        grad_e_v = J[0, 1:]

        dt_e_sigma = J[1:, 0]

        div_e_sigma = jnp.trace(J[1:, 1:])

        residual_1 = dt_e_v - div_e_sigma
        residual_2 = dt_e_sigma - grad_e_v

        l2_part = e_v**2 + jnp.sum(e_sigma**2)
        graph_part = residual_1**2 + jnp.sum(residual_2**2)

        return l2_part + graph_part

    def boundary_error(x_boundary, normal_vector):
        def point_error(x):
            e = error_field(x)

            e_v = e[0]
            e_sigma = e[1:]

            return e_v**2 + jnp.sum(e_sigma**2)

        errors = jax.vmap(point_error)(x_boundary)
        is_initial_face = normal_vector[:, 0] < 0
        return jnp.where(is_initial_face, errors, 0.0)

    interior, boundary = integrator.integrate(
        interior_func=jax.vmap(interior_error),
        boundary_func=boundary_error,
    )

    return float(jnp.sum(interior) + jnp.sum(boundary))


def calculate_dof(
    input_dim: int,
    model_cfg: AnyModelConfig
) -> int:
    if isinstance(model_cfg, MLPConfig):
        '''
        Assume:
        input dimension: p
        Output dimension: q
        hidden layers: n
        neurons in each hidden layer: k

        input -> hidden
        weights: pk
        biases: k

        hidden -> hidden
        n - 1 transitions with each k^2 + k so (n - 1)(k^2 + k)

        hidden -> output
        weights: kq
        biases: q

        total:
        pk + k + (n - 1)(k^2 + k) + kq + q
        '''
        p = input_dim
        q = len(model_cfg.output_heads)
        n = model_cfg.num_layers
        k = model_cfg.hidden_dim

        return p*k + k + (n - 1)*(k**2 + k) + k*q + q

    if isinstance(model_cfg, KANConfig):
        '''
        Assume:
        input dimension: p
        Output dimension: q
        hidden layers: n
        neurons in each hidden layer: k
        degree: d

        Each function of degree d has (d + 1) coeffcients.

        Number of edges is pk + (n - 1)k^2 + kq

        total:
        (d + 1)(pk + (n - 1)k^2 + kq) + kn + q
        '''
        p = input_dim
        q = len(model_cfg.output_heads)
        n = model_cfg.num_layers
        k = model_cfg.hidden_dim
        d = model_cfg.degree

        return (d + 1)*(p*k + (n - 1)*k**2 + k*q) + k*n + q


