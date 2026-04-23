import jax.numpy as jnp

from dataclasses import dataclass, field
from typing import Iterable
from src.models import (
    NeuralNetModelConfig,
    KANModelConfig,
    AnyModelConfig,
)
from src.loss_functions import (
    PINNConfig,
    LSConfig,
    AlgorithmConfig
)
from src.integration import MonteCarloConfig
from src.train import TrainConfig


@dataclass
class ProblemConfig:
    L: float = 1.0
    T: float = 1.0
    c: float = 1.0
    methods: list[str] = field(default_factory=lambda: ["pinn", "ls"])
    models: list[str] = field(default_factory=lambda: ["neural", "kan"])
    hidden_dim: int = 32
    num_layers: int = 3


@dataclass(frozen=True)
class ExperimentCombination:
    method: str
    model: str
    label: str
    model_config: AnyModelConfig
    algorithm_config: AlgorithmConfig


def get_problem_config() -> ProblemConfig:
    return ProblemConfig(
        L=1.0,
        T=1.0,
        c=1.0,
    )


def _canonical_name(name: str) -> str:
    return name.strip().lower()


def _ensure_supported(kind: str, requested: Iterable[str], supported: set[str]) -> list[str]:
    normalized = [_canonical_name(item) for item in requested]
    unsupported = sorted(set(normalized) - supported)
    if unsupported:
        raise ValueError(
            f"Unsupported {kind}(s): {unsupported}. "
            f"Supported {kind}s: {sorted(supported)}"
        )
    return normalized


def get_pinn_model_config(
    problem_config: ProblemConfig,
    model_type: str = "neural",
) -> AnyModelConfig:
    model_type = _canonical_name(model_type)
    if model_type == "neural":
        return NeuralNetModelConfig(
            hidden_dim=problem_config.hidden_dim,
            num_layers=problem_config.num_layers,
            output_heads={"u": 1},
        )
    if model_type == "kan":
        return KANModelConfig(
            hidden_dim=problem_config.hidden_dim,
            num_layers=problem_config.num_layers,
            output_heads={"u": 1},
        )

    raise ValueError("Requested model type must be neural or kan.")


def get_ls_model_config(
    problem_config: ProblemConfig,
    model_type: str = "neural",
) -> AnyModelConfig:
    pinn_model = get_pinn_model_config(problem_config, model_type=model_type)
    if model_type == "neural":
        return NeuralNetModelConfig(
            hidden_dim=pinn_model.hidden_dim,
            num_layers=pinn_model.num_layers,
            output_heads={"v": 1, "sigma": 1},
        )
    if model_type == "kan":
        return KANModelConfig(
            hidden_dim=pinn_model.hidden_dim,
            num_layers=pinn_model.num_layers,
            output_heads={"v": 1, "sigma": 1},
            input_dim=2,
        )

    raise ValueError("Requested model type must be neural or kan.")


def get_pinn_config(
    problem_config: ProblemConfig,
    model_type: str = "neural",
) -> PINNConfig:
    return PINNConfig(
        model=get_pinn_model_config(problem_config, model_type=model_type),
        f=lambda v: source_function(
            jnp.array(v[0]), jnp.array(v[1]), problem_config
        ),
        u0=lambda v: analytical_solution(
            jnp.array(v[0]), jnp.array(v[1]), problem_config
        ),
        ut0=lambda v: analytical_solution_t(
            jnp.array(v[0]), jnp.array(v[1]), problem_config
        ),
        c=problem_config.c,
        ic_weight=10.0,
        bc_weight=100.0,
    )

def get_ls_config(
    problem_config: ProblemConfig,
    model_type: str = "neural",
) -> LSConfig:
    return LSConfig(
        model=get_ls_model_config(problem_config, model_type=model_type),
        f=lambda v: source_function(
            jnp.array(v[0]), jnp.array(v[1]), problem_config
        ),
        g=lambda v: zero_vector_source(
            jnp.array(v[0]), jnp.array(v[1]), problem_config
        ),
        v0=lambda v: analytical_solution_t(
            jnp.array(v[0]), jnp.array(v[1]), problem_config
        ),
        sigma0=lambda v: jnp.array([
            analytical_solution_x(
                jnp.array(v[0]), jnp.array(v[1]), problem_config
            )
        ]),
    )


def get_experiment_combination(
    problem_config: ProblemConfig,
    method: str,
    model: str,
) -> ExperimentCombination:
    method_name = _canonical_name(method)
    model_name = _canonical_name(model)

    _ensure_supported("method", [method_name], {"pinn", "ls"})
    _ensure_supported("model", [model_name], {"neural", "kan"})

    label = f"{method_name.upper()}-{model_name.upper()}"
    if method_name == "pinn":
        model_cfg = get_pinn_model_config(problem_config, model_type=model_name)
        algorithm_cfg = get_pinn_config(problem_config, model_type=model_name)
    elif method_name == "ls":
        model_cfg = get_ls_model_config(problem_config, model_type=model_name)
        algorithm_cfg = get_ls_config(problem_config, model_type=model_name)
    else:
        raise ValueError(f"Unsupported method: {method_name}")

    return ExperimentCombination(
        method=method_name,
        model=model_name,
        label=label,
        model_config=model_cfg,
        algorithm_config=algorithm_cfg,
    )


def build_experiment_combinations(problem_config: ProblemConfig) -> list[ExperimentCombination]:
    supported_methods = {"pinn", "ls"}
    supported_models = {"neural", "kan"}

    selected_methods = _ensure_supported("method", problem_config.methods, supported_methods)
    selected_models = _ensure_supported("model", problem_config.models, supported_models)

    combinations: list[ExperimentCombination] = []
    for method in selected_methods:
        for model in selected_models:
            combinations.append(
                get_experiment_combination(
                    problem_config=problem_config,
                    method=method,
                    model=model,
                )
            )

    return combinations


def get_integration_config(
    problem_config: ProblemConfig,
) -> MonteCarloConfig:
    return MonteCarloConfig(
        dim=2,
        x_min=-problem_config.L,
        x_max=problem_config.L,
        monte_carlo_interior_samples=1600,
        monte_carlo_boundary_samples=1600,
        monte_carlo_seed=42,
    )


def get_training_config() -> TrainConfig:
    return TrainConfig(
        epochs=1000,
        learning_rate=1e-4,
        optimiser="adamw",
        use_jit=True,
        seed=42,
        log_every=50,
    )


def analytical_solution(
    t: jnp.ndarray,
    x: jnp.ndarray,
    config: ProblemConfig,
) -> jnp.ndarray:
    """
    Analytical solution.

    u(t, x) = sin(π*t/T) * sin(π*(x+L)/(2L))
    """
    # Handle floating-point precision at t=T (where sin(π) should be exactly 0)
    t_normalized = t / config.T
    t_at_boundary = jnp.isclose(t_normalized, 1.0, atol=1e-10)

    t_term = jnp.sin(jnp.pi * t / config.T)
    x_term = jnp.sin(jnp.pi * (x + config.L) / (2 * config.L))

    # If t is at the boundary (t=T), return 0 regardless of spatial terms
    result = t_term * x_term
    return jnp.where(t_at_boundary, 0.0, result)


def analytical_solution_t(
    t: jnp.ndarray,
    x: jnp.ndarray,
    config: ProblemConfig,
) -> jnp.ndarray:
    """
    Time derivative of analytical solution: ∂u/∂t.

    u_t(t, x) = (π/T) * cos(π*t/T) * sin(π*(x+L)/(2L))

    This is the initial velocity constraint at t=0.
    """
    t_term = jnp.cos(jnp.pi * t / config.T) * (jnp.pi / config.T)
    x_term = jnp.sin(jnp.pi * (x + config.L) / (2 * config.L))
    return t_term * x_term


def analytical_solution_x(
    t: jnp.ndarray,
    x: jnp.ndarray,
    config: ProblemConfig,
) -> jnp.ndarray:
    """Spatial derivative ∂u/∂x of analytical solution."""
    t_term = jnp.sin(jnp.pi * t / config.T)
    x_term = (
        jnp.cos(jnp.pi * (x + config.L) / (2 * config.L))
        * (jnp.pi / (2 * config.L))
    )
    return t_term * x_term


def source_function(
    t: jnp.ndarray,
    x: jnp.ndarray,
    config: ProblemConfig,
) -> jnp.ndarray:
    """
    Source term computed analytically from u_tt - c²∇²u = f.

        For u(t,x) = sin(πt/T) * sin(π(x+L)/2L):

        u_tt = -π²/T² * sin(πt/T) * sin(π(x+L)/2L)
        u_xx = -π²/(4L²) * sin(πt/T) * sin(π(x+L)/2L)

        f = u_tt - c²u_xx
            = (-π²/T² + c²π²/(4L²)) * sin(πt/T) * sin(π(x+L)/2L)
            = (-π²/T² + c²π²/(4L²)) * u(t,x)
    """
    u = analytical_solution(t, x, config)

    coeff = -jnp.pi**2 / config.T**2 + config.c**2 * jnp.pi**2 / (4 * config.L**2)
    return coeff * u


def zero_vector_source(
    t: jnp.ndarray,
    x: jnp.ndarray,
    config: ProblemConfig,
) -> jnp.ndarray:
    """Vector source g for LS first-order system (zero for this IVP)."""
    _ = (t, x, config)
    return jnp.zeros((1,))
