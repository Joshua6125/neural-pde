"""
Utilities for manufactured solution wave equation experiment.

Provides:
- Problem setup (domain, parameters, source function)
- Analytical/reference solution computation
- Test data generation
- Visualization helpers
"""

import jax.numpy as jnp
from jax import vmap
from functools import partial
import numpy as np
from dataclasses import dataclass
from typing import Tuple
import matplotlib.pyplot as plt
import os


@dataclass
class ProblemConfig:
    L: float = 5.0
    T: float = 1.0
    c: float = 1.0

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


# Vectorize for batch inputs
analytical_solution_batch = vmap(
    partial(analytical_solution, config=ProblemConfig()),
    in_axes=(0, 0),
)

source_function_batch = vmap(
    partial(source_function, config=ProblemConfig()),
    in_axes=(0, 0),
)


def generate_test_points(
    config: ProblemConfig,
    n_time: int = 20,
    n_space: int = 80,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)

    t_points = np.linspace(0, config.T, n_time)

    x_points = np.linspace(-config.L, config.L, n_space)

    # TODO: use numpy cartesian product instead
    test_points_list = []
    for t in t_points:
        for x in x_points:
            test_points_list.append([t, x])

    test_points = np.array(test_points_list)

    print(f"Computing analytical solutions for {len(test_points)} test points...")
    reference_solutions = []

    for i, (t, x) in enumerate(test_points):
        if (i + 1) % 500 == 0:
            print(f"  Progress: {i + 1}/{len(test_points)}")

        u_ref = float(analytical_solution(
            jnp.array(t), jnp.array(x), config
        ))
        reference_solutions.append(u_ref)

    print()

    reference_solutions = np.array(reference_solutions)
    print(f"Reference solutions computed.")

    return test_points, reference_solutions


def save_test_data(
    test_points: np.ndarray,
    reference_solutions: np.ndarray,
    output_dir: str = "data",
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "reference_solution.npz")

    np.savez(
        filepath,
        test_points=test_points,
        reference_solutions=reference_solutions,
    )

    print(f"Test data saved to {filepath}")
    return filepath


def load_test_data(filepath: str = "data/reference_solution.npz") -> Tuple[np.ndarray, np.ndarray]:
    """Load precomputed test data."""
    data = np.load(filepath)
    return data["test_points"], data["reference_solutions"]


def plot_convergence(
    metrics: list,
    output_path: str = "results/convergence_curve.png",
) -> None:
    losses = [m.total_loss for m in metrics]
    epochs = list(range(len(losses)))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, linewidth=2, label="Training Loss")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("PINN Training Convergence", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.yscale("log")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Convergence plot saved to {output_path}")


def plot_solution_comparison(
    test_points: np.ndarray,
    u_pred: np.ndarray,
    u_exact: np.ndarray,
    time_indices: np.ndarray | list | None = None,
    output_path: str = "results/solution_snapshots.png",
    config: ProblemConfig | None = None,
) -> None:
    if config is None:
        config = ProblemConfig()

    if time_indices is None:
        unique_times = np.unique(test_points[:, 0])
        if len(unique_times) > 4:
            selected_time_indices = np.round(np.linspace(0, len(unique_times) - 1, 4)).astype(int)
            # Use max(test_points[:, 0]) for exact robustness
            time_indices = []
            for idx in selected_time_indices:
                target_time = unique_times[idx]
                # Find closest match to handle floating-point precision
                closest_idx = np.argmin(np.abs(test_points[:, 0] - target_time))
                time_indices.append(closest_idx)
        else:
            time_indices = list(range(len(unique_times)))

    n_snapshots = len(time_indices)
    fig, axes = plt.subplots(n_snapshots, 1, figsize=(10, 3.2 * n_snapshots), sharex=True)

    if n_snapshots == 1:
        axes = np.array([axes])

    for row, idx in enumerate(time_indices):
        t_val = test_points[idx, 0]

        # Extract spatial points at this time - use tight tolerance
        time_mask = np.isclose(test_points[:, 0], t_val, atol=1e-10)
        t_space = test_points[time_mask]
        u_p = u_pred[time_mask]
        u_e = u_exact[time_mask]

        x_sorted_idx = np.argsort(t_space[:, 1])
        x_vals = t_space[x_sorted_idx, 1]
        u_p_sorted = u_p[x_sorted_idx]
        u_e_sorted = u_e[x_sorted_idx]

        axes[row].plot(x_vals, u_e_sorted, label="Exact", linewidth=2)
        axes[row].plot(x_vals, u_p_sorted, label="Predicted", linestyle="--", linewidth=2)
        axes[row].set_ylabel("u", fontsize=11)
        axes[row].set_title(f"t={t_val:.3f}", fontsize=11)
        axes[row].grid(True, alpha=0.3)
        if row == 0:
            axes[row].legend(fontsize=10)

    axes[-1].set_xlabel("x", fontsize=11)

    fig.suptitle("1D Solution Snapshots: PINN Predictions vs Reference", fontsize=14)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSolution comparison plot saved to {output_path}")


def plot_error_map(
    test_points: np.ndarray,
    u_pred: np.ndarray,
    u_exact: np.ndarray,
    output_path: str = "results/error_map.png",
) -> None:
    unique_t = np.unique(test_points[:, 0])
    unique_x = np.unique(test_points[:, 1])

    error = np.abs(u_pred - u_exact)

    if len(unique_t) * len(unique_x) == len(error):
        error_grid = error.reshape(len(unique_t), len(unique_x))

        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(
            error_grid,
            extent=(
                float(unique_x.min()),
                float(unique_x.max()),
                float(unique_t.min()),
                float(unique_t.max()),
            ),
            origin="lower",
            aspect="auto",
            cmap="hot",
        )
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Absolute Error |u_pred - u_exact|", fontsize=11)

        ax.set_xlabel("x", fontsize=11)
        ax.set_ylabel("t", fontsize=11)
        ax.set_title("Error Map Over Time-Space Grid", fontsize=13)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Error map saved to {output_path}")


def save_experiment_log(
    log_dict: dict,
    output_path: str = "results/experiment_log.json",
) -> None:

    import json
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(log_dict, f, indent=2)
    print(f"Experiment log saved to {output_path}")


if __name__ == "__main__":
    config = ProblemConfig()
    print("Generating test data...")
    test_points, ref_solutions = generate_test_points(
        config, n_time=10, n_space=40
    )
    save_test_data(test_points, ref_solutions)
    print("Done!")
