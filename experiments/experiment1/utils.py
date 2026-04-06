"""
Utilities for manufactured solution wave equation experiment.

Provides:
- Problem setup (domain, parameters, source function)
- Analytical/reference solution computation
- Test data generation
- Visualization helpers
"""

import jax
import jax.numpy as jnp
from jax import vmap
from functools import partial
import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple
import matplotlib.pyplot as plt
from matplotlib import cm
import os


@dataclass
class ProblemConfig:
    L: float = 5.0
    T: float = 1.0
    c: float = 1.0

def analytical_solution(
    t: jnp.ndarray,
    x: jnp.ndarray,
    y: jnp.ndarray,
    config: ProblemConfig,
) -> jnp.ndarray:
    """
    Analytical solution.

    u(t, x, y) = sin(π*t/T) * sin(π*(x+L)/(2L)) * sin(π*(y+L)/(2L))
    """
    t_term = jnp.sin(jnp.pi * t / config.T)
    x_term = jnp.sin(jnp.pi * (x + config.L) / (2 * config.L))
    y_term = jnp.sin(jnp.pi * (y + config.L) / (2 * config.L))
    return t_term * x_term * y_term


def analytical_solution_t(
    t: jnp.ndarray,
    x: jnp.ndarray,
    y: jnp.ndarray,
    config: ProblemConfig,
) -> jnp.ndarray:
    """
    Time derivative of analytical solution: ∂u/∂t.

    u_t(t, x, y) = (π/T) * cos(π*t/T) * sin(π*(x+L)/(2L)) * sin(π*(y+L)/(2L))

    This is the initial velocity constraint at t=0.
    """
    t_term = jnp.cos(jnp.pi * t / config.T) * (jnp.pi / config.T)
    x_term = jnp.sin(jnp.pi * (x + config.L) / (2 * config.L))
    y_term = jnp.sin(jnp.pi * (y + config.L) / (2 * config.L))
    return t_term * x_term * y_term


def source_function(
    t: jnp.ndarray,
    x: jnp.ndarray,
    y: jnp.ndarray,
    config: ProblemConfig,
) -> jnp.ndarray:
    """
    Source term computed analytically from u_tt - c²∇²u = f.

    For u(t,x,y) = sin(πt/T) * sin(π(x+L)/2L) * sin(π(y+L)/2L):

    u_tt = -π²/T² * sin(πt/T) * sin(π(x+L)/2L) * sin(π(y+L)/2L)
    u_xx = -π²/(4L²) * sin(πt/T) * sin(π(x+L)/2L) * sin(π(y+L)/2L)
    u_yy = -π²/(4L²) * sin(πt/T) * sin(π(x+L)/2L) * sin(π(y+L)/2L)

    f = u_tt - c²(u_xx + u_yy)
      = (-π²/T² + c² * 2π²/(4L²)) * sin(πt/T) * sin(π(x+L)/2L) * sin(π(y+L)/2L)
      = (-π²/T² + c²π²/(2L²)) * u(t,x,y)
    """
    u = analytical_solution(t, x, y, config)

    coeff = -jnp.pi**2 / config.T**2 + config.c**2 * jnp.pi**2 / (2 * config.L**2)
    return coeff * u


# Vectorize for batch inputs
analytical_solution_batch = vmap(
    partial(analytical_solution, config=ProblemConfig()),
    in_axes=(0, 0, 0),
)

source_function_batch = vmap(
    partial(source_function, config=ProblemConfig()),
    in_axes=(0, 0, 0),
)


def generate_test_points(
    config: ProblemConfig,
    n_time: int = 20,
    n_space_per_dim: int = 15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)

    t_points = np.linspace(0, config.T, n_time)

    x_points = np.linspace(-config.L, config.L, n_space_per_dim)
    y_points = np.linspace(-config.L, config.L, n_space_per_dim)

    test_points_list = []
    for t in t_points:
        for x in x_points:
            for y in y_points:
                test_points_list.append([t, x, y])

    test_points = np.array(test_points_list)

    print(f"Computing analytical solutions for {len(test_points)} test points...")
    reference_solutions = []

    for i, (t, x, y) in enumerate(test_points):
        if (i + 1) % 500 == 0:
            print(f"  Progress: {i + 1}/{len(test_points)}")

        u_ref = float(analytical_solution(
            jnp.array(t), jnp.array(x), jnp.array(y), config
        ))
        reference_solutions.append(u_ref)

    reference_solutions = np.array(reference_solutions)
    print(f"Reference solutions computed.")

    return test_points, reference_solutions


def save_test_data(
    test_points: np.ndarray,
    reference_solutions: np.ndarray,
    output_dir: str = "experiments/data",
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


def load_test_data(filepath: str = "experiments/data/reference_solution.npz") -> Tuple[np.ndarray, np.ndarray]:
    """Load precomputed test data."""
    data = np.load(filepath)
    return data["test_points"], data["reference_solutions"]


def plot_convergence(
    metrics: list,
    output_path: str = "experiments/results/convergence_curve.png",
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
    output_path: str = "experiments/results/solution_snapshots.png",
    config: ProblemConfig | None = None,
) -> None:
    if config is None:
        config = ProblemConfig()

    if time_indices is None:
        # Sample 4 time steps uniformly
        unique_times = np.unique(test_points[:, 0])
        if len(unique_times) > 4:
            time_indices = np.linspace(0, len(unique_times) - 1, 4, dtype=int)
            time_indices = [np.where(test_points[:, 0] == unique_times[i])[0][0]
                           for i in time_indices]
        else:
            time_indices = list(range(len(unique_times)))

    n_snapshots = len(time_indices)
    fig, axes = plt.subplots(n_snapshots, 2, figsize=(12, 4 * n_snapshots))

    if n_snapshots == 1:
        axes = axes.reshape(1, -1)

    for row, idx in enumerate(time_indices):
        t_val = test_points[idx, 0]

        # Extract spatial points at this time
        time_mask = np.isclose(test_points[:, 0], t_val)
        t_space = test_points[time_mask]
        u_p = u_pred[time_mask]
        u_e = u_exact[time_mask]

        # Reshape for 2D plotting if on grid
        n_unique_x = len(np.unique(t_space[:, 1]))
        n_unique_y = len(np.unique(t_space[:, 2]))

        if n_unique_x * n_unique_y == len(u_p):
            u_p_grid = u_p.reshape(n_unique_x, n_unique_y)
            u_e_grid = u_e.reshape(n_unique_x, n_unique_y)
            x_grid = t_space[:, 1].reshape(n_unique_x, n_unique_y)
            y_grid = t_space[:, 2].reshape(n_unique_x, n_unique_y)

            # Plot predicted
            im1 = axes[row, 0].contourf(x_grid, y_grid, u_p_grid, levels=15, cmap="RdBu_r")
            axes[row, 0].set_title(f"Predicted u at t={t_val:.3f}", fontsize=11)
            axes[row, 0].set_xlabel("x")
            axes[row, 0].set_ylabel("y")
            plt.colorbar(im1, ax=axes[row, 0])

            # Plot exact
            im2 = axes[row, 1].contourf(x_grid, y_grid, u_e_grid, levels=15, cmap="RdBu_r")
            axes[row, 1].set_title(f"Exact u at t={t_val:.3f}", fontsize=11)
            axes[row, 1].set_xlabel("x")
            axes[row, 1].set_ylabel("y")
            plt.colorbar(im2, ax=axes[row, 1])

    fig.suptitle("Solution Snapshots: PINN Predictions vs Reference", fontsize=14)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Solution comparison plot saved to {output_path}")


def plot_error_map(
    test_points: np.ndarray,
    u_pred: np.ndarray,
    u_exact: np.ndarray,
    output_path: str = "experiments/results/error_map.png",
) -> None:
    # Get final time points
    t_final = np.max(test_points[:, 0])
    final_mask = np.isclose(test_points[:, 0], t_final)

    t_final_space = test_points[final_mask]
    u_p_final = u_pred[final_mask]
    u_e_final = u_exact[final_mask]

    error = np.abs(u_p_final - u_e_final)

    n_unique_x = len(np.unique(t_final_space[:, 1]))
    n_unique_y = len(np.unique(t_final_space[:, 2]))

    if n_unique_x * n_unique_y == len(error):
        error_grid = error.reshape(n_unique_x, n_unique_y)
        x_grid = t_final_space[:, 1].reshape(n_unique_x, n_unique_y)
        y_grid = t_final_space[:, 2].reshape(n_unique_x, n_unique_y)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.contourf(x_grid, y_grid, error_grid, levels=20, cmap="hot")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Absolute Error |u_pred - u_exact|", fontsize=11)

        ax.set_xlabel("x", fontsize=11)
        ax.set_ylabel("y", fontsize=11)
        ax.set_title(f"Spatial Error Distribution at t={t_final:.3f}", fontsize=13)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Error map saved to {output_path}")


def save_experiment_log(
    log_dict: dict,
    output_path: str = "experiments/results/experiment_log.json",
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
        config, n_time=10, n_space_per_dim=8
    )
    save_test_data(test_points, ref_solutions)
    print("Done!")
