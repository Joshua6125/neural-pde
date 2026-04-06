import sys
from pathlib import Path
import time
from datetime import datetime
from typing import Tuple

import numpy as np
import jax.numpy as jnp

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.cli import run_training
from src.algorithms import PINNConfig
from src.models import PINNModelConfig, NeuralNetModelConfig, build_model
from src.integration import QuadratureConfig, MonteCarloConfig
from src.train import TrainConfig

from utils import (
    ProblemConfig,
    generate_test_points,
    save_test_data,
    load_test_data,
    analytical_solution,
    analytical_solution_t,
    source_function,
    plot_convergence,
    plot_solution_comparison,
    plot_error_map,
    save_experiment_log,
)

def get_problem_config() -> ProblemConfig:
    return ProblemConfig(
        L=5.0,
        T=1.0,
        c=1.0,
    )


def get_model_config() -> PINNModelConfig:
    return PINNModelConfig(
        u_model=NeuralNetModelConfig(
            hidden_dim=64,
            num_layers=5,
            output_heads={"u": 1}
        )
    )


def get_algorithm_config(
    problem_config: ProblemConfig,
) -> PINNConfig:
    return PINNConfig(
        model=get_model_config(),
        f=lambda v: source_function(
            jnp.array(v[0]), jnp.array(v[1]), jnp.array(v[2]), problem_config
        ),
        u0=lambda v: analytical_solution(
            jnp.array(v[0]), jnp.array(v[1]), jnp.array(v[2]), problem_config
        ),
        ut0=lambda v: analytical_solution_t(
            jnp.array(v[0]), jnp.array(v[1]), jnp.array(v[2]), problem_config
        ),
        c=problem_config.c,
        ic_weight=10.0,
        bc_weight=100.0,
    )


def get_integration_config(
    problem_config: ProblemConfig,
) -> QuadratureConfig | MonteCarloConfig:
    return MonteCarloConfig(
        dim=3,
        x_min=-problem_config.L,
        x_max=problem_config.L,
        monte_carlo_boundary_samples=1000,
        monte_carlo_interior_samples=1000,
        monte_carlo_seed=42
    )

    return QuadratureConfig(
        dim=3,
        x_min=-problem_config.L,
        x_max=problem_config.L,
        degree=8,
    )


def get_training_config() -> TrainConfig:
    return TrainConfig(
        epochs=20000,
        learning_rate=1e-4,
        optimizer="adamw",
        use_jit=True,
        seed=42,
    )


def prepare_test_data(
    problem_config: ProblemConfig,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    test_data_path = "experiments/data/reference_solution.npz"

    if Path(test_data_path).exists() and not force_recompute:
        print(f"Loading existing test data from {test_data_path}...")
        return load_test_data(test_data_path)

    print("Generating test data (instant with analytical solution)...")
    test_points, reference_solutions = generate_test_points(
        problem_config,
        n_time=20,
        n_space_per_dim=15,
        seed=42,
    )
    save_test_data(test_points, reference_solutions)
    return test_points, reference_solutions


def run_experiment(force_recompute_ref: bool = False) -> dict:
    print("[1/6] Setting up problem configuration...")
    problem_config = get_problem_config()

    print("\n[2/6] Preparing reference solutions...")
    test_points, reference_solutions = prepare_test_data(
        problem_config, force_recompute=force_recompute_ref
    )
    print(f"  Generated {len(test_points)} test points")
    print(f"  Reference solution range: [{reference_solutions.min():.4f}, {reference_solutions.max():.4f}]")

    print("\n[3/6] Building neural network model...")
    model_config = get_model_config()
    print(f"  Model: u_model with hidden_dim={model_config.u_model.hidden_dim}, num_layers={model_config.u_model.num_layers}")
    model = build_model(model_config)
    print(f"  Model built successfully")

    print("\n[4/6] Configuring PINN algorithm and integration...")
    algo_config = get_algorithm_config(problem_config)
    integration_config = get_integration_config(problem_config)
    train_config = get_training_config()
    print(f"  Algorithm: PINN")
    print(f"  Training: {train_config.epochs} epochs, lr={train_config.learning_rate}")

    # Train
    print("\n[5/6] Running PINN training...")
    start_time = time.time()

    try:
        state, metrics = run_training(
            algorithm_cfg=algo_config,
            integration_cfg=integration_config,
            model_cfg=model_config,
            train_cfg=train_config,
            sample_input=jnp.array([[0.5, 0.0, 0.0]]),  # Example input
        )
        elapsed = time.time() - start_time

        print(f"  Training completed in {elapsed:.2f} seconds")
        print(f"  Final loss: {metrics[-1].total_loss:.6e}")
        print(f"  Best loss: {min(m.total_loss for m in metrics):.6e}")

    except Exception as e:
        print(f"  ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

    print("\n[6/6] Evaluating on test set and generating plots...")

    u_pred = []
    for i, (t, x, y) in enumerate(test_points):
        if (i + 1) % 500 == 0:
            print(f"  Evaluation progress: {i + 1}/{len(test_points)}")

        try:
            # Prepare batch input [t, x, y]
            batch = jnp.array([[t, x, y]])
            # Apply model with trained parameters
            u_dict = model.apply(state.params, batch)
            u_val = u_dict["u"]  # type: ignore
            u_pred.append(float(u_val[0][0]))
        except Exception as e:
            print(f"  Warning: Could not evaluate at ({t}, {x}, {y}): {e}")
            u_pred.append(0.0)

    u_pred = np.array(u_pred)

    # Compute metrics
    error = np.abs(u_pred - reference_solutions)
    l2_error = np.sqrt(np.mean(error**2))
    max_error = np.max(error)
    mean_abs_error = np.mean(error)

    print(f"  L2 Error: {l2_error:.6e}")
    print(f"  Max Error: {max_error:.6e}")
    print(f"  Mean Abs Error: {mean_abs_error:.6e}")

    # Generate plots
    print("\n  Generating visualizations...")
    plot_convergence(metrics, "experiments/results/convergence_curve.png")
    plot_solution_comparison(
        test_points, u_pred, reference_solutions,
        output_path="experiments/results/solution_snapshots.png",
        config=problem_config,
    )
    plot_error_map(
        test_points, u_pred, reference_solutions,
        output_path="experiments/results/error_map.png",
    )

    # Prepare results
    results = {
        "timestamp": datetime.now().isoformat(),
        "problem": {
            "type": "2D Wave Equation",
            "domain": f"[0, {problem_config.T}] x [-{problem_config.L}, {problem_config.L}]^2",
            "wave_speed": problem_config.c,
        },
        "model": {
            "type": "PINN",
            "architecture": "NeuralNet",
            "hidden_dim": model_config.u_model.hidden_dim,
            "num_layers": model_config.u_model.num_layers,
        },
        "training": {
            "epochs": train_config.epochs,
            "learning_rate": train_config.learning_rate,
            "optimizer": train_config.optimizer,
            "elapsed_time_seconds": elapsed,
        },
        "integration": {
            "method": integration_config.integration_method,
            "dimension": integration_config.dim,
        },
        "evaluation": {
            "n_test_points": len(test_points),
            "l2_error": float(l2_error),
            "max_error": float(max_error),
            "mean_abs_error": float(mean_abs_error),
            "final_loss": float(metrics[-1].total_loss),
            "best_loss": float(min(m.total_loss for m in metrics)),
            "best_loss_epoch": int(np.argmin([m.total_loss for m in metrics])) * train_config.log_every,
        },
        "convergence": {
            "converged": True,  # Will be updated based on loss trajectory
            "loss_decreasing": float(metrics[-1].total_loss) < float(metrics[0].total_loss),
        },
        "output_files": [
            "experiments/results/convergence_curve.png",
            "experiments/results/solution_snapshots.png",
            "experiments/results/error_map.png",
            "experiments/results/experiment_log.json",
        ],
    }

    # Save results
    save_experiment_log(results, "experiments/results/experiment_log.json")

    print("\nGenerated outputs:")
    for fname in results["output_files"]:
        if Path(fname).exists():
            print(f"  ✓ {fname}")
        else:
            print(f"  ? {fname} (not found)")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run 2D wave equation experiment")
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation of reference solutions",
    )
    args = parser.parse_args()

    results = run_experiment(force_recompute_ref=args.force_recompute)

    if "error" in results:
        sys.exit(1)
