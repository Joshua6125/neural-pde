import sys
from pathlib import Path
import time
from datetime import datetime
from typing import Any, Tuple, cast

import jax
import numpy as np
import jax.numpy as jnp

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.cli import run_training
from src.models import build_model
from config import (
    ExperimentCombination,
    ProblemConfig,
    build_experiment_combinations,
    get_integration_config,
    get_problem_config,
    get_training_config,
    analytical_solution_t,
    analytical_solution_x,
)

from utils import (
    generate_test_points,
    save_test_data,
    load_test_data,
    reconstruct_u_from_v,
    plot_convergence,
    plot_solution_comparison,
    plot_error_map,
    save_experiment_log,
    save_experiment_arrays,
    save_model_checkpoint,
)


def prepare_test_data(
    problem_config: ProblemConfig,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    test_data_path = "data/reference_solution.npz"

    if Path(test_data_path).exists() and not force_recompute:
        print(f"Loading existing test data from {test_data_path}...")
        test_points, reference_solutions = load_test_data(test_data_path)
        if test_points.ndim == 2 and test_points.shape[1] == 2:
            return test_points, reference_solutions

        print("  Existing reference data has incompatible dimension. Recomputing for 1D setup...")

    print("Generating test data (instant with analytical solution)...")
    test_points, reference_solutions = generate_test_points(
        problem_config,
        n_time=20,
        n_space=80,
        seed=42,
    )
    save_test_data(test_points, reference_solutions)
    return test_points, reference_solutions


def _safe_label(label: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in label).strip("_")


def _error_stats(pred: np.ndarray, exact: np.ndarray) -> dict[str, float]:
    error = np.abs(pred - exact)
    return {
        "l2_error": float(np.sqrt(np.mean(error**2))),
        "max_error": float(np.max(error)),
        "mean_abs_error": float(np.mean(error)),
    }


def _evaluate_combination(
    combination: ExperimentCombination,
    params: Any,
    test_points: np.ndarray,
) -> dict[str, np.ndarray]:
    model = build_model(combination.model_config)
    batch_points = jnp.asarray(test_points)

    if combination.method == "pinn":
        outputs = cast(dict[str, jnp.ndarray], model.apply(params, batch_points))
        u_pred = np.asarray(outputs["u"]).reshape(-1)

        def scalar_u(point: jnp.ndarray) -> jnp.ndarray:
            batch = point[None, :]
            out = cast(dict[str, jnp.ndarray], model.apply(params, batch))
            return jnp.asarray(out["u"]).reshape(-1)[0]

        grad_u = jax.vmap(jax.grad(scalar_u))(batch_points)
        return {
            "u": u_pred,
            "v": np.asarray(grad_u[:, 0]),
            "sigma": np.asarray(grad_u[:, 1]),
        }

    if combination.method == "ls":
        outputs = cast(dict[str, jnp.ndarray], model.apply(params, batch_points))
        v_pred = np.asarray(outputs["v"]).reshape(-1)
        sigma_pred = np.asarray(outputs["sigma"]).reshape(-1)
        return {
            "u": reconstruct_u_from_v(test_points, v_pred),
            "v": v_pred,
            "sigma": sigma_pred,
        }

    raise ValueError(f"Unsupported method for evaluation: {combination.method}")


def run_experiment(force_recompute_ref: bool = False) -> dict:
    print("[1/6] Setting up problem configuration...")
    problem_config = get_problem_config()

    combinations = build_experiment_combinations(problem_config)
    if not combinations:
        raise ValueError("No experiment combinations were generated from ProblemConfig.")

    print(
        "  Requested combinations: "
        + ", ".join(combo.label for combo in combinations)
    )

    print("\n[2/6] Preparing reference solutions...")
    test_points, reference_solutions = prepare_test_data(
        problem_config, force_recompute=force_recompute_ref
    )
    print(f"  Generated {len(test_points)} test points")
    print(f"  Reference solution range: [{reference_solutions.min():.4f}, {reference_solutions.max():.4f}]")

    print("\n[3/6] Building run plan for all method-model combinations...")
    for combo in combinations:
        print(f"  - {combo.label}: method={combo.method}, model={combo.model}")

    print("\n[4/6] Building shared integration and training config...")
    integration_config = get_integration_config(problem_config)
    train_config = get_training_config()
    print(
        f"  Shared training: {train_config.epochs} epochs, "
        f"lr={train_config.learning_rate}"
    )

    print("\n[5/6] Running training across all combinations...")
    total_start = time.time()
    trained_runs: dict[str, dict[str, Any]] = {}
    metrics_by_label: dict[str, list] = {}
    elapsed_by_label: dict[str, float] = {}
    sample_input = jnp.array([[0.5, 0.0]])

    try:
        for combo in combinations:
            run_start = time.time()
            print(f"  Starting training for {combo.label}...")

            state, metrics = run_training(
                algorithm_cfg=combo.algorithm_config,
                integration_cfg=integration_config,
                model_cfg=combo.model_config,
                train_cfg=train_config,
                sample_input=sample_input,
            )

            run_elapsed = time.time() - run_start
            elapsed_by_label[combo.label] = float(run_elapsed)
            metrics_by_label[combo.label] = metrics
            trained_runs[combo.label] = {
                "combination": combo,
                "state": state,
                "metrics": metrics,
            }

            if not metrics:
                raise RuntimeError(f"No training metrics recorded for {combo.label}.")

            print(f"  {combo.label} training completed in {run_elapsed:.2f} seconds")
            print(f"  {combo.label} final loss: {metrics[-1].total_loss:.6e}")
            print(f"  {combo.label} best loss: {min(m.total_loss for m in metrics):.6e}")

    except Exception as e:
        print(f"  ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

    elapsed = time.time() - total_start

    print("\n[6/6] Evaluating on test set and generating plots...")

    predictions_by_label: dict[str, dict[str, np.ndarray]] = {}
    for label, run_data in trained_runs.items():
        combo = cast(ExperimentCombination, run_data["combination"])
        state = run_data["state"]
        print(f"  Evaluating {label}...")
        predictions_by_label[label] = _evaluate_combination(combo, state.params, test_points)
    batch_points = jnp.asarray(test_points)
    v_exact_jax = jax.vmap(
    lambda p: analytical_solution_t(p[0], p[1], problem_config)
    )(batch_points)
    sigma_exact_jax = jax.vmap(
    lambda p: analytical_solution_x(p[0], p[1], problem_config)
    )(batch_points)

    v_exact = np.asarray(jax.device_get(v_exact_jax))
    sigma_exact = np.asarray(jax.device_get(sigma_exact_jax))


    # v_exact = np.array([
    #     float(analytical_solution_t(jnp.array(t), jnp.array(x), problem_config))
    #     for t, x in test_points
    # ])
    # sigma_exact = np.array([
    #     float(analytical_solution_x(jnp.array(t), jnp.array(x), problem_config))
    #     for t, x in test_points
    # ])

    evaluation: dict[str, dict[str, float | int]] = {}
    loss_decreasing: dict[str, bool] = {}
    u_predictions_for_plot: dict[str, np.ndarray] = {}
    error_payload: dict[str, dict[str, np.ndarray]] = {}

    for label, run_data in trained_runs.items():
        metrics = run_data["metrics"]
        predictions = predictions_by_label[label]
        u_pred = predictions["u"]
        v_pred = predictions["v"]
        sigma_pred = predictions["sigma"]

        u_stats = _error_stats(u_pred, reference_solutions)
        v_stats = _error_stats(v_pred, v_exact)
        sigma_stats = _error_stats(sigma_pred, sigma_exact)

        total_losses = [m.total_loss for m in metrics]
        best_idx = int(np.argmin(total_losses))
        evaluation[label] = {
            **u_stats,
            "v_l2_error": v_stats["l2_error"],
            "v_max_error": v_stats["max_error"],
            "v_mean_abs_error": v_stats["mean_abs_error"],
            "sigma_l2_error": sigma_stats["l2_error"],
            "sigma_max_error": sigma_stats["max_error"],
            "sigma_mean_abs_error": sigma_stats["mean_abs_error"],
            "final_loss": float(metrics[-1].total_loss),
            "best_loss": float(total_losses[best_idx]),
            "best_loss_epoch": int(metrics[best_idx].step),
        }

        loss_decreasing[label] = float(metrics[-1].total_loss) < float(metrics[0].total_loss)
        u_predictions_for_plot[label] = u_pred
        error_payload[label] = {
            "u_error": np.abs(u_pred - reference_solutions),
            "v_error": np.abs(v_pred - v_exact),
            "sigma_error": np.abs(sigma_pred - sigma_exact),
        }

        print(f"  {label} L2 Error: {u_stats['l2_error']:.6e}")
        print(f"  {label} v L2 Error: {v_stats['l2_error']:.6e}")
        print(f"  {label} sigma L2 Error: {sigma_stats['l2_error']:.6e}")

    # Generate plots
    print("\n  Generating visualisations...")
    plot_convergence(metrics_by_label, "results/convergence_curve.png")
    plot_solution_comparison(
        test_points,
        u_predictions_for_plot,
        reference_solutions,
        output_path="results/solution_snapshots.png",
        config=problem_config,
    )
    plot_error_map(
        test_points,
        u_predictions_for_plot,
        reference_solutions,
        output_path="results/error_map.png",
    )

    artifact_dir = Path("results") / "artifacts"

    serialisable_predictions = {
        label: {
            key: value for key, value in data.items()
        }
        for label, data in predictions_by_label.items()
    }

    save_experiment_arrays(
        str(artifact_dir / "evaluation_outputs.pkl"),
        test_points=test_points,
        reference_solutions=reference_solutions,
        predictions=serialisable_predictions,
        v_exact=v_exact,
        sigma_exact=sigma_exact,
        errors=error_payload,
    )

    training_history = {
        label: {
            "step": np.array([m.step for m in metrics]),
            "total_loss": np.array([m.total_loss for m in metrics]),
            "interior_loss": np.array([m.interior_loss for m in metrics]),
            "boundary_loss": np.array([m.boundary_loss for m in metrics]),
        }
        for label, metrics in metrics_by_label.items()
    }

    save_experiment_arrays(
        str(artifact_dir / "training_history.pkl"),
        training_history=training_history,
    )

    checkpoint_files: list[str] = []
    for label, run_data in trained_runs.items():
        filename = f"{_safe_label(label)}_params.msgpack"
        output_path = str(artifact_dir / filename)
        save_model_checkpoint(run_data["state"].params, output_path)
        checkpoint_files.append(output_path)

    # Prepare results
    results = {
        "timestamp": datetime.now().isoformat(),
        "problem": {
            "type": "1D Wave Equation",
            "domain": f"[0, {problem_config.T}] x [-{problem_config.L}, {problem_config.L}]",
            "wave_speed": problem_config.c,
            "methods": problem_config.methods,
            "models": problem_config.models,
        },
        "combinations": [
            {
                "label": combo.label,
                "method": combo.method,
                "model": combo.model,
            }
            for combo in combinations
        ],
        "training": {
            "epochs": train_config.epochs,
            # "learning_rate": train_config.learning_rate, # TODO: Should figure out a way to log this as well.
            "optimiser": train_config.optimiser,
            "elapsed_time_seconds": float(elapsed),
            "combinations_elapsed_time_seconds": elapsed_by_label,
        },
        "integration": {
            "method": integration_config.integration_method,
            "dimension": integration_config.dim,
        },
        "evaluation": {
            "n_test_points": len(test_points),
            "combinations": evaluation,
        },
        "convergence": {
            "converged": bool(all(loss_decreasing.values())),
            "loss_decreasing": loss_decreasing,
        },
        "output_files": [
            "results/convergence_curve.png",
            "results/solution_snapshots.png",
            "results/error_map.png",
            "results/experiment_log.json",
            "results/artifacts/evaluation_outputs.pkl",
            "results/artifacts/training_history.pkl",
            *checkpoint_files,
        ],
    }

    # Save results
    save_experiment_log(results, "results/experiment_log.json")

    print("\nGenerated outputs:")
    for fname in results["output_files"]:
        if Path(fname).exists():
            print(f"  ✓ {fname}")
        else:
            print(f"  ? {fname} (not found)")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run 1D wave equation experiment")
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation of reference solutions",
    )
    args = parser.parse_args()

    results = run_experiment(force_recompute_ref=args.force_recompute)

    if "error" in results:
        sys.exit(1)
