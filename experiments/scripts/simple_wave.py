"""
The purpose of this experiment is to compar pairs of loss functions and neural architectures.

In particular, all loss functions are compared to one another with MLP.
"""

from glob import glob
from itertools import product
from collections import defaultdict
from omegaconf import DictConfig
from utils import (
    build_model_config,
    build_method_config,
    build_integration_config,
    build_trainer_config,
    create_evaluation_domain,
    evaluate_metrics
)
from src.trainer import TrainState, run_training
from src.models import AnyModelConfig, build_model
from src.loss_functions import AlgorithmConfig

import jax.numpy as jnp
import numpy as np

import time
import os
import pickle
import dataclasses
import jax


class ProblemDefinition:
    """Analytical problem definition of a simple wave equation."""

    def __init__(self, cfg: DictConfig):
        """
        Parameters
        ----------
        cfg : DictConfig
            The overridden experiment configuration.
        """

        self.x_min = float(cfg.integration.get("x_min", 0.0))
        self.x_max = float(cfg.integration.get("x_max", 1.0))
        self.T = float(cfg.problem_params.get("T", 1.0))
        self.c = float(cfg.problem_params.get("c", 1.0))
        self.cfg = cfg

    def analytical_solution(self, t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Compute the manufactured analytical solution."""
        t_array = jnp.asarray(t)
        x_array = jnp.asarray(x)

        t_term = jnp.sin(jnp.pi * t_array / self.T)
        x_term = jnp.sin(jnp.pi * (x_array - self.x_min) / (self.x_max - self.x_min))
        result = t_term * x_term

        return jnp.where(jnp.isclose(t_array / self.T, 1.0, atol=1e-10), 0.0, result)

    def analytical_solution_t(self, t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Time derivative of the manufactured analytical solution."""
        return (
            jnp.pi / self.T
        ) * jnp.cos(jnp.pi * t / self.T) * jnp.sin(jnp.pi * (x - self.x_min) / (self.x_max - self.x_min))

    def analytical_solution_x(self, t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Spatial derivative of the manufactured analytical solution."""
        return (
            jnp.sin(jnp.pi * t / self.T)
            * jnp.cos(jnp.pi * (x - self.x_min) / (self.x_max - self.x_min))
            * (jnp.pi / (self.x_max - self.x_min))
        )

    def source_function(self, t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Closed-form source term for the manufactured PDE."""
        u = self.analytical_solution(t, x)
        coeff = -jnp.pi**2 / self.T**2 + self.c**2 * jnp.pi**2 / (self.x_max - self.x_min)**2
        return coeff * u

    def zero_vector_source(self, t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Zero vector source used by the SLS formulation."""
        _ = (t, x)
        return jnp.zeros((1,))

    def get_sample_input(self) -> jnp.ndarray:
        """Return a representative sample input for model initialisation."""
        return jnp.asarray([[0.5, 0.0]], dtype=jnp.float32)


class RunTraining:
    """Run training of experiment 1. Saves artifacts to output_dir/ eagerly."""

    def __init__(self, problem: ProblemDefinition, cfg: DictConfig, output_dir: str):
        """
        Parameters
        ----------
        problem : ProblemDefinition
            Definition of the handled wave equation
        cfg : DictConfig
            The overridden experiment configuration.
        output_dir : str
            The path where all results/artifacts should be stored eagerly.
        """
        self.cfg = cfg
        self.problem = problem
        self.output_dir = output_dir

        # Define the simple wave equation lambdas
        self.f = lambda v: self.problem.source_function(jnp.array(v[0]), jnp.array(v[1]))
        self.g = lambda v: self.problem.zero_vector_source(jnp.array(v[0]), jnp.array(v[1]))
        self.v0 = lambda v: self.problem.analytical_solution_t(jnp.array(v[0]), jnp.array(v[1]))
        self.sigma0 = lambda v: jnp.array(
            [self.problem.analytical_solution_x(jnp.array(v[0]), jnp.array(v[1]))]
        )
        self.u0 = lambda v: self.problem.analytical_solution(jnp.array(v[0]), jnp.array(v[1]))
        self.ut0 = lambda v: self.problem.analytical_solution_t(jnp.array(v[0]), jnp.array(v[1]))
        self.c = float(cfg.problem_params.get("c", 1.0))

        self.wave_functions = {
            "f": self.f,
            "g": self.g,
            "v0": self.v0,
            "sigma0": self.sigma0,
            "u0": self.u0,
            "ut0": self.ut0,
            "c": self.c,
        }

    def _generate_combinations(self) -> list[tuple[AnyModelConfig, AlgorithmConfig]]:
        """Generate combination of all configs"""
        models = self.cfg.get("models", "")
        methods = self.cfg.get("methods", "")

        pairs = []
        for method, model in product(methods, models):
            heads = method.get("output_heads", "")
            model_config = build_model_config(model, heads)
            method_config = build_method_config(method, model_config, self.wave_functions)

            pairs.append((model_config, method_config))

        if not pairs:
            ValueError("No valid models-method pairs found in config")

        return pairs

    def train_multiple(self, iterations: int):
        """Train all model-method combinations for multiple iterations"""
        for i in range(iterations):
            self.train_all(iteration=i)

    def train_all(self, iteration: int = 0):
        """Train all model-method combinations"""
        sample_input = self.problem.get_sample_input()

        integrator_data = self.cfg.get("integration", "")
        integrator_config = build_integration_config(integrator_data)

        trainer_data = self.cfg.get("training")
        base_trainer_config = build_trainer_config(trainer_data)

        # Adjust seed based on iteration
        trainer_config = dataclasses.replace(
            base_trainer_config,
            seed=base_trainer_config.seed + iteration
        )

        pairs = self._generate_combinations()
        total_runs = len(pairs)

        print(f"--- Starting Training Phase (Iteration {iteration+1}) ---")
        print(f"Total configurations to train: {total_runs}")
        init_lr = self.cfg.get("training", {}).get("learning_rate", {}).get("init_value", "Unknown")
        print(f"Epochs: {trainer_config.epochs}, Initial LR: {init_lr}, Seed: {trainer_config.seed}\n")

        domain_pts = create_evaluation_domain(self.cfg)
        f_fn = lambda t, x: self.problem.source_function(t, x)
        g_fn = lambda t, x: self.problem.zero_vector_source(t, x)
        v_true = lambda t, x: self.problem.analytical_solution_t(t, x)
        sigma_true = lambda t, x: self.problem.analytical_solution_x(t, x)

        models_dir = os.path.join(self.output_dir, "models")
        logs_dir = os.path.join(self.output_dir, "logs")
        evals_dir = os.path.join(self.output_dir, "evals")
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(evals_dir, exist_ok=True)

        for i, (model, method) in enumerate(pairs, 1):
            print(f"[{i}/{total_runs}] Training configuration: Model={model.kind}, Method={method.kind}")

            built_model = build_model(model)
            current_run_evals = []

            def eval_callback(metrics, state: TrainState):
                if not state.step % self.cfg.training.get("log_every", 1000) == 0:
                    return
                eval_data = evaluate_metrics(
                    model_apply_fn=built_model.apply,
                    params=state.params,
                    method_kind=method.kind,
                    f_fn=f_fn,
                    g_fn=g_fn,
                    analytical_v_fn=v_true,
                    analytical_sigma_fn=sigma_true,
                    domain_points=domain_pts
                )
                eval_data["step"] = state.step
                current_run_evals.append(eval_data)

            start_time = time.time()
            try:
                final_state, logged_metrics = run_training(
                    method,
                    integrator_config,
                    model,
                    trainer_config,
                    sample_input,
                    callback=eval_callback
                )
                elapsed_time = time.time() - start_time

                name = f"{model.kind}-{method.kind}"
                with open(os.path.join(models_dir, f"{name}_iter{iteration}.pkl"), "wb") as f:
                    pickle.dump(final_state.params, f)
                with open(os.path.join(logs_dir, f"{name}_iter{iteration}.pkl"), "wb") as f:
                    pickle.dump(logged_metrics, f)
                with open(os.path.join(evals_dir, f"{name}_iter{iteration}.pkl"), "wb") as f:
                    pickle.dump(current_run_evals, f)

                final_loss = logged_metrics[-1].total_loss if logged_metrics else "N/A"
                print(f"  -> Success! Time: {elapsed_time:.1f}s of which {final_state.total_training_time:.1f}s training time, Final Loss: {final_loss}\n")
            except Exception as exc:
                print(f"  -> Failed: {exc}\n")

class DataProcessor:
    def __init__(self, problem: ProblemDefinition, results_dir: str):
        """"""
        self.problem = problem
        self.results_dir = results_dir
        self.logs_dir = os.path.join(results_dir, "logs")
        self.models_dir = os.path.join(results_dir, "models")
        self.evals_dir = os.path.join(results_dir, "evals")

        self.metrics_data = defaultdict(list)
        self.evals_data = defaultdict(list)
        if os.path.exists(self.logs_dir):
            loaded_count = 0
            for log_file in glob(os.path.join(self.logs_dir, "*.pkl")):
                name = os.path.basename(log_file).replace(".pkl", "")
                base_name = name.rsplit("_iter", 1)[0]
                with open(log_file, "rb") as f:
                    self.metrics_data[base_name].append(pickle.load(f))
                loaded_count += 1
            print(f"Loaded {loaded_count} metric logs from {self.logs_dir}")
        else:
            print(f"Warning: Logs directory not found at {self.logs_dir}")

        if os.path.exists(self.evals_dir):
            loaded_count = 0
            for eval_file in glob(os.path.join(self.evals_dir, "*.pkl")):
                name = os.path.basename(eval_file).replace(".pkl", "")
                base_name = name.rsplit("_iter", 1)[0]
                with open(eval_file, "rb") as f:
                    self.evals_data[base_name].append(pickle.load(f))
                loaded_count += 1
            print(f"Loaded {loaded_count} evaluation logs from {self.evals_dir}")
        else:
            print(f"Warning: Evals directory not found at {self.evals_dir}")

    def plot_eval_metric(self, metric_key: str, ylabel: str, title: str, filename: str):
        import matplotlib.pyplot as plt

        if not self.evals_data:
            print(f"No evaluation data found. Cannot plot {title}.")
            return

        plot_config = self.problem.cfg.get("plot_loss", {})
        show_error = bool(plot_config.get("show_error", True))
        error_low = max(0, min(100, int(plot_config.get("error_low", 0))))
        error_high = max(0, min(100, int(plot_config.get("error_high", 100))))
        window_size = int(plot_config.get("size_windowed_average", 1)) // int(self.problem.cfg.get("training", {}).get("log_every", 1))
        size_windowed_average = max(1, window_size)

        plt.figure(figsize=(10, 6))
        for name, evals_list in self.evals_data.items():
            if not evals_list:
                continue

            # Steps as recorded in evaluation callbacks (these are epoch numbers)
            steps = np.array([e["step"] for e in evals_list[0]])

            # Gather metric values across iterations
            all_vals = []
            for evals in evals_list:
                all_vals.append([e[metric_key] for e in evals])
            all_vals = np.array(all_vals)
            median_val = np.median(all_vals, axis=0)

            # Build a matrix of training times aligned to evaluation steps across iterations
            metrics_lists = self.metrics_data.get(name, [])
            n_iters = len(evals_list)
            n_points = steps.shape[0]
            training_times = np.empty((n_iters, n_points), dtype=float)

            def _times_for_eval_points(evals, metrics_seq):
                # metrics_seq is a list of TrainStepMetrics-like objects
                if not metrics_seq:
                    return np.array([np.nan] * len(evals))
                m_steps = np.array([m.step for m in metrics_seq])
                m_times = np.array([m.training_time for m in metrics_seq], dtype=float)
                out = []
                for s in [e["step"] for e in evals]:
                    if s in m_steps:
                        out.append(float(m_times[m_steps == s][0]))
                    else:
                        # Linear interpolate or extrapolate using nearest two points
                        if m_steps.size >= 2:
                            if s < m_steps.min():
                                i0, i1 = 0, 1
                            elif s > m_steps.max():
                                i0, i1 = -2, -1
                            else:
                                idx = np.searchsorted(m_steps, s)
                                i0, i1 = idx - 1, idx
                            s0, s1 = m_steps[i0], m_steps[i1]
                            t0, t1 = m_times[i0], m_times[i1]
                            if s1 == s0:
                                out.append(float(t1))
                            else:
                                frac = (s - s0) / (s1 - s0)
                                out.append(float(t0 + frac * (t1 - t0)))
                        else:
                            out.append(float(m_times[0]))
                return np.array(out)

            for i, evals in enumerate(evals_list):
                if i < len(metrics_lists):
                    metrics_seq = metrics_lists[i]
                elif metrics_lists:
                    metrics_seq = metrics_lists[0]
                else:
                    metrics_seq = []
                training_times[i, :] = _times_for_eval_points(evals, metrics_seq)

            # If no training time information is available, fall back to steps (keep original behavior)
            if np.all(np.isnan(training_times)):
                x_vals = steps
            else:
                # Use median training time across iterations as the x-axis
                median_time = np.nanmedian(training_times, axis=0)
                x_vals = median_time

            # Windowed averaging if requested
            if size_windowed_average == 1:
                x = x_vals
                y = median_val
                if show_error:
                    low_val = np.percentile(all_vals, error_low, axis=0)
                    high_val = np.percentile(all_vals, error_high, axis=0)
            else:
                w = size_windowed_average
                y = np.convolve(median_val, np.ones(w) / w, mode='valid')
                L = y.shape[0]
                half = (w - 1) // 2
                x = x_vals[half: half + L]
                if show_error:
                    low_val = np.percentile(all_vals, error_low, axis=0)
                    high_val = np.percentile(all_vals, error_high, axis=0)
                    low_val = np.convolve(low_val, np.ones(w) / w, mode='valid')
                    high_val = np.convolve(high_val, np.ones(w) / w, mode='valid')

            line = plt.plot(x, y, label=name)[0]
            if show_error:
                plt.fill_between(x, low_val, high_val, color=line.get_color(), alpha=0.3) # type: ignore ; It is always bound.

        plt.yscale("log")
        # Prefer training time (seconds) on the x-axis; fall back to steps if unavailable
        plt.xlabel("Training Time (s)")
        plt.ylabel(ylabel)
        plt.title(title if size_windowed_average == 1 else f"Windowed {title} (size={size_windowed_average})")
        plt.legend()
        plt.grid(True)

        plots_dir = os.path.join(self.results_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir, filename)
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved to {plot_path}")


def run(
    cfg: DictConfig,
    output_dir: str,
    generate_data: bool=True,
    make_plots: bool=True
):
    """
    Entry point for an experiment.

    Parameters
    ----------
    cfg : DictConfig
        The overridden experiment configuration.
    output_dir : str
        The path where all results/artifacts should be stored.
    prev_data : str | None
        Path to previous run of experiments. This skips execution.
    make_plots : bool
        Should plots be generated.
    """
    problem = ProblemDefinition(cfg)

    if generate_data:
        iterations = cfg.get("iterations", 1)
        print(f"[PHASE 1] Generating Data and Training Models ({iterations} iterations)...")
        trainer = RunTraining(problem, cfg, output_dir)
        trainer.train_multiple(iterations)
        print("[PHASE 1] Complete.\n")

    if make_plots:
        print("[PHASE 2] Processing Data and Generating Plots...")
        processor = DataProcessor(problem, output_dir)
        processor.plot_eval_metric(
            metric_key="total_l2_error",
            ylabel="Total L2 Error (v & sigma)",
            title="L2 Prediction Error vs Training Time",
            filename="l2_error_plot.png"
        )
        processor.plot_eval_metric(
            metric_key="fosls_norm",
            ylabel="FOSLS Space-Time Integral Norm",
            title="FOSLS Norm vs Training Time",
            filename="fosls_norm_plot.png"
        )
        print("[PHASE 2] Complete.\n")

    print("Experiment pipeline finished successfully.")
