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
    make_first_order_model
)
from src.trainer import run_training
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
    """Run training of experiment 1. Saves artifacts to output_dir/"""

    def __init__(self, problem: ProblemDefinition, cfg: DictConfig):
        """
        Parameters
        ----------
        problem : ProblemDefinition
            Definition of the handled wave equation
        cfg : DictConfig
            The overridden experiment configuration.
        """
        self.cfg = cfg
        self.problem = problem

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
        self.results: dict = defaultdict(list)

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

        for i, (model, method) in enumerate(pairs, 1):
            print(f"[{i}/{total_runs}] Training configuration: Model={model.kind}, Method={method.kind}")

            start_time = time.time()
            try:
                final_state, logged_metrics = run_training(
                    method,
                    integrator_config,
                    model,
                    trainer_config,
                    sample_input
                )
                elapsed_time = time.time() - start_time
                self.results[f"{model.kind}-{method.kind}"].append({
                    "state": final_state,
                    "metrics": logged_metrics
                })

                final_loss = logged_metrics[-1].total_loss if logged_metrics else "N/A"
                print(f"  -> Success! Time: {elapsed_time:.1f}s, Final Loss: {final_loss}\n")
            except Exception as exc:
                print(f"  -> Failed: {exc}\n")

    def save_data(self, output_dir: str):
        """
        Parameters
        ----------
        output_dir : str
            The path where all results/artifacts should be stored.
        """
        models_dir = os.path.join(output_dir, "models")
        logs_dir = os.path.join(output_dir, "logs")
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

        for name, data_list in self.results.items():
            for i, data in enumerate(data_list):
                # Save final state params
                with open(os.path.join(models_dir, f"{name}_iter{i}.pkl"), "wb") as f:
                    pickle.dump(data["state"].params, f)

                # Save metrics
                with open(os.path.join(logs_dir, f"{name}_iter{i}.pkl"), "wb") as f:
                    pickle.dump(data["metrics"], f)

        print(f"Saved artifacts to {output_dir}")



class DataProcessor:
    def __init__(self, problem: ProblemDefinition, results_dir: str):
        """"""
        self.problem = problem
        self.results_dir = results_dir
        self.logs_dir = os.path.join(results_dir, "logs")
        self.models_dir = os.path.join(results_dir, "models")

        self.metrics_data = defaultdict(list)
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

    def plot_loss(self):
        import matplotlib.pyplot as plt

        if not self.metrics_data:
            print("No metrics data found. Cannot plot loss.")
            return

        plot_loss_config = self.problem.cfg.get("plot_loss", {})

        show_error = bool(plot_loss_config.get("show_error", True))
        error_low = max(100, min(0, int(plot_loss_config.get("error_low", 0))))
        error_high = max(100, min(0, int(plot_loss_config.get("error_high", 100))))
        window_size = int(plot_loss_config.get("size_windowed_average", 1)) // int(self.problem.cfg.get("training", {}).get("log_every", 1))
        size_windowed_average = max(1, window_size)

        plt.figure(figsize=(10, 6))
        for name, metrics_list in self.metrics_data.items():
            if not metrics_list:
                continue

            steps = [m.step for m in metrics_list[0]]
            all_losses = []
            for metrics in metrics_list:
                all_losses.append([m.total_loss for m in metrics])

            all_losses = np.array(all_losses)
            median_loss = np.median(all_losses, axis=0)
            windowed_median_loss = np.convolve(median_loss, np.ones(size_windowed_average)/size_windowed_average, mode='valid')[1:]
            line = plt.plot(steps[size_windowed_average//2:-size_windowed_average//2], windowed_median_loss, label=name)[0]

            if show_error:
                low_loss = np.percentile(all_losses, error_low, axis=0)
                high_loss = np.percentile(all_losses, error_high, axis=0)
                plt.fill_between(steps, low_loss, high_loss, color=line.get_color(), alpha=0.3)

        plt.yscale("log")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss" if size_windowed_average == 1 else f"Windowed Training Loss (size={size_windowed_average})")
        plt.legend()
        plt.grid(True)

        plots_dir = os.path.join(self.results_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir, "loss_plot.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Loss plot saved to {plot_path}")

    def plot_sls_error(self):
        import matplotlib.pyplot as plt

        if not os.path.exists(self.models_dir):
            print("No models directory found. Cannot plot SLS error.")
            return

        plot_loss_config = self.problem.cfg.get("plot_real_space_time_error", {})

        show_error = bool(plot_loss_config.get("show_error", True))
        error_low = min(100, max(0, int(plot_loss_config.get("error_low", 0))))
        error_high = min(100, max(0, int(plot_loss_config.get("error_high", 100))))

        # Time and space discretization
        t_vals = np.linspace(0, self.problem.T, 50)
        x_vals = np.linspace(self.problem.x_min, self.problem.x_max, 50)
        dt = t_vals[1] - t_vals[0]
        dx = x_vals[1] - x_vals[0]

        plt.figure(figsize=(10, 6))

        methods = self.problem.cfg.get("methods", [])
        models = self.problem.cfg.get("models", [])
        combinations = []
        for method in methods:
            for model in models:
                combinations.append((model, method))

        for (model_cfg, method_cfg) in combinations:
            heads = method_cfg.get("output_heads", "")
            model_obj_cfg = build_model_config(model_cfg, heads)
            model_obj = build_model(model_obj_cfg)
            name = f"{model_cfg.name}-{method_cfg.name}"

            model_files = glob(os.path.join(self.models_dir, f"{name}_iter*.pkl"))
            if not model_files:
                continue

            first_order_apply = make_first_order_model(model_obj.apply, method_cfg.name)
            batched_apply = jax.jit(jax.vmap(first_order_apply, in_axes=(None, None, 0)))

            errors_per_iter = []
            for m_file in model_files:
                with open(m_file, "rb") as f:
                    params = pickle.load(f)

                iter_errors_over_time = []
                for t in t_vals:
                    # Analytical solution terms
                    true_v = self.problem.analytical_solution_t(t, jnp.array(x_vals))
                    true_sigma = self.problem.analytical_solution_x(t, jnp.array(x_vals))
                    true_vector = jnp.stack([true_v, true_sigma], axis=-1)

                    # Model prediction (v, sigma)
                    pred_vector = batched_apply(params, t, x_vals)

                    # L2 norm over space at current time point t
                    sq_diff = jnp.sum((pred_vector - true_vector)**2, axis=-1)
                    l2_norm = jnp.sqrt(jnp.sum(sq_diff) * dx)
                    iter_errors_over_time.append(l2_norm)

                errors_per_iter.append(iter_errors_over_time)

            if not errors_per_iter:
                continue

            errors_per_iter = np.array(errors_per_iter)
            median_error = np.median(errors_per_iter, axis=0)
            line = plt.plot(t_vals, median_error, label=name)[0]

            if show_error:
                low_error = np.percentile(errors_per_iter, error_low, axis=0)
                high_error = np.percentile(errors_per_iter, error_high, axis=0)
                plt.fill_between(t_vals, low_error, high_error, color=line.get_color(), alpha=0.3)

        plt.xlabel("Time (t)")
        plt.ylabel("Avg log L2 SLS Error Norm over x")
        plt.yscale("log")
        plt.title("Error Plot over Space-Time")
        plt.legend()
        plt.grid(True)

        plots_dir = os.path.join(self.results_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir, "space_time_error_plot.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Error plot saved to {plot_path}")


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

    print("\n==========================================")
    print(f"   Executing Experiment 1 Pipeline")
    print("==========================================\n")

    if generate_data:
        iterations = cfg.get("iterations", 1)
        print(f"[PHASE 1] Generating Data and Training Models ({iterations} iterations)...")
        trainer = RunTraining(problem, cfg)
        trainer.train_multiple(iterations)
        trainer.save_data(output_dir)
        print("[PHASE 1] Complete.\n")

    if make_plots:
        print("[PHASE 2] Processing Data and Generating Plots...")
        processor = DataProcessor(problem, output_dir)
        processor.plot_loss()
        processor.plot_sls_error()
        print("[PHASE 2] Complete.\n")

    print("Experiment pipeline finished successfully.")
