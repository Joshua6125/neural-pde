from glob import glob
from itertools import product
from collections import defaultdict
from omegaconf import DictConfig
from utils import (
    build_model_config,
    build_method_config,
    build_integration_config,
    build_trainer_config,
    calculate_fosls_norm,
    calculate_true_l2_error,
    calculate_true_v_error,
    make_second_order_model
)
from src.trainer import TrainState, TrainStepMetrics, run_training
from src.models import AnyModelConfig, build_model
from src.loss_functions import AlgorithmConfig
from src.integration import get_integrator

import jax.numpy as jnp
import numpy as np

import time
import os
import pickle
import dataclasses
import jax

class ProblemDefinition:
    """Analytical problem definition of the wave equation from Experiment 5.1."""

    def __init__(self, cfg: DictConfig):
        """
        Parameters
        ----------
        cfg : DictConfig
            The overridden experiment configuration.
        """
        # Experiment 5.1 uses the unit cube Q = (0,1) x (0,1)
        self.x_min = float(cfg.integration.get("x_min", 0.0))
        self.x_max = float(cfg.integration.get("x_max", 1.0))
        self.T = float(cfg.problem_params.get("T", 1.0))
        self.cfg = cfg

    def solution_u(self, t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Compute the manufactured analytical solution u(t, x) = 0.5 * t^2 * sin(pi * x)."""
        t_array = jnp.asarray(t)
        x_array = jnp.asarray(x)

        # Scaling spatial coordinates dynamically to the domain limits
        x_scaled = (x_array - self.x_min) / (self.x_max - self.x_min)

        return 0.5 * (t_array ** 2) * jnp.sin(jnp.pi * x_scaled)

    def solution_v(self, t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Time derivative of the analytical solution (v = dt u)."""
        t_array = jnp.asarray(t)
        x_array = jnp.asarray(x)
        x_scaled = (x_array - self.x_min) / (self.x_max - self.x_min)

        return t_array * jnp.sin(jnp.pi * x_scaled)

    def solution_sigma(self, t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Spatial derivative of the analytical solution (sigma = dx u)."""
        t_array = jnp.asarray(t)
        x_array = jnp.asarray(x)
        x_scaled = (x_array - self.x_min) / (self.x_max - self.x_min)
        dx_factor = jnp.pi / (self.x_max - self.x_min)

        return 0.5 * (t_array ** 2) * jnp.cos(jnp.pi * x_scaled) * dx_factor

    def source_f(self, t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Closed-form source term (f = dtt u - dxx u)."""
        t_array = jnp.asarray(t)
        x_array = jnp.asarray(x)
        x_scaled = (x_array - self.x_min) / (self.x_max - self.x_min)
        dx_factor = jnp.pi / (self.x_max - self.x_min)

        dtt_u = jnp.sin(jnp.pi * x_scaled)
        dxx_u = -0.5 * (t_array ** 2) * jnp.sin(jnp.pi * x_scaled) * (dx_factor ** 2)

        return dtt_u - dxx_u

    def source_g(self, t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
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
        self.f = lambda v: self.problem.source_f(jnp.array(v[0]), jnp.array(v[1]))
        self.g = lambda v: self.problem.source_g(jnp.array(v[0]), jnp.array(v[1]))
        self.v0 = lambda v: self.problem.solution_v(jnp.array(v[0]), jnp.array(v[1]))
        self.sigma0 = lambda v: jnp.array(
            [self.problem.solution_sigma(jnp.array(v[0]), jnp.array(v[1]))]
        )
        self.u0 = lambda v: self.problem.solution_u(jnp.array(v[0]), jnp.array(v[1]))
        self.ut0 = lambda v: self.problem.solution_v(jnp.array(v[0]), jnp.array(v[1]))
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

    def _read_combinations(self) -> list[tuple[AnyModelConfig, AlgorithmConfig]]:
        combinations = self.cfg.get("combinations", ())

        all_models = self.cfg.get("models", "")
        all_methods = self.cfg.get("methods", "")

        pairs = []
        for method_name, model_name in combinations:
            method = all_methods.get(method_name, {})
            model = all_models.get(model_name, {})

            heads = method.get("output_heads", "")
            model_config = build_model_config(model_name, model, heads)
            method_config = build_method_config(
                method_name,
                method,
                model_config,
                self.wave_functions,
                self.cfg.get("integration")
            )

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

        pairs = self._read_combinations()
        total_runs = len(pairs)

        print(f"--- Starting Training Phase (Iteration {iteration+1}) ---")
        print(f"Total configurations to train: {total_runs}")
        init_lr = self.cfg.get("training", {}).get("learning_rate", {}).get("init_value", "Unknown")
        print(f"Epochs: {trainer_config.epochs}, Initial LR: {init_lr}, Seed: {trainer_config.seed}\n")

        integrator_config = build_integration_config(self.cfg.callback_integration)
        integrator = get_integrator(integrator_config)

        models_dir = os.path.join(self.output_dir, "models")
        logs_dir = os.path.join(self.output_dir, "logs")
        evals_dir = os.path.join(self.output_dir, "evals")
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(evals_dir, exist_ok=True)

        for i, (model, method) in enumerate(pairs, 1):
            print(f"[{i}/{total_runs}] Training configuration: Model={model.kind}, Method={method.kind}")

            built_model = build_model(model)
            current_run_evals = {
                "fosls_loss": [],
                "true_v_error": [],
                "true_l2_error": []
            }

            def eval_callback(metrics: TrainStepMetrics, state: TrainState):
                true_l2_error = calculate_true_l2_error(
                    model_apply_fn=built_model.apply,
                    params=state.params,
                    method_kind=method.kind,
                    v_sol=self.problem.solution_v,
                    sigma_sol=self.problem.solution_sigma,
                    integrator=integrator
                )

                true_v_error = calculate_true_v_error(
                    model_apply_fn=built_model.apply,
                    params=state.params,
                    method_kind=method.kind,
                    v_sol=self.problem.solution_v,
                    sigma_sol=self.problem.solution_sigma,
                    integrator=integrator
                )

                if method.kind == "fosls":
                    total_loss = metrics.total_loss
                else:
                    total_loss = calculate_fosls_norm(
                        model_apply_fn=built_model.apply,
                        params=state.params,
                        method_kind=method.kind,
                        f_fn=self.f,
                        g_fn=self.g,
                        v0_fn=self.v0,
                        sigma0_fn=self.sigma0,
                        integrator=integrator
                    )

                current_run_evals["fosls_loss"].append(total_loss)
                current_run_evals["true_l2_error"].append(true_l2_error)
                current_run_evals["true_v_error"].append(true_v_error)

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
            except Exception as e:
                print(f"  -> Training {model.kind}-{method.kind} failed with error: {e}\n")
                continue

            name = f"{model.kind}-{method.kind}"
            with open(os.path.join(models_dir, f"{name}_iter{iteration}.pkl"), "wb") as f:
                pickle.dump(final_state.params, f)
            with open(os.path.join(logs_dir, f"{name}_iter{iteration}.pkl"), "wb") as f:
                pickle.dump(logged_metrics, f)
            with open(os.path.join(evals_dir, f"{name}_iter{iteration}.pkl"), "wb") as f:
                pickle.dump(current_run_evals, f)

            print(f"  -> Success! Time: {elapsed_time:.1f}s\n")


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

    def plot_vs_time(self, ylabel: str, title: str, filename: str, y_type: str, cutoff_time: float|None=None):
        import matplotlib.pyplot as plt

        if not self.evals_data:
            print(f"No evaluation data found. Cannot plot {title}.")
            return
        if not self.metrics_data:
            print(f"No metrics data found. Cannot plot {title}.")
            return

        plot_config = self.problem.cfg.get(f"plot_{y_type}", {})
        show_error = bool(plot_config.get("show_error", True))
        error_low = max(0, min(100, int(plot_config.get("error_low", 0))))
        error_high = max(0, min(100, int(plot_config.get("error_high", 100))))
        grid_resolution = max(2, int(plot_config.get("grid_resolution", 1000)))

        plt.figure(figsize=(10, 7))
        for name, evals in self.evals_data.items():
            all_vals = [evals[k][y_type] for k in range(len(evals))]

            metrics = self.metrics_data[name]

            all_training_times = [[m.training_time for m in run_metrics] for run_metrics in metrics]

            min_time = min(times[0] for times in all_training_times if len(times) > 0)
            max_time = max(times[-1] for times in all_training_times if len(times) > 0)

            if cutoff_time:
                max_time = min(cutoff_time, max_time)

            common_time_grid = np.linspace(min_time, max_time, grid_resolution)

            interpolated_runs = []
            for run_times, run_vals in zip(all_training_times, all_vals):
                run_times_arr = np.array(run_times)
                run_vals_arr = np.array(run_vals)

                if cutoff_time:
                    valid_run_times = run_times_arr < cutoff_time
                    run_times_arr = np.ma.masked_where(valid_run_times, run_times_arr)
                    run_vals_arr = np.ma.masked_where(valid_run_times, run_vals_arr)

                interp_vals = np.interp(common_time_grid, run_times_arr, run_vals_arr)
                interpolated_runs.append(interp_vals)

            interpolated_matrix = np.vstack(interpolated_runs)

            median_val = np.median(interpolated_matrix, axis=0)

            line = plt.plot(common_time_grid, median_val, label=name)[0]

            if show_error:
                low_val = np.percentile(interpolated_matrix, error_low, axis=0)
                high_val = np.percentile(interpolated_matrix, error_high, axis=0)
                plt.fill_between(common_time_grid, low_val, high_val, color=line.get_color(), alpha=0.3)

        plt.yscale("log")
        plt.xlabel("Training Time (seconds)", fontsize=22)
        plt.ylabel(ylabel, fontsize=22)
        plt.title(title, fontsize=24)
        plt.legend(fontsize=22)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid(True)

        plots_dir = os.path.join(self.results_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir, filename)
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved to {plot_path}")

    def plot_specific_times(self, time: float):
        import matplotlib.pyplot as plt

        if not os.path.exists(self.models_dir):
            print("No models directory found.")
            return

        plot_loss_config = self.problem.cfg.get("plot_specific_t", {})

        show_error = bool(plot_loss_config.get("show_error", True))
        error_low = min(100, max(0, int(plot_loss_config.get("error_low", 0))))
        error_high = min(100, max(0, int(plot_loss_config.get("error_high", 100))))

        x_vals = jnp.linspace(self.problem.x_min, self.problem.x_max, 50)

        combinations = self.problem.cfg.get("combinations", [])
        all_models = self.problem.cfg.get("models", {})
        all_methods = self.problem.cfg.get("methods", {})

        plt.figure(figsize=(10, 6))

        batch_inputs = jnp.stack([jnp.full_like(x_vals, float(time)), x_vals], axis=1)
        plotted_any = False

        for method_name, model_name in combinations:
            method_cfg = all_methods.get(method_name, {})
            model_cfg = all_models.get(model_name, {})
            if not method_cfg or not model_cfg:
                continue

            heads = method_cfg.get("output_heads", "")
            model_obj_cfg = build_model_config(model_name, model_cfg, heads)
            model_obj = build_model(model_obj_cfg)
            name = f"{model_name}-{method_name}"

            model_files = glob(os.path.join(self.models_dir, f"{name}_iter*.pkl"))
            if not model_files:
                continue

            u0 = lambda v: self.problem.solution_u(jnp.array(0.0), jnp.array(v[1]))
            second_order_apply = make_second_order_model(model_obj.apply, method_name, u0_fn=u0)
            batched_apply = jax.jit(jax.vmap(second_order_apply, in_axes=(None, 0)))

            combo_predictions = []
            for m_file in model_files:
                with open(m_file, "rb") as f:
                    params = pickle.load(f)

                pred_vector = batched_apply(params, batch_inputs)
                combo_predictions.append(np.asarray(pred_vector).squeeze())

            if not combo_predictions:
                continue

            predictions_matrix = np.vstack(combo_predictions)
            mean_prediction = np.mean(predictions_matrix, axis=0)

            line = plt.plot(x_vals, mean_prediction, label=name)[0]
            plotted_any = True

            if show_error:
                low_error = np.percentile(predictions_matrix, error_low, axis=0)
                high_error = np.percentile(predictions_matrix, error_high, axis=0)

                plt.fill_between(x_vals, low_error, high_error, color=line.get_color(), alpha=0.3)

        if not plotted_any:
            print("No predictions found. Cannot plot specific times.")
            return

        plt.plot(
            x_vals,
            self.problem.solution_u(jnp.atleast_1d(time), x_vals),
            label="Exact solution",
            linestyle="--",
            color="black",
        )

        plt.xlabel("x", fontsize=12)
        plt.ylabel("u(x)", fontsize=12)
        plt.title(f"Error of predicted displacement at t={time}s", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True)

        plots_dir = os.path.join(self.results_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir, f"error_pred_at_time_{time}.png")
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
        processor.plot_vs_time(
            ylabel="Error Estimator",
            title="Error Estimator vs Training Time",
            filename="fosls_norm_plot.png",
            y_type="fosls_loss",
            cutoff_time=80.0
        )
        processor.plot_vs_time(
            ylabel="True L2 Error",
            title="True L2 Error vs Training Time",
            filename="true_ls_error.png",
            y_type="true_l2_error",
            cutoff_time=80.0
        )
        processor.plot_vs_time(
            ylabel="True Error",
            title="True Error vs Training Time",
            filename="true_v_error.png",
            y_type="true_v_error",
            cutoff_time=80.0
        )
        processor.plot_specific_times(0.0)
        processor.plot_specific_times(0.333)
        processor.plot_specific_times(0.666)
        processor.plot_specific_times(1.0)
        print("[PHASE 2] Complete.\n")

    print("Experiment pipeline finished successfully.")
