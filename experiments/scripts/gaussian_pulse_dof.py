from glob import glob
from itertools import product
from collections import defaultdict
from omegaconf import DictConfig
from utils import (
    build_model_config,
    build_method_config,
    build_integration_config,
    build_trainer_config,
    calculate_dof,
    calculate_fosls_norm
)
from src.trainer import run_training
from src.models import AnyModelConfig, build_model
from src.loss_functions import AlgorithmConfig
from src.integration import get_integrator

import jax.numpy as jnp
import numpy as np
import pandas as pd

import time
import os
import pickle
import dataclasses


class ProblemDefinition:
    """Gaussian pulse benchmark inspired by FGK23 Section 5.2."""
    def __init__(self, cfg: DictConfig):
        self.x_min = float(cfg.integration.get("x_min", 0.0))
        self.x_max = float(cfg.integration.get("x_max", 1.0))

        self.T = float(cfg.problem_params.get("T", 1.0))
        self.c = float(cfg.problem_params.get("c", 1.0))

        # Pulse parameters
        self.kappa = float(cfg.problem_params.get("kappa", 100.0))
        self.mu = float(cfg.problem_params.get("mu", 0.25))

        self.cfg = cfg

    # def solution_u(self, t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    #     xi = x - self.mu - self.c * t
    #     return jnp.exp(-self.kappa * xi**2)

    # def solution_v(self, t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    #     xi = x - self.mu - self.c * t
    #     return 2.0 * self.kappa* self.c * xi * jnp.exp(-self.kappa * xi**2)

    # def solution_sigma(self, t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    #     xi = x - self.mu - self.c * t
    #     return -2.0 * self.kappa * xi * jnp.exp(-self.kappa * xi**2)

    def initial_u(self, x: jnp.ndarray) -> jnp.ndarray:
        xi = x - self.mu
        return jnp.exp(-self.kappa * xi**2)

    def initial_v(self, x: jnp.ndarray) -> jnp.ndarray:
        xi = x - self.mu
        return 2.0 * self.kappa* self.c * xi * jnp.exp(-self.kappa * xi**2)

    def initial_sigma(self, x: jnp.ndarray) -> jnp.ndarray:
        xi = x - self.mu
        return -2.0 * self.kappa * xi * jnp.exp(-self.kappa * xi**2)


    def source_f(self, t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(x)

    def source_g(self, t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros((1,))

    def get_sample_input(self) -> jnp.ndarray:
        return jnp.asarray([[0.0, 0.5]], dtype=jnp.float32)


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
        self.v0 = lambda v: self.problem.initial_v(jnp.array(v[1]))
        self.sigma0 = lambda v: jnp.array(
            [self.problem.initial_sigma(jnp.array(v[1]))]
        )
        self.u0 = lambda v: self.problem.initial_u(jnp.array(v[1]))
        self.ut0 = lambda v: self.problem.initial_v(jnp.array(v[1]))
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

    def _read_combinations(self) -> list[tuple[list[AnyModelConfig], AlgorithmConfig]]:
        combinations = self.cfg.get("combinations", ())

        all_models = self.cfg.get("models", "")
        all_methods = self.cfg.get("methods", "")

        pairs = []
        for method_name, model_name in combinations:
            method = all_methods.get(method_name, {})
            heads = method.get("output_heads", "")

            # We want to save all configs
            model_versions = all_models.get(model_name, {})
            models_list = []
            model_config = None
            for model in model_versions:
                model_config = build_model_config(model_name, model, heads)
                models_list.append(model_config)

            assert model_config
            method_config = build_method_config(
                method_name,
                method,
                model_config,
                self.wave_functions,
                self.cfg.get("integration")
            )

            pairs.append((models_list, method_config))

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
        total_runs = sum(len(p[0]) for p in pairs)

        print(f"--- Starting Training Phase (Iteration {iteration+1}) ---")
        print(f"Total configurations to train: {total_runs}")
        init_lr = self.cfg.get("training", {}).get("learning_rate", {}).get("init_value", "Unknown")
        print(f"Epochs: {trainer_config.epochs}, Initial LR: {init_lr}, Seed: {trainer_config.seed}\n")

        models_dir = os.path.join(self.output_dir, "models")
        logs_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

        i = 0
        for models, method in pairs:
            for model in models:
                dof = calculate_dof(2, model)
                i += 1
                print(f"[{i}/{total_runs}] Training configuration: Model={model.kind}, Method={method.kind}, dof={dof}")

                start_time = time.time()
                final_state, logged_metrics = run_training(
                    method,
                    integrator_config,
                    model,
                    trainer_config,
                    sample_input,
                )
                elapsed_time = time.time() - start_time

                name = f"{model.kind}-{method.kind}"
                with open(os.path.join(models_dir, f"{name}_{dof}_iter{iteration}.pkl"), "wb") as f:
                    pickle.dump(final_state.params, f)
                with open(os.path.join(logs_dir, f"{name}_{dof}_iter{iteration}.pkl"), "wb") as f:
                    pickle.dump(logged_metrics, f)

                print(f"  -> Success! Time: {elapsed_time:.1f}s\n")


class DataProcessor:
    def __init__(self, problem: ProblemDefinition, results_dir: str):
        """"""
        self.problem = problem
        self.results_dir = results_dir
        self.models_dir = os.path.join(results_dir, "models")

        self.model_params = defaultdict(lambda: defaultdict(list))

        if os.path.exists(self.models_dir):
            loaded_count = 0
            # Expected format: {name}_{dof}_iter{iter}.pkl
            for model_file in glob(os.path.join(self.models_dir, "*.pkl")):
                filename = os.path.basename(model_file).replace(".pkl", "")

                try:
                    # Split from right to handle names with underscores
                    base_part, iter_part = filename.rsplit("_iter", 1)
                    name, dof_str = base_part.rsplit("_", 1)
                    dof = int(dof_str)

                    with open(model_file, "rb") as f:
                        params = pickle.load(f)
                        self.model_params[name][dof].append(params)
                    loaded_count += 1
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse filename format for {filename}")

            print(f"Loaded {loaded_count} model parameter sets from {self.models_dir}")
        else:
            print(f"Warning: Models directory not found at {self.models_dir}")

    def plot_dof_vs_loss(self, ylabel: str, title: str, filename: str):
        import matplotlib.pyplot as plt

        if not self.model_params:
            print(f"No model parameters found. Cannot plot {title}.")
            return

        eval_cfg_data = self.problem.cfg.get("plot_integration", self.problem.cfg.get("callback_integration"))
        eval_cfg = build_integration_config(eval_cfg_data)
        eval_integrator = get_integrator(eval_cfg)

        plot_config = self.problem.cfg.get("plot_loss", {})
        show_error = bool(plot_config.get("show_error", True))
        error_low = max(0, min(100, int(plot_config.get("error_low", 25))))
        error_high = max(0, min(100, int(plot_config.get("error_high", 75))))

        f = lambda v: self.problem.source_f(jnp.array(v[0]), jnp.array(v[1]))
        g = lambda v: self.problem.source_g(jnp.array(v[0]), jnp.array(v[1]))
        v0 = lambda v: self.problem.initial_v(jnp.array(v[1]))
        sigma0 = lambda v: jnp.array(
            [self.problem.initial_sigma(jnp.array(v[1]))]
        )

        plt.figure(figsize=(10, 7))

        all_models_cfg = self.problem.cfg.models
        all_methods_cfg = self.problem.cfg.methods

        all_csv = []

        for name in sorted(self.model_params.keys()):
            model_kind, method_kind = name.split("-")

            dof_points = []
            loss_central = []
            loss_lows = []
            loss_highs = []

            sorted_dofs = sorted(self.model_params[name].keys())
            for dof in sorted_dofs:
                runs_params = self.model_params[name][dof]
                run_losses = []

                current_model_cfg = None
                heads = all_methods_cfg.get(method_kind, {}).get("output_heads", {"u": 1})

                for variant in all_models_cfg.get(model_kind, []):
                    test_cfg = build_model_config(model_kind, variant, heads)
                    if calculate_dof(2, test_cfg) == dof:
                        current_model_cfg = test_cfg
                        break

                if current_model_cfg is None:
                    print(f"Warning: Could not find config variant for {name} with DOF {dof}. Skipping.")
                    continue

                model_inst = build_model(current_model_cfg)
                ic_weight = all_methods_cfg.get(method_kind, {}).get("ic_weight", 1.0)

                for params in runs_params:
                    loss = calculate_fosls_norm(
                        model_inst.apply,
                        params,
                        method_kind,
                        f,
                        g,
                        v0,
                        sigma0,
                        eval_integrator,
                        ic_weight=ic_weight
                    )
                    run_losses.append(loss)

                if not run_losses:
                    print(f"Warning: No losses computed for {name} at DOF {dof}. Skipping.")
                    continue

                loss_central.append(np.median(run_losses))
                loss_lows.append(np.percentile(run_losses, error_low))
                loss_highs.append(np.percentile(run_losses, error_high))
                dof_points.append(dof)

                all_csv.append(
                    pd.DataFrame({
                        "plot-name": name,
                        "dof": dof,
                        "median": loss_central,
                        "low": loss_lows,
                        "high": loss_highs,
                    })
                )

            if len(dof_points) == 0:
                continue

            line = plt.plot(dof_points, loss_central, 'o-', label=name)[0]

            if show_error:
                lower_error = np.asarray(loss_central) - np.asarray(loss_lows)
                upper_error = np.asarray(loss_highs) - np.asarray(loss_central)
                plt.errorbar(
                    dof_points,
                    loss_central,
                    yerr=np.vstack([lower_error, upper_error]),
                    fmt='none',
                    ecolor=line.get_color(),
                    elinewidth=1.2,
                    capsize=4,
                    alpha=0.8,
                )

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Degrees of Freedom (DOF)", fontsize=22)
        plt.ylabel(ylabel, fontsize=22)
        plt.title(title, fontsize=24)
        plt.legend(fontsize=22)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid(True, which="both", ls="-", alpha=0.5)

        plots_dir = os.path.join(self.results_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir, filename)
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

        csv_dir = os.path.join(self.results_dir, "csv")
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, filename.replace(".png", ".csv"))
        all_csv = pd.concat(all_csv, ignore_index=True)
        all_csv.to_csv(csv_path, index=False)
        print(f"CSV saved to {csv_path}")

        pdf_dir = os.path.join(self.results_dir, "pdf")
        os.makedirs(pdf_dir, exist_ok=True)
        pdf_path = os.path.join(pdf_dir, filename.replace(".png", ".pdf"))
        plt.savefig(pdf_path)
        print(f"PDF-plot saved to {pdf_path}")

        plt.close()


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
        processor.plot_dof_vs_loss(
            ylabel="Error Estimator",
            title="Error Estimator Convergence vs DOF",
            filename="fosls_norm_vs_dof.png"
        )
        print("[PHASE 2] Complete.\n")

    print("Experiment pipeline finished successfully.")
