"""
The purpose of this experiment is to compar pairs of loss functions and neural architectures.

In particular, all loss functions are compared to one another with MLP.
"""

from itertools import product
from omegaconf import DictConfig
from utils import (
    build_model_config,
    build_method_config,
    build_integration_config,
    build_trainer_config,
)
from src.trainer import run_training
from src.models import AnyModelConfig
from src.loss_functions import AlgorithmConfig


import jax.numpy as jnp

import time
import os


class ProblemDefinition:
    """Analytical problem definition of a simple wave equation."""

    def __init__(self, cfg: DictConfig):
        """
        Parameters
        ----------
        cfg : DictConfig
            The overridden experiment configuration.
        """

        self.L = float(cfg.problem_params.get("L", 1.0))
        self.T = float(cfg.problem_params.get("T", 1.0))
        self.c = float(cfg.problem_params.get("c", 1.0))

    def analytical_solution(self, t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Compute the manufactured analytical solution."""
        t_array = jnp.asarray(t)
        x_array = jnp.asarray(x)

        t_term = jnp.sin(jnp.pi * t_array / self.T)
        x_term = jnp.sin(jnp.pi * (x_array + self.L) / (2 * self.L))
        result = t_term * x_term

        return jnp.where(jnp.isclose(t_array / self.T, 1.0, atol=1e-10), 0.0, result)

    def analytical_solution_t(self, t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Time derivative of the manufactured analytical solution."""
        return (
            jnp.pi / self.T
        ) * jnp.cos(jnp.pi * t / self.T) * jnp.sin(jnp.pi * (x + self.L) / (2 * self.L))

    def analytical_solution_x(self, t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Spatial derivative of the manufactured analytical solution."""
        return (
            jnp.sin(jnp.pi * t / self.T)
            * jnp.cos(jnp.pi * (x + self.L) / (2 * self.L))
            * (jnp.pi / (2 * self.L))
        )

    def source_function(self, t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Closed-form source term for the manufactured PDE."""
        u = self.analytical_solution(t, x)
        coeff = -jnp.pi**2 / self.T**2 + self.c**2 * jnp.pi**2 / (4 * self.L**2)
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
        self.f = lambda v: self.problem.source_function(jnp.array(v[0]), jnp.array(v[1])),
        self.g = lambda v: self.problem.zero_vector_source(jnp.array(v[0]), jnp.array(v[1]))
        self.v0 = lambda v: self.problem.analytical_solution_t(jnp.array(v[0]), jnp.array(v[1])),
        self.sigma0 = lambda v: jnp.array(
            [self.problem.analytical_solution_x(jnp.array(v[0]), jnp.array(v[1]))]
        ),
        self.u0 = lambda v: self.problem.analytical_solution(jnp.array(v[0]), jnp.array(v[1])),
        self.ut0 = lambda v: self.problem.analytical_solution_t(jnp.array(v[0]), jnp.array(v[1])),
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

    def train_all(self):
        """Train all model-method combinations"""
        sample_input = self.problem.get_sample_input()

        integrator_data = self.cfg.get("integration", "")
        integrator_config = build_integration_config(integrator_data)

        trainer_data = self.cfg.get("training")
        trainer_config = build_trainer_config(trainer_data)

        pairs = self._generate_combinations()

        for model, method in pairs:
            print(f"Starting training: {model.kind}-{method.kind}")

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

                print(f"Success! {elapsed_time:.1f}s")
            except Exception as exc:
                print(f"Failed: {exc}")

    def save_data(self, output_dir: str):
        """
        Parameters
        ----------
        output_dir : str
            The path where all results/artifacts should be stored.
        """

        # Should save generated artifacts here, i.e. the models and training logs

        artifact_path = os.path.join(output_dir, "artifacts/")

        print(f"Saved artifact to {artifact_path}")



class DataProcessor:
    def __init__(self, problem: ProblemDefinition, results_dir: str):
        """"""

        # Should read the config and such from the results_dir here and save them as properties


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
        trainer = RunTraining(problem, cfg)
        trainer.train_all()
        trainer.save_data(output_dir)


    if make_plots:
        processor = DataProcessor(problem, output_dir)

        # Make plots and do analysis here
