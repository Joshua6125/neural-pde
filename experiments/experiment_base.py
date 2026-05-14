"""
Base experiment class: implements the common 6-step pipeline.

Experiments inherit from this class and provide a high-level YAML config,
while this base class bridges that config to the source-native dataclasses
used by the actual training stack.
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import jax.numpy as jnp
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.config_loader import ConfigLoader
from experiments.config.schema import ExperimentCombination, ExperimentConfig
from experiments.domains import get_domain
from experiments.results_manager import ResultsManager, RunManager
from src.cli import run_training
from src.integration import MonteCarloConfig
from src.train import TrainConfig as SourceTrainConfig


class BaseExperiment:
    """Base class for all experiments.

    Subclasses can override get_config_file() if they want a different YAML
    entry point.
    """

    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.config = config or self._load_config()
        self.domain = get_domain(self.config.domain)(**self.config.problem_params)
        self.results_manager = ResultsManager()
        self.run_manager: Optional[RunManager] = None
        self.train_config: Optional[SourceTrainConfig] = None
        self.integration_config: Optional[MonteCarloConfig] = None
        self.generate_plots = True
        self.sample_input = jnp.asarray(self.domain.get_sample_input(), dtype=jnp.float32)

        print(f"\n{'=' * 60}")
        print(f"Experiment: {self.config.name}")
        print(f"Domain: {self.domain.name} - {self.domain.description}")
        print(f"{'=' * 60}\n")

    def _load_config(self) -> ExperimentConfig:
        loader = ConfigLoader(str(Path(__file__).resolve().parent / "config"))
        return loader.load(self.get_config_file())

    def get_config_file(self) -> str:
        return "experiment_1.yaml"

    def execute(self, run_id: Optional[str] = None) -> RunManager:
        self.run_manager = self.results_manager.create_run(
            config=self.config,
            domain=self.config.domain,
            run_id=run_id,
        )
        print(f"Run ID: {self.run_manager.run_id}\n")

        self._setup()
        test_points, reference_solutions = self._prepare_test_data()
        combinations = self._build_combinations()
        metrics_by_method = self._train_all(combinations)
        if self.generate_plots:
            self._evaluate_and_visualise(
                test_points=test_points,
                reference_solutions=reference_solutions,
                metrics_by_method=metrics_by_method,
            )
        else:
            print("Step 6: Visualisation skipped (--no-plots)")

        print(f"\nExperiment complete. Results saved to {self.run_manager.run_dir}")
        return self.run_manager

    def _setup(self) -> None:
        print("Step 1: Setup")
        self.config.validate()
        print("  Configuration validated")
        print(f"  Methods: {[m.get('name', '') for m in self.config.methods]}")
        print(f"  Models: {[m.get('name', '') for m in self.config.models]}")
        print(f"  Learning rate spec: {self.config.training.learning_rate}")
        print()

    def _prepare_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        print("Step 2: Prepare test data")

        test_params = self.config.test_data_params or {}
        n_time = int(test_params.get("n_time", 20))
        n_space = int(test_params.get("n_space", 80))
        seed = int(test_params.get("seed", 42))

        test_points, reference_solutions = self.domain.get_test_data(
            n_time=n_time,
            n_space=n_space,
            seed=seed,
        )

        assert self.run_manager is not None, "No run manager available."
        print(f"  Test data: {len(test_points)} points")
        print(f"  Saved to: {self.run_manager.artifacts_dir}\n")

        import pickle

        with open(self.run_manager.artifacts_dir / "test_points.pkl", "wb") as f:
            pickle.dump(test_points, f)
        with open(self.run_manager.artifacts_dir / "reference_solutions.pkl", "wb") as f:
            pickle.dump(reference_solutions, f)

        return test_points, reference_solutions

    def _build_combinations(self) -> list[ExperimentCombination]:
        print("Step 3: Build method-model combinations")
        combinations: list[ExperimentCombination] = []

        for method in self.config.methods:
            for model in self.config.models:
                model_cfg, algorithm_cfg = self.domain.build_source_configs(model, method)
                combo = ExperimentCombination(
                    label=f"{method}_{model}",
                    model_config=model_cfg,
                    algorithm_config=algorithm_cfg,
                )
                combinations.append(combo)
                print(f"  {combo.label}")

        print(f"  Total: {len(combinations)} combinations\n")
        return combinations

    def _train_all(self, combinations: list[ExperimentCombination]) -> Dict[str, list]:
        print("Step 5: Train all combinations")

        metrics_by_method: Dict[str, list] = {}

        for index, combo in enumerate(combinations, 1):
            print(f"  [{index}/{len(combinations)}] {combo.label}...", end=" ", flush=True)
            start_time = time.time()

            try:
                metrics = self._train_single(combo)
                metrics_by_method[combo.label] = metrics
                elapsed = time.time() - start_time
                print(f"✓ ({elapsed:.1f}s)")
            except Exception as exc:
                print(f"✗ Failed: {exc}")

        print()
        return metrics_by_method

    def _train_single(self, combo: ExperimentCombination) -> list:
        assert combo.model_config is not None, "Model config not built"
        assert combo.algorithm_config is not None, "Algorithm config not built"
        assert self.config.training is not None, "Train config not built"
        assert self.config.integration is not None, "Integration config not built"

        print(combo.model_config)
        _, metrics = run_training(
            algorithm_cfg=combo.algorithm_config,
            integration_cfg=self.config.integration,
            model_cfg=combo.model_config,
            train_cfg=self.config.training,
            sample_input=self.sample_input,
        )
        return metrics

    def _evaluate_and_visualise(
        self,
        test_points: np.ndarray,
        reference_solutions: np.ndarray,
        metrics_by_method: Dict[str, list],
    ) -> None:
        print("Step 6: Evaluate and visualise")

        assert self.run_manager, "No run manager found"

        if metrics_by_method:
            convergence_path = self.run_manager.plots_dir / "convergence.png"
            self.domain.plot_domain_specific(
                data={"plot_type": "convergence", "metrics_by_method": metrics_by_method},
                output_path=str(convergence_path),
            )

        print(f"  Results saved to {self.run_manager.plots_dir}\n")

    def print_summary(self) -> None:
        if not self.run_manager:
            print("No run executed yet")
            return

        metadata = self.run_manager.get_metadata()
        print("\nExperiment Summary:")
        print(f"  Run ID: {metadata['run_id']}")
        print(f"  Domain: {metadata['domain']}")
        print(f"  Timestamp: {metadata['timestamp']}")
        if metadata.get("git_commit"):
            print(f"  Git commit: {metadata['git_commit'][:8]}")
