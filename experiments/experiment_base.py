"""
Base experiment class: implements the common 6-step pipeline.

Experiments inherit from this class and provide:
1. Problem configuration (via config)
2. Domain plugin (for analytical solutions, test data, visualization)

The base class handles:
- Setup: Load configs and instantiate domain
- Prepare test data: Get/cache analytical solutions
- Build combinations: Cartesian product of methods × models
- Configure training: Merge training settings
- Train: Run training loop for each combination
- Evaluate: Compute metrics and visualize results
"""

import sys
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
import numpy as np
import time
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config_loader import ConfigLoader
from config.schema import ExperimentConfig, ExperimentCombination
from domains import get_domain, DomainPlugin
from results_manager import ResultsManager, RunManager

# Import from src (these should already exist)
from src.cli import run_training
from src.models import build_model
from src.loss_functions import (
    PINNConfig, SLSConfig, gPINNConfig, AlgorithmConfig, Loss
)
from src.integration import MonteCarloConfig


class BaseExperiment:
    """Base class for all experiments.

    Subclasses should override:
    - get_config_file(): return name of experiment YAML config
    - (optional) customize_domain(): modify domain after initialisation
    """

    def __init__(self, config: Optional[ExperimentConfig] = None):
        """Initialize experiment.

        Args:
            config: ExperimentConfig. If None, loads from get_config_file()
        """
        self.config = config or self._load_config()
        self.domain = get_domain(self.config.domain)(**self.config.problem_params)
        self.results_manager = ResultsManager()
        self.run_manager: Optional[RunManager] = None

        print(f"\n{'='*60}")
        print(f"Experiment: {self.config.name}")
        print(f"Domain: {self.domain.name} - {self.domain.description}")
        print(f"{'='*60}\n")

    def _load_config(self) -> ExperimentConfig:
        """Load experiment configuration. Override to customize.

        Returns:
            ExperimentConfig
        """
        loader = ConfigLoader("experiments/config")
        config_file = self.get_config_file()
        return loader.load(config_file)

    def get_config_file(self) -> str:
        """Return name of YAML config file. Override in subclass.

        Returns:
            Config filename (relative to experiments/config/)
        """
        return "experiment_1.yaml"

    def execute(self, run_id: Optional[str] = None) -> RunManager:
        """Execute the full 6-step experiment pipeline.

        Args:
            run_id: Optional explicit run ID; generates timestamp if not provided

        Returns:
            RunManager with results
        """
        # Step 0: Create run directory
        self.run_manager = self.results_manager.create_run(
            config=self.config.__dict__,
            domain=self.config.domain,
            run_id=run_id,
        )
        print(f"Run ID: {self.run_manager.run_id}\n")

        # Step 1: Setup
        self._setup()

        # Step 2: Prepare test data
        test_points, reference_solutions = self._prepare_test_data()

        # Step 3: Build combinations
        combinations = self._build_combinations()

        # Step 4: Configure training
        # (merged with individual combination configs during training)

        # Step 5: Train all combinations
        metrics_by_method = self._train_all(combinations)

        # Step 6: Evaluate and visualise
        self._evaluate_and_visualise(
            test_points=test_points,
            reference_solutions=reference_solutions,
            metrics_by_method=metrics_by_method,
        )

        print(f"\nExperiment complete. Results saved to {self.run_manager.run_dir}")

        return self.run_manager

    def _setup(self) -> None:
        """Step 1: Setup phase. Override to customize."""
        print("Step 1: Setup")
        self.config.validate()
        print("  Configuration validated")
        print(f"  Methods: {[m.name for m in self.config.methods]}")
        print(f"  Models: {[m.name for m in self.config.models]}")
        print()

    def _prepare_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Step 2: Prepare test data using domain plugin."""
        print("Step 2: Prepare test data")

        test_params = self.config.test_data_params or {}
        n_time = test_params.get("n_time", 20)
        n_space = test_params.get("n_space", 80)
        seed = test_params.get("seed", 42)

        test_points, reference_solutions = self.domain.get_test_data(
            n_time=n_time,
            n_space=n_space,
            seed=seed,
        )

        print(f"  Test data: {len(test_points)} points")

        assert self.run_manager is not None, "No run Manager available."
        print(f"  Saved to: {self.run_manager.artifacts_dir}\n")

        # Save test data
        import pickle
        with open(self.run_manager.artifacts_dir / "test_points.pkl", "wb") as f:
            pickle.dump(test_points, f)
        with open(self.run_manager.artifacts_dir / "reference_solutions.pkl", "wb") as f:
            pickle.dump(reference_solutions, f)

        return test_points, reference_solutions

    def _build_combinations(self) -> list[ExperimentCombination]:
        """Step 3: Build Cartesian product of methods × models."""
        print("Step 3: Build method-model combinations")
        combinations = []

        for method in self.config.methods:
            for model in self.config.models:
                label = f"{method.name}_{model.name}"
                combo = ExperimentCombination(
                    method=method,
                    model=model,
                    label=label,
                )
                combinations.append(combo)
                print(f"  {label}")

        print(f"  Total: {len(combinations)} combinations\n")
        return combinations

    def _train_all(
        self,
        combinations: list[ExperimentCombination],
    ) -> Dict[str, list]:
        """Step 5: Train all combinations sequentially."""
        print("Step 5: Train all combinations")

        metrics_by_method = {}

        for i, combo in enumerate(combinations, 1):
            print(f"  [{i}/{len(combinations)}] {combo.label}...", end=" ", flush=True)
            start_time = time.time()

            try:
                # TODO: This is a placeholder; actual training logic integrates with src.cli.run_training
                metrics = self._train_single(combo)
                metrics_by_method[combo.label] = metrics
                elapsed = time.time() - start_time
                print(f"✓ ({elapsed:.1f}s)")
            except Exception as e:
                print(f"✗ Failed: {e}")
                # Continue with next combination

        print()
        return metrics_by_method

    def _train_single(self, combo: ExperimentCombination) -> list:
        """Train a single method-model combination.

        Returns:
            List of TrainStepMetrics for each epoch
        """
        # TODO: Placeholder: actual implementation integrates with src.cli.run_training
        #       For now, return dummy metrics

        state, metrics = run_training(
            algorithm_cfg=combo.algorithm_config,
            integration_cfg=self.config.integration,
            model_cfg=combo.model_config,
            train_cfg=train_config,
            sample_input=sample_input,
        )
        return []

    def _evaluate_and_visualise(
        self,
        test_points: np.ndarray,
        reference_solutions: np.ndarray,
        metrics_by_method: Dict[str, list],
    ) -> None:
        """Step 6: Evaluate and visualise results."""
        print("Step 6: Evaluate and visualise")

        assert self.run_manager, "No run manager found"

        # Create convergence plot
        if metrics_by_method:
            convergence_path = self.run_manager.plots_dir / "convergence.png"
            self.domain.plot_domain_specific(
                data={"plot_type": "convergence", "metrics_by_method": metrics_by_method},
                output_path=str(convergence_path),
            )

        print(f"  Results saved to {self.run_manager.plots_dir}\n")

    def print_summary(self) -> None:
        """Print experiment summary."""
        if not self.run_manager:
            print("No run executed yet")
            return

        metadata = self.run_manager.get_metadata()
        print(f"\nExperiment Summary:")
        print(f"  Run ID: {metadata['run_id']}")
        print(f"  Domain: {metadata['domain']}")
        print(f"  Timestamp: {metadata['timestamp']}")
        if metadata.get("git_commit"):
            print(f"  Git commit: {metadata['git_commit'][:8]}")
