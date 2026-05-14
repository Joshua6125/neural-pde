"""Tests for experiment config loading and results persistence."""

from pathlib import Path
import json

from experiments.config_loader import ConfigLoader
from experiments.results_manager import ResultsManager

from src.integration import MonteCarloConfig


DEFAULTS_YAML = """\
name: default_experiment
domain: simple_wave_equation
problem_params:
  L: 1.0
  T: 1.0
  c: 1.0
training:
  epochs: 100
  learning_rate:
    kind: constant
    value: 1.0e-3
  seed: 42
  log_every: 50
  optimiser: adamw
integration:
  strategy: monte_carlo
  dim: 2
  x_min: 0.0
  x_max: 1.0
  monte_carlo:
    monte_carlo_interior_samples: 1600
    monte_carlo_boundary_samples: 1000
    monte_carlo_seed: 42
methods:
  - name: pinn
    output_heads:
      u: 1
models:
  - name: mlp
    hidden_dim: 32
    num_layers: 3
"""

EXPERIMENT_YAML = """\
name: experiment_1
domain: simple_wave_equation
problem_params:
  L: 2.0
training:
  learning_rate:
    kind: exponential_decay
    init_value: 1.0e-3
    transition_steps: 100
    decay_rate: 0.9
    staircase: true
integration:
  monte_carlo:
    monte_carlo_interior_samples: 3200
"""


def _write_config_files(config_dir: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "defaults.yaml").write_text(DEFAULTS_YAML)
    (config_dir / "experiment_1.yaml").write_text(EXPERIMENT_YAML)


class TestConfigLoader:
    def test_deep_merge_preserves_nested_defaults_and_source_spec(self, tmp_path):
        config_dir = tmp_path / "config"
        _write_config_files(config_dir)

        loader = ConfigLoader(str(config_dir))
        config = loader.load("experiment_1.yaml")

        assert config.problem_params["L"] == 2.0
        assert config.problem_params["T"] == 1.0
        assert isinstance(config.integration, MonteCarloConfig)
        assert config.integration.monte_carlo_interior_samples == 3200
        assert config.integration.monte_carlo_boundary_samples == 1000
        assert config.source_spec["problem_params"]["L"] == 2.0
        assert config.source_spec["training"]["learning_rate"]["kind"] == "exponential_decay"
        assert config.source_spec["integration"]["monte_carlo"]["monte_carlo_interior_samples"] == 3200


class TestResultsManager:
    def test_create_run_stores_spec_and_resolved_config(self, tmp_path):
        config_dir = tmp_path / "config"
        _write_config_files(config_dir)

        loader = ConfigLoader(str(config_dir))
        config = loader.load("experiment_1.yaml")
        original_learning_rate = config.training.learning_rate

        results_root = tmp_path / "results"
        manager = ResultsManager(str(results_root))
        run_manager = manager.create_run(config=config, domain=config.domain, run_id="run-1")

        assert run_manager.run_id == "run-1"
        assert config.training.learning_rate is original_learning_rate

        metadata = json.loads((results_root / "run-1" / "metadata.json").read_text())
        resolved_config = json.loads((results_root / "run-1" / "config_resolved.json").read_text())
        source_spec = json.loads((results_root / "run-1" / "config_spec.json").read_text())

        assert metadata["config_spec"]["training"]["learning_rate"]["kind"] == "exponential_decay"
        assert resolved_config["training"]["learning_rate"]["kind"] == "exponential_decay"
        assert source_spec["problem_params"]["L"] == 2.0
