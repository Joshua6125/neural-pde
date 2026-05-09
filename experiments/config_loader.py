"""
Configuration loader: loads YAML configs and merges with defaults.

Usage:
    config = ConfigLoader("config/experiment_1.yaml").load()

    # With Python overrides:
    config = ConfigLoader("config/experiment_1.yaml").load(
        overrides={"training.epochs": 200, "methods[0].name": "pinn"}
    )
"""

import yaml
from pathlib import Path
from typing import Any, Optional, Dict
from config.schema import (
    ExperimentConfig,
    IntegrationConfig,
    TrainingConfig,
    ModelConfig,
    MethodConfig,
)


class ConfigLoader:
    """Load and merge experiment configurations."""

    def __init__(self, config_dir: str = "experiments/config"):
        """Initialize loader with config directory.

        Args:
            config_dir: Directory containing YAML config files
        """
        self.config_dir = Path(config_dir)
        self.defaults_path = self.config_dir / "defaults.yaml"

        if not self.defaults_path.exists():
            raise FileNotFoundError(f"defaults.yaml not found in {self.config_dir}")

    def load(
        self,
        config_file: str = "experiment_1.yaml",
        overrides: Optional[Dict[str, Any]] = None,
    ) -> ExperimentConfig:
        """Load and merge configuration.

        Args:
            config_file: Name of experiment config (relative to config_dir)
            overrides: Dict of dot-notation overrides (e.g., {"training.epochs": 200})

        Returns:
            Merged ExperimentConfig

        Raises:
            FileNotFoundError: If config files don't exist
            ValueError: If configuration is invalid
        """
        # Load defaults
        with open(self.defaults_path, "r") as f:
            defaults = yaml.safe_load(f) or {}

        # Load experiment config
        experiment_path = self.config_dir / config_file
        if not experiment_path.exists():
            raise FileNotFoundError(f"Config file not found: {experiment_path}")

        with open(experiment_path, "r") as f:
            experiment_yaml = yaml.safe_load(f) or {}

        # Merge: experiment overrides defaults
        merged = {**defaults, **experiment_yaml}

        # Apply programmatic overrides
        if overrides:
            merged = self._apply_overrides(merged, overrides)

        # Convert to dataclass
        config = self._build_config(merged)
        config.validate()

        return config

    @staticmethod
    def _apply_overrides(config: dict, overrides: dict) -> dict:
        """Apply dot-notation overrides to config dict.

        Examples:
            "training.epochs" -> config["training"]["epochs"]
            "methods[0].name" -> config["methods"][0]["name"]
        """
        for key_path, value in overrides.items():
            keys = key_path.split(".")
            current = config

            # Navigate to parent
            for key in keys[:-1]:
                # Handle list indexing: "methods[0]" -> "methods", 0
                if "[" in key:
                    list_name, index_str = key.split("[")
                    index = int(index_str.rstrip("]"))
                    current = current[list_name][index]
                else:
                    if key not in current:
                        current[key] = {}
                    current = current[key]

            # Set final value
            final_key = keys[-1]
            if "[" in final_key:
                list_name, index_str = final_key.split("[")
                index = int(index_str.rstrip("]"))
                current[list_name][index] = value
            else:
                current[final_key] = value

        return config

    @staticmethod
    def _build_config(data: dict) -> ExperimentConfig:
        """Build ExperimentConfig from dict."""
        return ExperimentConfig(
            name=data.get("name", "unnamed"),
            domain=data.get("domain", ""),
            problem_params=data.get("problem_params", {}),
            methods=[
                MethodConfig(
                    name=m.get("name", ""),
                    extra_params={k: v for k, v in m.items() if k != "name"},
                )
                for m in data.get("methods", [])
            ],
            models=[
                ModelConfig(
                    name=m.get("name", ""),
                    hidden_dim=m.get("hidden_dim", 32),
                    num_layers=m.get("num_layers", 3),
                    extra_params={
                        k: v for k, v in m.items()
                        if k not in ["name", "hidden_dim", "num_layers"]
                    },
                )
                for m in data.get("models", [])
            ],
            training=TrainingConfig(
                epochs=data.get("training", {}).get("epochs", 100),
                learning_rate=data.get("training", {}).get("learning_rate", 1e-3),
                seed=data.get("training", {}).get("seed", 42),
                optimiser=data.get("training", {}).get("optimizer", "adamw"),
            ),
            integration=IntegrationConfig(
                strategy=data.get("integration", {}).get("strategy", "monte_carlo"),
                n_interior=data.get("integration", {}).get("n_interior", 1600),
                n_boundary=data.get("integration", {}).get("n_boundary", 100),
                seed=data.get("integration", {}).get("seed", 42),
            ),
            test_data_params=data.get("test_data_params", {}),
        )
